"""Interruption-safe forward-operator training for Google Colab / Kaggle.

Usage (one notebook cell, after cloning the repo)::

    !python cloud_train.py --out /content/drive/MyDrive/thesis_runs \
        --models DNO,GNO,STO --physical

What it does:

1. Moves ``erp_forward_operators/models`` and ``.../plots`` into ``--out``
   (a Google Drive folder on Colab, ``/kaggle/working/...`` on Kaggle) and
   leaves symlinks in their place, so EVERY checkpoint, progress file and
   plot is written straight to persistent storage. Files already in
   ``--out`` (from an earlier, interrupted run) are kept, never overwritten.
2. Trains the selected models on the chosen dataset. Every 5 epochs each
   model's full training state is saved as ``<model>.resume.pt`` plus
   ``<model>_progress.csv/.png`` (loss so far) next to the checkpoints.
3. If the session dies, run the same command again: finished models are
   evaluated instead of retrained, and the interrupted one continues from
   its last saved epoch.
4. Finally re-scores every checkpoint of the dataset (error breakdown).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _copy_missing(src: Path, dst: Path) -> None:
    """Copy src into dst, skipping every file that already exists in dst."""
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def link_outputs(out: Path) -> None:
    for sub in ("models", "plots"):
        local = ROOT / "erp_forward_operators" / sub
        remote = out / sub
        remote.mkdir(parents=True, exist_ok=True)
        if local.is_symlink():
            if local.resolve() != remote.resolve():
                local.unlink()
                local.symlink_to(remote, target_is_directory=True)
        else:
            if local.exists():
                _copy_missing(local, remote)
                shutil.rmtree(local)
            local.symlink_to(remote, target_is_directory=True)
        print(f"{local} -> {remote}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="persistent folder for models + plots")
    parser.add_argument("--dataset", default="100k", help="10k | 100k | 200k_18modes (default 100k)")
    parser.add_argument("--models", default="DNO,GNO,STO", help="comma-separated: DON,DNO,FNO,DCO,GNO,STO,SIREN,WNO,NN,LNO")
    parser.add_argument("--physical", action="store_true", help="physical feature scaling (_phys)")
    parser.add_argument("--sorted", action="store_true", help="f_t-sorted encoder branch (_sorted)")
    parser.add_argument("--fno-padding", default="replicate", choices=("replicate", "reflect", "zero"))
    parser.add_argument("--nn-permutation", action="store_true", help="NN resonator-order augmentation (_perm)")
    parser.add_argument("--epochs", type=int, default=None, help="override every model's default (200)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--configurations", type=int, default=None, help="default: whole dataset")
    args = parser.parse_args()

    link_outputs(Path(args.out))

    import torch
    from erp_forward_operators import diagnose
    from erp_forward_operators.operator_registry import OPERATORS
    from utils.cli import train_all_models
    from utils.erp_dataset import DATASETS, select_dataset_modal_resolution

    print("CUDA available:", torch.cuda.is_available(),
          f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "-- running on CPU")
    by_short = {spec["short"]: spec for spec in OPERATORS.values()}
    wanted = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in by_short]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {sorted(by_short)}")

    select_dataset_modal_resolution(args.dataset)
    files = DATASETS[args.dataset]["files"]
    train_all_models(
        operator_specs=[by_short[m] for m in wanted],
        num_configurations=args.configurations or int(DATASETS[args.dataset]["num_configurations"]),
        batch_size=args.batch_size,
        dataset_file=files if len(files) > 1 else files[0],
        epochs_override=args.epochs,
        forward_options={
            "sorted": args.sorted,
            "physical": args.physical,
            "padding_mode": args.fno_padding,
            "permutation": args.nn_permutation,
        },
        skip_existing=True,
    )
    diagnose.main(args.dataset)


if __name__ == "__main__":
    main()
