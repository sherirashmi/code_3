"""Generate the 7 block datasets of the position-only model bank.

Each block: 10,000 configurations of NUM_RES identical resonators (2 by default,
``--num-res 3`` for three) with m = 0.2 kg and one tuning frequency f_t in
{40, 50, ..., 100} Hz (k = m (2 pi f_t)^2); only the positions vary (Latin hypercube, x in [0.05, 1.35] m,
y in [0.05, 0.45] m); 6 x 3 = 18 plate modes. Saved as
datasets/erp/blocks_<N>res/dataset_erp_10k_<N>res_fixed_m0p2_ft<f_t>_18modes.pth, registered as
"10k_<N>res_fixed_m0.2_ft<f_t>_18modes". Each block is split 80/10/10 into
training / validation / test data when it is loaded (seeded, as every dataset).

Run from the repository root:  python datasets/scripts/generate_fixed_blocks.py [--num-res 3]
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

MASS = 0.2
NUM_RES = 2
NUM_CONFIGS = 10_000
MODES = (6, 3)
BASE_SEED = 1000


def _generate_block(args) -> str:
    f_t, num_res = args
    import utils.physics as physics
    from utils.erp_dataset import DATASETS, ERPDataset, fixed_block_tag

    physics.set_modal_resolution(*MODES, verbose=False)
    filename = DATASETS[fixed_block_tag(f_t, MASS, num_res)]["files"][0]
    t0 = time.perf_counter()
    dataset = ERPDataset(num_samples=NUM_CONFIGS, num_res=num_res, seed=BASE_SEED + 100 * (num_res - 2) + int(f_t),
                         fixed_resonator=(MASS, f_t))
    dataset.generate(save=True, filename=filename, verbose=False)
    print(f"[f_t = {f_t:g} Hz] {NUM_CONFIGS:,} configs -> {filename} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return filename


def main() -> None:
    import argparse

    from utils.erp_dataset import FIXED_BLOCK_FREQUENCIES

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-res", type=int, default=NUM_RES)
    num_res = parser.parse_args().num_res

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(len(FIXED_BLOCK_FREQUENCIES), os.cpu_count() or 1),
                             mp_context=get_context("spawn")) as pool:
        list(pool.map(_generate_block, [(f, num_res) for f in FIXED_BLOCK_FREQUENCIES]))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
