"""Generate the 7 block datasets of the position-only model bank.

Each block: 10,000 configurations of 2 identical resonators with m = 0.2 kg and
one tuning frequency f_t in {40, 50, ..., 100} Hz (k = m (2 pi f_t)^2); only the
positions x1, y1, x2, y2 vary (Latin hypercube, x in [0.05, 1.35] m,
y in [0.05, 0.45] m); 6 x 3 = 18 plate modes. Saved as
datasets/dataset_erp_10k_2res_fixed_m0p2_ft<f_t>_18modes.pth, registered as
"10k_2res_fixed_m0.2_ft<f_t>_18modes". Each block is split 80/10/10 into
training / validation / test data when it is loaded (seeded, as every dataset).

Run from the repository root:  python datasets/generate_fixed_blocks.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

MASS = 0.2
NUM_RES = 2
NUM_CONFIGS = 10_000
MODES = (6, 3)
BASE_SEED = 1000


def _generate_block(f_t: float) -> str:
    import utils.physics as physics
    from utils.erp_dataset import DATASETS, ERPDataset, fixed_block_tag

    physics.set_modal_resolution(*MODES, verbose=False)
    filename = DATASETS[fixed_block_tag(f_t, MASS)]["files"][0]
    t0 = time.perf_counter()
    dataset = ERPDataset(num_samples=NUM_CONFIGS, num_res=NUM_RES, seed=BASE_SEED + int(f_t),
                         fixed_resonator=(MASS, f_t))
    dataset.generate(save=True, filename=filename, verbose=False)
    print(f"[f_t = {f_t:g} Hz] {NUM_CONFIGS:,} configs -> {filename} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return filename


def main() -> None:
    from utils.erp_dataset import FIXED_BLOCK_FREQUENCIES

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(len(FIXED_BLOCK_FREQUENCIES), os.cpu_count() or 1),
                             mp_context=get_context("spawn")) as pool:
        list(pool.map(_generate_block, FIXED_BLOCK_FREQUENCIES))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
