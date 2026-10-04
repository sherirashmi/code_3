"""Generate the 200k "2 resonators, everything varies, 18 plate modes" dataset.

* 2 resonators per configuration.
* m in [0.1, 1.0] kg, f_t in [10, 160] Hz, x in [0.05, 1.35] m, y in
  [0.05, 0.45] m: one Latin hypercube over all 200k configurations (8 numbers
  each); stiffness derived, k = m (2 pi f_t)^2.
* Plate basis 6 x 3 = 18 modes.

Written as 8 shards of 25,000 configurations, registered as "200k_2res_18modes".

Run from the repository root:  python datasets/generate_2res_18modes.py
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

NUM_RES = 2
MODES = (6, 3)
NUM_SHARDS = 8
TOTAL = 200_000
BASE_SEED = 2002
TAG = "200k_2res_18modes"


def _generate_shard(args) -> str:
    index, samples = args
    import utils.physics as physics
    from utils.erp_dataset import DATASETS, ERPDataset

    physics.set_modal_resolution(*MODES, verbose=False)
    filename = DATASETS[TAG]["files"][index]
    t0 = time.perf_counter()
    dataset = ERPDataset(num_samples=len(samples), num_res=NUM_RES, seed=BASE_SEED + index)
    dataset.generate(save=True, filename=filename, verbose=False, samples=samples)
    print(f"[shard {index + 1}/{NUM_SHARDS}] {len(samples):,} configs -> {filename} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return filename


def main() -> None:
    import numpy as np
    from utils.physics import resonator_bounds
    from utils.support import lhs_sampling

    samples = lhs_sampling(TOTAL, bounds=resonator_bounds(NUM_RES), seed=BASE_SEED)  # one LHS over all 200k
    chunks = np.array_split(samples, NUM_SHARDS)
    print(f"Generating {TOTAL:,} configurations: {NUM_RES} resonators, m, f_t, x, y by LHS, "
          f"{MODES[0]}x{MODES[1]} plate modes, {NUM_SHARDS} shards")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(NUM_SHARDS, os.cpu_count() or 1), mp_context=get_context("spawn")) as pool:
        list(pool.map(_generate_shard, list(enumerate(chunks))))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
