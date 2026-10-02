"""Generate the 100k "2 resonators on a 14 x 5 grid, 18 plate modes" dataset.

* 2 resonators per configuration.
* Positions only on the 14 x 5 equidistant grid (x = 0.05..1.35 m,
  y = 0.05..0.45 m, 0.1 m spacing): two distinct cells per configuration,
  each of the 2,415 cell pairs used 41-42 times, resonator order random.
* m in [0.1, 1.0] kg and f_t in [10, 160] Hz by Latin hypercube; k derived.
* Plate basis 6 x 3 = 18 modes.

Written as 4 shards of 25,000 configurations (each well below GitHub's
100 MB file limit), registered as dataset "100k_2res_grid_18modes".

Run from the repository root:  python datasets/generate_grid_2res_18modes.py
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
GRID = (14, 5)
MODES = (6, 3)
NUM_SHARDS = 4
TOTAL = 100_000
BASE_SEED = 727
TAG = "100k_2res_grid_18modes"


def _generate_shard(args) -> str:
    index, cells = args
    import utils.physics as physics
    from utils.erp_dataset import DATASETS, ERPDataset

    physics.set_modal_resolution(*MODES, verbose=False)
    filename = DATASETS[TAG]["files"][index]
    t0 = time.perf_counter()
    dataset = ERPDataset(num_samples=len(cells), num_res=NUM_RES, seed=BASE_SEED + index, position_grid=GRID)
    dataset.generate(save=True, filename=filename, verbose=False, grid_cells=cells)
    print(f"[shard {index + 1}/{NUM_SHARDS}] {len(cells):,} configs -> {filename} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return filename


def main() -> None:
    import numpy as np
    from utils.erp_dataset import balanced_grid_cells

    cells = balanced_grid_cells(TOTAL, NUM_RES, *GRID, seed=BASE_SEED)  # balanced over all 100k
    chunks = np.array_split(cells, NUM_SHARDS)
    print(f"Generating {TOTAL:,} configurations: {NUM_RES} resonators on a {GRID[0]}x{GRID[1]} grid, "
          f"{MODES[0]}x{MODES[1]} plate modes, {NUM_SHARDS} shards")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(NUM_SHARDS, os.cpu_count() or 1), mp_context=get_context("spawn")) as pool:
        list(pool.map(_generate_shard, list(enumerate(chunks))))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
