"""Generate a 1,000,000-configuration ERP dataset at reduced (6x3=18) modal
resolution, sharded the same way the original 100k dataset is (50,000
configurations per shard -- here, 20 shards instead of 2).

Requires ``utils.physics.Nx/Ny`` to already be set to 6/3 on disk before this
script is launched (each worker process imports ``utils.physics`` fresh, so
whatever Nx/Ny the file holds at worker-spawn time is what gets baked into
every shard it generates). This script does not edit physics.py itself.

Each shard's resonator configurations come from an independent LHS draw
(seed = 727 + shard_index), so shards don't duplicate each other's samples,
matching how ``prepare_operator_data``/``prepare_inverse_data`` already load
multiple shard files into one combined dataset via ``ERPDataset.load_shards``.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NUM_SHARDS = 20
CONFIGS_PER_SHARD = 50_000
BASE_SEED = 727
OUT_PATTERN = "datasets/dataset_erp_ft_100k_18_modes_part{index}.pth"

SPAWN_CONTEXT = get_context("spawn")


def _generate_shard(shard_index: int) -> str:
    from utils.erp_dataset import ERPDataset
    from utils.physics import N

    seed = BASE_SEED + shard_index
    filename = OUT_PATTERN.format(index=shard_index + 1)
    t0 = time.perf_counter()
    dataset = ERPDataset(num_samples=CONFIGS_PER_SHARD, num_res=3, seed=seed)
    dataset.generate(save=True, filename=filename, verbose=False)
    elapsed = time.perf_counter() - t0
    print(
        f"[shard {shard_index + 1}/{NUM_SHARDS}] N_modes={N} seed={seed} "
        f"{CONFIGS_PER_SHARD} configs -> {filename} ({elapsed:.1f}s)",
        flush=True,
    )
    return filename


def main() -> None:
    from utils.physics import N, Nx, Ny

    print(f"Modal resolution: Nx={Nx}, Ny={Ny} -> N={N} plate modes")
    print(f"Generating {NUM_SHARDS * CONFIGS_PER_SHARD:,} total configurations "
          f"across {NUM_SHARDS} shards of {CONFIGS_PER_SHARD:,} each")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=SPAWN_CONTEXT) as pool:
        for filename in pool.map(_generate_shard, range(NUM_SHARDS)):
            pass
    elapsed = time.perf_counter() - t0
    print(f"\nDONE. Total elapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
