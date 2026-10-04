"""Train iFNO / iDCO / iGNO on a registered dataset with the recommended options.

Sets the solver's modal resolution for the dataset first (so the solver
checks in the evaluation use the same plate-mode basis as the data), then
runs train.main with RECOMMENDED_OPTIONS (bounded design, bounded gate,
binned readout, padded FFT for iFNO, cycle/alignment terms, stage-2 VAE on
stage-1 estimates). Interrupted runs resume from <model>.resume.pt.

Usage (from the repository root):
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes          # iFNO, iDCO, iGNO
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes 1        # iFNO only
"""

from __future__ import annotations

import sys

from utils.erp_dataset import DATASETS, select_dataset_modal_resolution

from .train import RECOMMENDED_OPTIONS, main

if __name__ == "__main__":
    tag = sys.argv[1]
    keys = tuple(sys.argv[2:]) or ("1", "2", "3")
    spec = DATASETS[tag]
    select_dataset_modal_resolution(tag)
    files = list(spec["files"])
    main(keys=keys, dataset_file=files if len(files) > 1 else files[0],
         num_configurations=int(spec["num_configurations"]), options=dict(RECOMMENDED_OPTIONS))
    print("ALL DONE", flush=True)
