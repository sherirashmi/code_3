"""Train iFNO / iDCO / iGNO on a registered dataset with the recommended options.

Sets the solver's modal resolution for the dataset first (so the solver
checks in the evaluation use the same plate-mode basis as the data), then
runs train.main with RECOMMENDED_OPTIONS (bounded design, bounded gate,
binned readout, padded FFT for iFNO, cycle/alignment terms, stage-2 VAE on
stage-1 estimates). Interrupted runs resume from <model>.resume.pt.

Usage (from the repository root):
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes          # iFNO, iDCO, iGNO
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes 1        # iFNO only
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes --sorted --physical
        # f_t-sorted resonator encoder (iFNO/iDCO) + plate mode shapes / physical detuning
"""

from __future__ import annotations

import sys

from utils.erp_dataset import DATASETS, select_dataset_modal_resolution

from .train import RECOMMENDED_OPTIONS, main

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    tag = args[0]
    keys = tuple(args[1:]) or ("1", "2", "3")
    options = dict(RECOMMENDED_OPTIONS)
    if "--sorted" in flags:
        options["encoder"] = "sorted"
    if "--physical" in flags:
        options["coordinate_features"] = "physical"
    spec = DATASETS[tag]
    select_dataset_modal_resolution(tag)
    files = list(spec["files"])
    main(keys=keys, dataset_file=files if len(files) > 1 else files[0],
         num_configurations=int(spec["num_configurations"]), options=options)
    print("ALL DONE", flush=True)
