"""Train the invertible operators (iFNO, iDCO, iGNO, iDNO, iWNO, iLNO, iSIREN,
iSTO) on a registered dataset with the recommended options.

Sets the solver's modal resolution for the dataset first (so the solver
checks in the evaluation use the same plate-mode basis as the data), then
runs train.main with RECOMMENDED_OPTIONS (bounded design, bounded gate,
binned readout, padded FFT for iFNO, cycle/alignment terms, stage-2 VAE on
stage-1 estimates, f_t-sorted resonator encoder, plate mode shapes /
physical detuning). Interrupted runs resume from <model>.resume.pt.

Usage (from the repository root):
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes          # all eight
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes 1 4      # iFNO and iDNO
    keys: 1 iFNO, 2 iDCO, 3 iGNO, 4 iDNO, 5 iWNO, 6 iLNO, 7 iSIREN, 8 iSTO
    --set-encoder / --zscored: the earlier pooled set encoder / z-scored sine features
"""

from __future__ import annotations

import sys

from utils.erp_dataset import DATASETS, select_dataset_modal_resolution

from .registry import INVERTIBLE_OPERATORS
from .train import RECOMMENDED_OPTIONS, main

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    tag = args[0]
    keys = tuple(args[1:]) or tuple(INVERTIBLE_OPERATORS)
    options = dict(RECOMMENDED_OPTIONS)
    if "--set-encoder" in flags:
        options["encoder"] = "set"
    if "--zscored" in flags:
        options["coordinate_features"] = "zscored"
    spec = DATASETS[tag]
    select_dataset_modal_resolution(tag)
    files = list(spec["files"])
    main(keys=keys, dataset_file=files if len(files) > 1 else files[0],
         num_configurations=int(spec["num_configurations"]), options=options)
    print("ALL DONE", flush=True)
