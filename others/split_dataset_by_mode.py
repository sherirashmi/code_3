"""Split the master ERP dataset into a low-frequency and high-frequency
sub-dataset, divided at (first plate structural mode - 5 Hz).

Below the plate's own first bending mode there is no structural resonance
to interact with, so the ERP response there is smooth and similar across
resonator configurations -- easy for any neural operator to fit. At and
above the first mode, mode-splitting/veering against the resonators makes
the target far more configuration-sensitive. See
utils.erp_dataset.split_dataset_by_frequency for the full reasoning.

Run:
    python -m others.split_dataset_by_mode

Produces datasets/dataset_erp_ft_low.pth and datasets/dataset_erp_ft_high.pth
from the existing datasets/dataset_erp_ft.pth, sharing the same resonator
configurations and differing only in which frequency points each keeps.
Either output can be used as-is with dataset_file= on any operator's main()
or run_operator_experiment() -- ERPDataset reads its frequency range from
the file itself, so no other code needs to change to train/evaluate on one
band only.
"""

from utils.erp_dataset import split_dataset_by_frequency

if __name__ == "__main__":
    split_dataset_by_frequency()
