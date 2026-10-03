# Dataset analysis

Plots, statistics and videos, grouped by the dataset they describe. Every dataset folder has
`plots/` (figures), `stats/` (the numbers behind them, CSV/TXT) and `videos/` (GIFs). Dataset
folder names are the tags of `utils/erp_dataset.py::DATASETS`. Scripts are in `scripts/`; run
them from the repository root and they write into the folder of their dataset.

| Folder | Dataset | Contents |
|---|---|---|
| `10k/` | `datasets/dataset_erp_ft.pth`: 10,000 configurations, 3 resonators, 150 plate modes | `plots/repo_mass_ft_histograms.png` (tuning-frequency and derived-mass histograms), `videos/erp_dataset_configs.gif` (80 random configurations: ERP spectrum and plate displacement) |
| `100k/` | 100,000 configurations, 3 resonators with varying mass, tuning frequency and position, 150 plate modes (two shard files) | `plots/`: `dataset_100k_statistics.png`, `erp_frequency_statistics.png`, `erp_frequency_boxplot.png`, `parameter_split_distributions.png`, `field_sensitivity.png`, `similar_erp_pairs.png`, `similar_erp_clusters.png`; `stats/`: `erp_frequency_statistics.csv`, `erp_frequency_boxplot.csv`; `videos/erp_dataset_configs.gif` |
| `100k_2res_fixed_m0.2_ft72_18modes/` | 100,000 configurations, 2 identical resonators (0.2 kg, tuning frequency 72 Hz), only positions vary, 18 plate modes | `plots/erp_frequency_boxplot.png`, `stats/erp_frequency_boxplot.csv`, `videos/erp_dataset_configs.gif` |
| `legacy_5k_peak_analysis/` | An earlier 5,000-configuration dataset (peak-count and peak-spacing analysis from `others/analyze_erp_dataset.py`) | `stats/peak_analysis_summary.txt`, `stats/per_configuration_peak_analysis.csv` |
| `sampling_studies/` | Not a stored dataset: how different sampling schemes (Latin hypercube over tuning frequency and mass, or over mass and stiffness) distribute the tuning frequency | `plots/ft_histogram_uniform.png`, `plots/ft_histogram_mk_lhs.png` |
| `solver_demos/` | No dataset: demonstrations of the plate solver (bare plate, one resonator moved or retuned, parameter sweeps) | `plots/erp_bare_plate.png`; `videos/`: `erp_x_sweep`, `erp_y_sweep`, `erp_y_sweep_maxdisp`, `erp_ft_sweep_1_resonator`, `erp_sweep_3_resonators`, `video1`-`video4` (mass / stiffness / tuning-frequency sweeps), `erp_config_noise_sensitivity.mp4` |

## Scripts

| Script | Output |
|---|---|
| `plot_erp_frequency_boxplot.py [tag]` | `<tag>/plots/erp_frequency_boxplot.png`, `<tag>/stats/erp_frequency_boxplot.csv` |
| `make_fixed_resonator_dataset_video.py [tag]` | `<tag>/videos/erp_dataset_configs.gif` (any dataset with a fixed number of resonators) |
| `make_dataset_video.py` | `10k/videos/erp_dataset_configs.gif` |
| `plot_erp_frequency_stats.py`, `plot_parameter_split_distributions.py`, `field_sensitivity_analysis.py`, `find_similar_erp_configs.py` | `100k/plots/`, `100k/stats/` |
| `make_position_sweep_videos.py`, `make_y_sweep_maxdisp.py`, `make_ft_sweep_video.py`, `make_erp_video.py`, `erp_param_videos.py`, `make_config_noise_video.py` | `solver_demos/videos/` |

`gif_common.py` holds the shared plot style (thesis fonts, labels, the displacement heatmap).
Figures use the thesis Computer Modern look from `utils/plot_style.py`; older figures
(`dataset_100k_statistics.png`, the histograms and `others/`-generated stats) predate it.
