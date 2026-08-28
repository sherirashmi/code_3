import time

import dcor
import matplotlib.pyplot as plt
import numpy as np
import torch


# ==================================================
# Settings
# ==================================================

DATASET_FILE = "datasets/dataset_erp_ft.pth"

INPUT_NAMES = [
    "ft1", "x1", "y1",
    "ft2", "x2", "y2",
    "ft3", "x3", "y3",
]


# ==================================================
# Load dataset
# ==================================================

print("=" * 70)
print("Loading ERP dataset")
print("=" * 70)

data = torch.load(
    DATASET_FILE,
    map_location="cpu",
    weights_only=False,
)


# ==================================================
# Extract arrays
# ==================================================

# Shape:
# configuration_features -> (num_configurations, num_resonators, 3)
#
# Each resonator:
# [f_t, x, y]

features = np.asarray(
    data["configuration_features"],
    dtype=np.float64,
)

# Shape:
# responses -> (num_configurations, num_frequencies, 1)

erp = np.asarray(
    data["responses"],
    dtype=np.float64,
)[:, :, 0]

frequencies = np.asarray(
    data["frequency_values"],
    dtype=np.float64,
)


# ==================================================
# Flatten configuration inputs
# ==================================================

# For 3 resonators:
#
# [ft1, x1, y1,
#  ft2, x2, y2,
#  ft3, x3, y3]

X = features.reshape(features.shape[0], -1)

# Make arrays contiguous float64.
# This allows dcor to use its compiled implementation.

X = np.ascontiguousarray(
    X,
    dtype=np.float64,
)

erp = np.ascontiguousarray(
    erp,
    dtype=np.float64,
)

frequencies = np.ascontiguousarray(
    frequencies,
    dtype=np.float64,
)


# ==================================================
# Dataset information
# ==================================================

num_configurations = X.shape[0]
num_inputs = X.shape[1]
num_frequencies = erp.shape[1]

print(f"Configurations       : {num_configurations}")
print(f"Configuration inputs : {num_inputs}")
print(f"Frequency points     : {num_frequencies}")
print(f"Frequency range      : "
      f"{frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")

print("\nInput ordering:")
for i, name in enumerate(INPUT_NAMES):
    print(f"{i}: {name}")


# ==================================================
# Distance correlation
# ==================================================

print("\n" + "=" * 70)
print("Calculating distance correlation")
print("=" * 70)

distance_corr = np.zeros(
    (num_inputs, num_frequencies),
    dtype=np.float64,
)

start_time = time.perf_counter()

for input_idx in range(num_inputs):

    input_values = X[:, input_idx]

    print(
        f"Processing input "
        f"{input_idx + 1}/{num_inputs}: "
        f"{INPUT_NAMES[input_idx]}"
    )

    for freq_idx in range(num_frequencies):

        erp_values = erp[:, freq_idx]

        distance_corr[input_idx, freq_idx] = (
            dcor.distance_correlation(
                input_values,
                erp_values,
            )
        )

elapsed = time.perf_counter() - start_time

print("\nDistance correlation completed.")
print(f"Elapsed time: {elapsed:.2f} s")


# ==================================================
# Heatmap
# ==================================================

plt.figure(figsize=(14, 6))

image = plt.imshow(
    distance_corr,
    aspect="auto",
    origin="lower",
    extent=[
        frequencies[0],
        frequencies[-1],
        -0.5,
        num_inputs - 0.5,
    ],
    vmin=0.0,
    vmax=1.0,
)

plt.yticks(
    np.arange(num_inputs),
    INPUT_NAMES,
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Configuration Input")

plt.title(
    "Distance Correlation between "
    "Configuration Inputs and ERP"
)

plt.colorbar(
    image,
    label="Distance Correlation",
)

plt.tight_layout()
plt.show()


# ==================================================
# Distance correlation vs frequency
# ==================================================

plt.figure(figsize=(13, 7))

for input_idx, name in enumerate(INPUT_NAMES):

    plt.plot(
        frequencies,
        distance_corr[input_idx],
        linewidth=1.8,
        label=name,
    )

plt.xlabel("Frequency (Hz)")
plt.ylabel("Distance Correlation")

plt.title(
    "Configuration Input–ERP "
    "Distance Correlation vs Frequency"
)

plt.ylim(0.0, 1.0)

plt.grid(True)

plt.legend(
    ncol=3,
    fontsize=9,
)

plt.tight_layout()
plt.show()


# ==================================================
# Individual tuning-frequency correlations
# ==================================================

plt.figure(figsize=(12, 6))

ft_indices = [0, 3, 6]

for input_idx in ft_indices:

    plt.plot(
        frequencies,
        distance_corr[input_idx],
        linewidth=2,
        label=INPUT_NAMES[input_idx],
    )

plt.xlabel("Frequency (Hz)")
plt.ylabel("Distance Correlation")

plt.title(
    "Resonator Tuning Frequency–ERP Dependence"
)

plt.ylim(0.0, 1.0)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ==================================================
# Individual position correlations
# ==================================================

plt.figure(figsize=(12, 6))

position_indices = [
    1, 2,
    4, 5,
    7, 8,
]

for input_idx in position_indices:

    plt.plot(
        frequencies,
        distance_corr[input_idx],
        linewidth=1.8,
        label=INPUT_NAMES[input_idx],
    )

plt.xlabel("Frequency (Hz)")
plt.ylabel("Distance Correlation")

plt.title(
    "Resonator Position–ERP Dependence"
)

plt.ylim(0.0, 1.0)

plt.grid(True)

plt.legend(
    ncol=2,
)

plt.tight_layout()
plt.show()


# ==================================================
# Average dependence over full spectrum
# ==================================================

mean_distance_corr = np.mean(
    distance_corr,
    axis=1,
)

print("\n" + "=" * 70)
print("Mean distance correlation over complete ERP spectrum")
print("=" * 70)

sorted_indices = np.argsort(
    mean_distance_corr
)[::-1]

for idx in sorted_indices:

    print(
        f"{INPUT_NAMES[idx]:>4s} : "
        f"{mean_distance_corr[idx]:.4f}"
    )


# ==================================================
# Bar plot of mean dependence
# ==================================================

plt.figure(figsize=(9, 5))

plt.bar(
    INPUT_NAMES,
    mean_distance_corr,
)

plt.xlabel("Configuration Input")
plt.ylabel("Mean Distance Correlation")

plt.title(
    "Mean Input–ERP Dependence over "
    "10–160 Hz"
)

plt.ylim(
    0,
    max(mean_distance_corr) * 1.15,
)

plt.grid(
    axis="y",
)

plt.tight_layout()
plt.show()

from sklearn.feature_selection import mutual_info_regression

mutual_information = np.zeros(
    (X.shape[1], erp.shape[1])
)

for freq_idx in range(erp.shape[1]):

    mutual_information[:, freq_idx] = (
        mutual_info_regression(
            X,
            erp[:, freq_idx],
            random_state=727,
        )
    )