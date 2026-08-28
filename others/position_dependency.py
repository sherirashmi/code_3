import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.physics import k, freqs
from utils.solver import compute_erp_spectrum


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = torch.load(
    "datasets/dataset_erp_ft.pth",
    map_location="cpu",
    weights_only=False,
)

configuration_features = np.asarray(data["configuration_features"])


# --------------------------------------------------
# Select 5 configurations for their x,y positions
# --------------------------------------------------

rng = np.random.default_rng(727)

configuration_ids = rng.choice(
    configuration_features.shape[0],
    size=5,
    replace=False,
)


# --------------------------------------------------
# Use SAME resonator tuning frequencies for all 5
# Take them from configuration 0
# --------------------------------------------------

reference_configuration = 0

fixed_ft = configuration_features[
    reference_configuration, :, 0
].copy()

print("\nFixed tuning frequencies used for ALL configurations:")
for i, ft in enumerate(fixed_ft):
    print(f"Resonator {i+1}: f_t = {ft:.3f} Hz")


# --------------------------------------------------
# Generate ERP spectra
# --------------------------------------------------

erp_spectra = []

print("\n" + "=" * 75)
print("CONFIGURATIONS CONSIDERED")
print("=" * 75)

for j, configuration_id in enumerate(configuration_ids):

    # Take only x,y from this configuration
    positions = configuration_features[
        configuration_id, :, 1:3
    ]

    resonators = []

    print(f"\nConfiguration {j+1} "
          f"(positions taken from dataset ID {configuration_id})")

    for i in range(len(fixed_ft)):

        ft_i = fixed_ft[i]
        x_i = positions[i, 0]
        y_i = positions[i, 1]

        # Convert ft -> mass only for physics solver
        m_i = k / (2.0 * np.pi * ft_i) ** 2

        resonators.append(
            {
                "f_t": float(ft_i),
                "x": float(x_i),
                "y": float(y_i),
                "m": float(m_i),
                "c": 1.0,
                "k": float(k),
            }
        )

        print(
            f"  Resonator {i+1}: "
            f"f_t={ft_i:8.3f} Hz | "
            f"x={x_i:7.4f} m | "
            f"y={y_i:7.4f} m"
        )

    erp = compute_erp_spectrum(
        resonators,
        frequencies=freqs,
    )

    erp_spectra.append(erp)


# --------------------------------------------------
# Plot complete ERP spectra
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for i, erp in enumerate(erp_spectra):

    plt.plot(
        freqs,
        erp,
        linewidth=2,
        label=f"Configuration {i+1}",
    )

plt.xlabel("Frequency (Hz)")
plt.ylabel("ERP (dB)")
plt.title(
    "ERP Spectra: Same Resonator Tuning Frequencies, Different Positions"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()