import importlib
import numpy as np
import matplotlib.pyplot as plt

import utils.physics as physics
import utils.solver


def plot_erp_mode_convergence(
    configuration,
    mode_settings=((7, 2), (9, 3), (12, 5), (15, 10)),
):
    """
    Compare the ERP spectrum of one resonator configuration
    using different modal truncations.

    configuration shape:
        (num_resonators, 3)

    each row:
        [f_t, x, y]

    Example:
        [
            [49.505, 0.5813, 0.1816],
            [128.590, 0.9216, 0.4470],
            [41.272, 0.0920, 0.4097],
        ]
    """

    configuration = np.asarray(configuration, dtype=float)

    # --------------------------------------------------
    # Convert [f_t, x, y] -> solver resonator format
    # --------------------------------------------------

    resonators = []

    for f_t, x, y in configuration:

        m = physics.k / (2.0 * np.pi * f_t) ** 2

        resonators.append(
            {
                "f_t": float(f_t),
                "x": float(x),
                "y": float(y),
                "m": float(m),
                "c": 1.0,
                "k": float(physics.k),
            }
        )


    # --------------------------------------------------
    # Save original modal settings
    # --------------------------------------------------

    original_resolution = physics.get_modal_resolution()
    results = {}

    try:

        # --------------------------------------------------
        # Calculate ERP for each modal truncation
        # --------------------------------------------------

        for Nx, Ny in mode_settings:
            # utils.solver reads the basis from utils.physics at call time
            # (caches are keyed on the resolution), so no reload is needed.
            physics.set_modal_resolution(Nx, Ny, verbose=False)

            print(
                f"Calculating: Nx={Nx}, Ny={Ny}, "
                f"N modes={physics.N}"
            )

            erp = utils.solver.compute_erp_spectrum(
                resonators,
                frequencies=physics.freqs,
            )

            results[(Nx, Ny)] = erp.copy()

    finally:
        physics.set_modal_resolution(*original_resolution, verbose=False)


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(figsize=(11, 6))

    for (Nx, Ny), erp in results.items():

        plt.plot(
            physics.freqs,
            erp,
            linewidth=2,
            label=f"Nx={Nx}, Ny={Ny} ({Nx * Ny} modes)",
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("ERP (dB)")
    plt.title("ERP Modal Convergence")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

configuration = np.array([
    [49.505, 0.5813, 0.1816],
    [128.590, 0.9216, 0.4470],
    [41.272, 0.0920, 0.4097],
])

results = plot_erp_mode_convergence(
    configuration,
    mode_settings=[
        (5, 2),
        (8, 5),
        (9, 7),
    ],
)