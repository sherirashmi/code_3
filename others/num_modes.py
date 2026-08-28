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

    original_values = {
        "Nx": physics.Nx,
        "Ny": physics.Ny,
        "modes": physics.modes.copy(),
        "omega_n": physics.omega_n.copy(),
        "omega_sq": physics.omega_sq.copy(),
        "m_idx": physics.m_idx.copy(),
        "n_idx": physics.n_idx.copy(),
        "N": physics.N,
    }


    results = {}

    try:

        # --------------------------------------------------
        # Calculate ERP for each modal truncation
        # --------------------------------------------------

        for Nx, Ny in mode_settings:

            # Rebuild modal basis
            physics.Nx = Nx
            physics.Ny = Ny

            physics.modes = physics.modal_table(
                nx_modes=Nx,
                ny_modes=Ny,
            )

            physics.omega_n = physics.modes[:, 0]

            physics.omega_sq = physics.omega_n**2

            physics.m_idx = physics.modes[:, 1].astype(np.int32)

            physics.n_idx = physics.modes[:, 2].astype(np.int32)

            physics.N = len(physics.modes)

            # Reload solver so imported N, omega_n, etc.
            # match the new modal basis
            importlib.reload(utils.solver)

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

        # --------------------------------------------------
        # Restore original physics settings
        # --------------------------------------------------

        physics.Nx = original_values["Nx"]
        physics.Ny = original_values["Ny"]
        physics.modes = original_values["modes"]
        physics.omega_n = original_values["omega_n"]
        physics.omega_sq = original_values["omega_sq"]
        physics.m_idx = original_values["m_idx"]
        physics.n_idx = original_values["n_idx"]
        physics.N = original_values["N"]

        importlib.reload(utils.solver)


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