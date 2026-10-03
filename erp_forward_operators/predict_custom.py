"""Run the trained forward operators on hand-picked configurations that are not
in the dataset, and compare their ERP with the solver's.

Each configuration is three resonators ``(m, f_t, x, y)`` (k = m (2 pi f_t)^2),
as in the ``100k`` dataset the operators were trained on (150 plate modes).
Configuration 4 deliberately leaves the training ranges (m > 1 kg, f_t outside
10-160 Hz, positions closer to the corners than the 5 cm sampling margin) to
show extrapolation.

For every configuration the figure shows the solver ERP against the best few
operators, the plate with the resonators, and the RMSE of every operator.
Saves to ``plots/GENERAL/100k/custom_configurations/``.

Usage (from the repository root)::

    python -m erp_forward_operators.predict_custom
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from erp_forward_operators.diagnose import _SPEC_BY_SHORT, _display_name
from erp_forward_operators.neural_operator_utils import build_operator_model, device, load_operator_checkpoint
from utils.erp_dataset import (
    configuration_to_resonators,
    denormalize_erp_array,
    normalize_configuration_array,
    normalize_frequency_array,
    select_dataset_modal_resolution,
)
from utils.paths import FORWARD_ROOT, GENERAL, forward_plot_root
from utils.physics import Lx, Ly, edge_margin, fmax, fmin, freqs, m_max, m_min, xf, yf
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.solver import compute_erp_spectrum

DATASET = "100k"
# Best variant of each architecture on the 100k test split (error_breakdown.txt); NN_perm left out.
MODELS = ("DNO", "DCO_sorted_phys", "LNO", "STO", "SIREN", "FNO", "WNO", "GNO_phys", "DON")
SHOWN = ("DNO", "DCO_sorted_phys", "STO")  # drawn on the ERP panel
LINE_COLORS = ("#2a78d6", "#eb6834", "#1f9e74")
RES_COLORS = ("#2a78d6", "#eb6834", "#1f9e74")

# (title, [(m [kg], f_t [Hz], x [m], y [m]), ...]); none of these is in the dataset.
CUSTOM_CONFIGURATIONS = [
    ("Ordinary configuration, not in the dataset",
     [(0.40, 45.0, 0.30, 0.12), (0.70, 95.0, 1.10, 0.38), (0.25, 130.0, 0.65, 0.25)]),
    ("All three resonators tuned to the same frequency (72 Hz)",
     [(0.30, 72.0, 0.30, 0.20), (0.50, 72.0, 0.80, 0.35), (0.70, 72.0, 1.20, 0.10)]),
    ("Two resonators almost touching, next to the excitation force",
     [(0.50, 60.0, 0.85, 0.30), (0.50, 90.0, 0.88, 0.32), (0.30, 120.0, 0.20, 0.40)]),
    ("Outside the training ranges ($m = 1.5$ kg, $f_t = 5$ and $175$ Hz, corners)",
     [(1.50, 40.0, 0.02, 0.03), (0.30, 5.0, 1.38, 0.47), (0.60, 175.0, 0.70, 0.25)]),
]


def build_configuration(rows) -> np.ndarray:
    return np.asarray([[m, m * (2 * np.pi * f) ** 2, f, x, y] for m, f, x, y in rows], dtype=np.float32)


def load_operator(name: str):
    checkpoint = load_operator_checkpoint(str(FORWARD_ROOT / "models" / GENERAL / DATASET / f"{name.lower()}.pth"))
    spec = _SPEC_BY_SHORT[_display_name(name.lower()).partition("_")[0]]
    state = checkpoint["preprocessing_state"]
    model = build_operator_model(spec["build_model"], int(state["num_res"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), state["norm_params"]


@torch.no_grad()
def predict(model, norm, configurations: np.ndarray, frequency_values: np.ndarray) -> np.ndarray:
    config = torch.from_numpy(normalize_configuration_array(configurations, norm)).to(device)
    freq = torch.from_numpy(normalize_frequency_array(frequency_values.astype(np.float32), norm)[None, :, None])
    freq = freq.expand(config.shape[0], -1, -1).contiguous().to(device)
    return denormalize_erp_array(model(config, freq).cpu().numpy()[..., 0], norm)


def out_of_range(row) -> list[str]:
    m, _, f_t, x, y = row
    notes = []
    if not m_min <= m <= m_max:
        notes.append(f"m outside {m_min:g}--{m_max:g} kg")
    if not fmin <= f_t <= fmax:
        notes.append(f"$f_t$ outside {fmin:g}--{fmax:g} Hz")
    if not (edge_margin <= x <= Lx - edge_margin and edge_margin <= y <= Ly - edge_margin):
        notes.append("inside the 5 cm edge margin")
    return notes


def draw_plate(ax, configuration):
    ax.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.1, zorder=0))
    ax.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False,
                           ec="#9a9a94", ls="--", lw=0.8, zorder=1, label="Sampled region of the dataset"))
    ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=14, mec="black", mew=0.8, ls="", zorder=6, label="Excitation force")
    for r, (m, _, f_t, x, y) in enumerate(configuration):
        ax.plot(x, y, "o", color=RES_COLORS[r % 3], ms=11, mec="black", mew=1.0, ls="", zorder=5,
                label=f"R{r + 1}: $m = {m:g}$ kg, $f_t = {f_t:g}$ Hz")
    ax.set_xlim(-0.02, Lx + 0.02)
    ax.set_ylim(-0.02, Ly + 0.02)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
    ax.set_yticks(np.arange(0, Ly + 1e-9, 0.1))
    ax.set_xlabel("Position $x$ (m)")
    ax.set_ylabel("Position $y$ (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8)


def main(configurations=CUSTOM_CONFIGURATIONS, models=MODELS) -> None:
    select_dataset_modal_resolution(DATASET)  # solver with the operators' 15 x 10 plate modes
    frequency_values = np.asarray(freqs, dtype=np.float64)
    designs = np.stack([build_configuration(rows) for _, rows in configurations])
    truth = np.stack([compute_erp_spectrum(configuration_to_resonators(d), frequencies=frequency_values) for d in designs])
    predictions = {}
    for name in models:
        model, norm = load_operator(name)
        predictions[name] = predict(model, norm, designs, frequency_values.astype(np.float32))
    out_dir = forward_plot_root(DATASET, GENERAL) / "custom_configurations"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["Forward operators (trained on the 100k dataset) on hand-picked configurations; RMSE vs the solver (dB)", ""]
    for i, (title, _) in enumerate(configurations):
        design = designs[i]
        rmse = {n: float(np.sqrt(((p[i] - truth[i]) ** 2).mean())) for n, p in predictions.items()}
        notes = sorted({note for row in design for note in out_of_range(row)})
        lines.append(f"Configuration {i + 1}: {title}" + (f"  [extrapolation: {'; '.join(notes)}]" if notes else ""))
        for r, (m, k, f_t, x, y) in enumerate(design):
            lines.append(f"  R{r + 1}: m={m:.2f} kg  k={k:,.0f} N/m  f_t={f_t:.0f} Hz  (x, y)=({x:.2f}, {y:.2f}) m")
        lines += [f"  {n:<16} RMSE {v:6.2f} dB" for n, v in sorted(rmse.items(), key=lambda kv: kv[1])]
        lines.append("")

        fig = plt.figure(figsize=(15, 9.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.05, 1.0], hspace=0.38, wspace=0.22)
        ax = fig.add_subplot(gs[0, :])
        for f_t in design[:, 2]:
            if frequency_values.min() <= f_t <= frequency_values.max():
                ax.axvline(f_t, color="#c2412c", ls="-.", lw=0.9, zorder=0)
        ax.plot([], [], color="#c2412c", ls="-.", lw=0.9, label="Resonator tuning frequencies")
        ax.plot(frequency_values, truth[i], color="black", lw=2.2, zorder=4, label="True ERP (solver)")
        for name, color in zip(SHOWN, LINE_COLORS):
            ax.plot(frequency_values, predictions[name][i], color=color, lw=1.4, zorder=3,
                    label=f"{name.replace('_', ' ')} (RMSE {rmse[name]:.2f} dB)")
        ax.set_xlim(frequency_values.min(), frequency_values.max())
        ax.set_xlabel(FREQ_LABEL)
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, ncol=2)
        ax.set_title("Predicted vs true ERP")

        draw_plate(fig.add_subplot(gs[1, 0]), design)
        ax_bar = fig.add_subplot(gs[1, 1])
        order = sorted(rmse, key=rmse.get)
        ax_bar.barh([n.replace("_", " ") for n in order][::-1], [rmse[n] for n in order][::-1], color="#2a78d6")
        for y_pos, n in enumerate(order[::-1]):
            ax_bar.text(rmse[n], y_pos, f" {rmse[n]:.2f}", va="center", fontsize=8)
        ax_bar.set_xlabel("RMSE vs solver (dB)")
        ax_bar.set_title("Error of every forward operator")
        ax_bar.grid(axis="x", alpha=0.3)
        subtitle = f"\n(extrapolation: {'; '.join(notes)})" if notes else ""
        fig.suptitle(f"New configuration {i + 1}: {title}{subtitle}", fontsize=13)
        save_figure(fig, out_dir / f"custom_config_{i + 1}.png")

    (out_dir / "custom_configurations.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
