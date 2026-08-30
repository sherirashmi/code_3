"""
Analyze peak statistics in the saved ERP dataset.

This script loads the raw ERP dataset saved by erp_dataset.py and reports:

1. Number of configurations with:
   - 0 peaks
   - 1 peak
   - 2 peaks
   - 3 peaks
   - 4 peaks
   - 5 or more peaks

2. Peak-spacing statistics based on the minimum separation between
   adjacent detected peaks in each configuration.

   Cumulative:
   - at least one pair within 5 Hz
   - at least one pair within 10 Hz
   - at least one pair within 20 Hz

   Exclusive bins:
   - <= 5 Hz
   - > 5 and <= 10 Hz
   - > 10 and <= 20 Hz
   - > 20 Hz

3. A CSV containing one row per configuration with:
   - configuration index
   - resonator features
   - number of peaks
   - peak frequencies
   - peak amplitudes
   - minimum adjacent peak spacing
   - spacing category

The default peak prominence is 3 dB, matching solver.get_peak_frequencies().

Run from the project root:

    python analyze_erp_dataset.py

Optional examples:

    python analyze_erp_dataset.py --dataset datasets/dataset_erp_ft.pth
    python analyze_erp_dataset.py --prominence 2.0
    python analyze_erp_dataset.py --prominence 5.0
    python analyze_erp_dataset.py --thresholds 5 10 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks


DEFAULT_DATASET = "datasets/dataset_erp_ft.pth"
DEFAULT_OUTPUT_DIR = "dataset_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze number of ERP peaks and separation between nearby peaks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to ERP dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=5.0,
        help="Peak prominence in dB (default: 3.0, same as solver.py)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        default=[5.0, 10.0, 20.0],
        metavar=("T1", "T2", "T3"),
        help="Peak-separation thresholds in Hz (default: 5 10 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/summary output (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_raw_dataset(filename: str) -> dict[str, object]:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run this script from the project root or pass --dataset with the correct path."
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError(
            "Expected the saved ERP dataset to be a dictionary payload."
        )

    required = {"frequency_values", "responses", "configuration_features"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")

    return payload


def validate_arrays(
    payload: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.asarray(payload["frequency_values"], dtype=np.float64)
    responses = np.asarray(payload["responses"], dtype=np.float64)
    configurations = np.asarray(payload["configuration_features"], dtype=np.float64)

    if frequencies.ndim != 1:
        raise ValueError(
            f"frequency_values must be 1D, got shape {frequencies.shape}"
        )

    if responses.ndim == 3 and responses.shape[-1] == 1:
        responses = responses[..., 0]

    if responses.ndim != 2:
        raise ValueError(
            "responses must have shape (n_configurations, n_frequencies, 1) "
            f"or (n_configurations, n_frequencies), got {responses.shape}"
        )

    if responses.shape[1] != frequencies.size:
        raise ValueError(
            f"responses has {responses.shape[1]} frequency points but "
            f"frequency_values has {frequencies.size}"
        )

    if configurations.ndim != 3 or configurations.shape[-1] != 3:
        raise ValueError(
            "configuration_features must have shape "
            f"(n_configurations, num_res, 3), got {configurations.shape}"
        )

    if configurations.shape[0] != responses.shape[0]:
        raise ValueError(
            "configuration_features and responses contain different numbers "
            "of configurations."
        )

    return frequencies, responses, configurations


def spacing_category(
    min_spacing: float | None,
    thresholds: tuple[float, float, float],
) -> str:
    if min_spacing is None:
        return "not_applicable"

    t1, t2, t3 = thresholds
    if min_spacing <= t1:
        return f"<= {t1:g} Hz"
    if min_spacing <= t2:
        return f"> {t1:g} to <= {t2:g} Hz"
    if min_spacing <= t3:
        return f"> {t2:g} to <= {t3:g} Hz"
    return f"> {t3:g} Hz"


def analyze_dataset(
    frequencies: np.ndarray,
    responses: np.ndarray,
    configurations: np.ndarray,
    *,
    prominence: float,
    thresholds: tuple[float, float, float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []

    peak_count_hist: dict[int, int] = {}

    cumulative_close = {
        thresholds[0]: 0,
        thresholds[1]: 0,
        thresholds[2]: 0,
    }

    exclusive_spacing_counts = {
        f"<= {thresholds[0]:g} Hz": 0,
        f"> {thresholds[0]:g} to <= {thresholds[1]:g} Hz": 0,
        f"> {thresholds[1]:g} to <= {thresholds[2]:g} Hz": 0,
        f"> {thresholds[2]:g} Hz": 0,
    }

    fewer_than_two_peaks = 0

    for idx, erp in enumerate(responses):
        peak_idx, properties = find_peaks(
            erp,
            prominence=prominence,
        )

        peak_freqs = frequencies[peak_idx]
        peak_amps = erp[peak_idx]
        num_peaks = int(peak_idx.size)

        peak_count_hist[num_peaks] = peak_count_hist.get(num_peaks, 0) + 1

        if num_peaks >= 2:
            # find_peaks returns indices in ascending order, therefore differences
            # are already adjacent peak spacings along the frequency axis.
            adjacent_spacings = np.diff(peak_freqs)
            min_spacing = float(adjacent_spacings.min())

            for threshold in cumulative_close:
                if min_spacing <= threshold:
                    cumulative_close[threshold] += 1

            category = spacing_category(min_spacing, thresholds)
            exclusive_spacing_counts[category] += 1
        else:
            min_spacing = None
            adjacent_spacings = np.asarray([], dtype=np.float64)
            category = "not_applicable"
            fewer_than_two_peaks += 1

        config = configurations[idx]

        rows.append(
            {
                "configuration_index": idx,
                "num_peaks": num_peaks,
                "peak_frequencies_hz": peak_freqs.tolist(),
                "peak_amplitudes_db": peak_amps.tolist(),
                "adjacent_peak_spacings_hz": adjacent_spacings.tolist(),
                "minimum_peak_spacing_hz": min_spacing,
                "spacing_category": category,
                "configuration": config.tolist(),
            }
        )

    n = int(responses.shape[0])

    summary = {
        "num_configurations": n,
        "num_frequencies": int(frequencies.size),
        "frequency_min_hz": float(frequencies.min()),
        "frequency_max_hz": float(frequencies.max()),
        "frequency_step_hz": (
            float(np.median(np.diff(frequencies)))
            if frequencies.size >= 2
            else None
        ),
        "prominence_db": float(prominence),
        "peak_count_hist": peak_count_hist,
        "cumulative_close": cumulative_close,
        "exclusive_spacing_counts": exclusive_spacing_counts,
        "fewer_than_two_peaks": fewer_than_two_peaks,
    }
    return rows, summary


def peak_group_count(
    peak_count_hist: dict[int, int],
    exact_count: int | None = None,
    minimum_count: int | None = None,
) -> int:
    if exact_count is not None:
        return peak_count_hist.get(exact_count, 0)
    if minimum_count is not None:
        return sum(
            count
            for n_peaks, count in peak_count_hist.items()
            if n_peaks >= minimum_count
        )
    raise ValueError("Provide exact_count or minimum_count.")


def pct(count: int, total: int) -> float:
    return 100.0 * count / max(total, 1)


def format_summary(summary: dict[str, object]) -> str:
    n = int(summary["num_configurations"])
    hist = summary["peak_count_hist"]
    cumulative = summary["cumulative_close"]
    exclusive = summary["exclusive_spacing_counts"]
    prominence = float(summary["prominence_db"])

    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("ERP DATASET PEAK ANALYSIS")
    lines.append("=" * 78)
    lines.append(f"Configurations        : {n}")
    lines.append(f"Frequencies/config    : {summary['num_frequencies']}")
    lines.append(
        f"Frequency range       : "
        f"{summary['frequency_min_hz']:g} to {summary['frequency_max_hz']:g} Hz"
    )
    if summary["frequency_step_hz"] is not None:
        lines.append(f"Approx. frequency step: {summary['frequency_step_hz']:g} Hz")
    lines.append(f"Peak prominence       : {prominence:g} dB")
    lines.append("")

    lines.append("-" * 78)
    lines.append("NUMBER OF DETECTED PEAKS PER CONFIGURATION")
    lines.append("-" * 78)

    groups = [
        ("0 peaks", peak_group_count(hist, exact_count=0)),
        ("1 peak", peak_group_count(hist, exact_count=1)),
        ("2 peaks", peak_group_count(hist, exact_count=2)),
        ("3 peaks", peak_group_count(hist, exact_count=3)),
        ("4 peaks", peak_group_count(hist, exact_count=4)),
        ("5 or more peaks", peak_group_count(hist, minimum_count=5)),
    ]

    for label, count in groups:
        lines.append(f"{label:<24s}: {count:5d}  ({pct(count, n):6.2f}%)")

    lines.append("")
    lines.append("Exact detected-peak histogram:")
    for num_peaks in sorted(hist):
        count = hist[num_peaks]
        lines.append(
            f"  {num_peaks:2d} peak(s): {count:5d}  ({pct(count, n):6.2f}%)"
        )

    lines.append("")
    lines.append("-" * 78)
    lines.append("CLOSE-PEAK ANALYSIS — CUMULATIVE")
    lines.append("-" * 78)
    lines.append(
        "A configuration is counted when its closest pair of detected peaks "
        "is within the stated distance."
    )

    for threshold, count in cumulative.items():
        lines.append(
            f"Any adjacent peak pair <= {threshold:g} Hz"
            f": {count:5d}  ({pct(count, n):6.2f}% of all configurations)"
        )

    lines.append("")
    lines.append("-" * 78)
    lines.append("CLOSE-PEAK ANALYSIS — NON-OVERLAPPING BINS")
    lines.append("-" * 78)

    multi_peak_total = n - int(summary["fewer_than_two_peaks"])
    lines.append(
        f"Configurations with >= 2 peaks: {multi_peak_total} "
        f"({pct(multi_peak_total, n):.2f}%)"
    )
    lines.append(
        f"Configurations with < 2 peaks : {summary['fewer_than_two_peaks']} "
        "(spacing not defined)"
    )
    lines.append("")

    for label, count in exclusive.items():
        lines.append(
            f"{label:<26s}: {count:5d}  "
            f"({pct(count, multi_peak_total):6.2f}% of >=2-peak configurations)"
        )

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


def write_csv(
    rows: list[dict[str, object]],
    configurations: np.ndarray,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    num_res = configurations.shape[1]

    fieldnames = [
        "configuration_index",
        "num_peaks",
        "peak_frequencies_hz",
        "peak_amplitudes_db",
        "adjacent_peak_spacings_hz",
        "minimum_peak_spacing_hz",
        "spacing_category",
    ]

    for r in range(num_res):
        fieldnames.extend(
            [
                f"resonator_{r+1}_f_t_hz",
                f"resonator_{r+1}_x_m",
                f"resonator_{r+1}_y_m",
            ]
        )

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            record = {
                "configuration_index": row["configuration_index"],
                "num_peaks": row["num_peaks"],
                "peak_frequencies_hz": "; ".join(
                    f"{v:.6g}" for v in row["peak_frequencies_hz"]
                ),
                "peak_amplitudes_db": "; ".join(
                    f"{v:.6g}" for v in row["peak_amplitudes_db"]
                ),
                "adjacent_peak_spacings_hz": "; ".join(
                    f"{v:.6g}" for v in row["adjacent_peak_spacings_hz"]
                ),
                "minimum_peak_spacing_hz": (
                    ""
                    if row["minimum_peak_spacing_hz"] is None
                    else f"{row['minimum_peak_spacing_hz']:.6g}"
                ),
                "spacing_category": row["spacing_category"],
            }

            config = np.asarray(row["configuration"], dtype=np.float64)
            for r in range(num_res):
                record[f"resonator_{r+1}_f_t_hz"] = f"{config[r, 0]:.6g}"
                record[f"resonator_{r+1}_x_m"] = f"{config[r, 1]:.6g}"
                record[f"resonator_{r+1}_y_m"] = f"{config[r, 2]:.6g}"

            writer.writerow(record)


def main() -> None:
    args = parse_args()

    thresholds = tuple(sorted(float(v) for v in args.thresholds))
    if thresholds[0] <= 0:
        raise ValueError("All spacing thresholds must be positive.")

    if args.prominence < 0:
        raise ValueError("prominence must be non-negative.")

    payload = load_raw_dataset(args.dataset)
    frequencies, responses, configurations = validate_arrays(payload)

    rows, summary = analyze_dataset(
        frequencies,
        responses,
        configurations,
        prominence=float(args.prominence),
        thresholds=thresholds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_text = format_summary(summary)
    print(summary_text)

    summary_file = output_dir / "peak_analysis_summary.txt"
    summary_file.write_text(summary_text + "\n", encoding="utf-8")

    csv_file = output_dir / "per_configuration_peak_analysis.csv"
    write_csv(rows, configurations, csv_file)

    print(f"\nSummary saved to : {summary_file.resolve()}")
    print(f"Detailed CSV saved: {csv_file.resolve()}")
    print(
        "\nTip: repeat with different prominence values, e.g. "
        "--prominence 2, 3, and 5, because the detected number of peaks "
        "depends strongly on the prominence threshold."
    )


if __name__ == "__main__":
    main()
