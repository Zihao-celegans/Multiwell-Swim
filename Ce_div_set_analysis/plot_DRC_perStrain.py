"""
plot_DRC_perStrain.py — Per-strain activity time-course across drug doses.

Combines the long-format CSVs exported by activity_by_time.py
(--by_condition --export_csv) for several dose conditions of the same
strain set (each dose recorded as a separate session, in its own
subfolder), e.g.:

    <input_dir>\\control\\activity_by_time_ActValS_by_condition_data.csv
    <input_dir>\\p0125\\activity_by_time_ActValS_by_condition_data.csv
    <input_dir>\\p025\\activity_by_time_ActValS_by_condition_data.csv

In those CSVs the "condition" column holds the strain name (the layout
used for --by_condition groups wells by strain, not by drug dose here —
dose is instead encoded by which subfolder/CSV the data came from).

For each strain, produces one figure with elapsed time (min) on the
x-axis and mean +/- SD activity (across wells) on the y-axis, with one
line per dose (one line per input CSV) so the time-course of each dose
can be compared directly.

Usage:
    python plot_DRC_perStrain.py \\
        --input_dir "E:\\MultiWell_swim\\08292026_CeDiv_Leva_test01" \\
        --metric ActValS --save

    # Explicit dose folder names / labels (mM), if not the default set
    python plot_DRC_perStrain.py \\
        --input_dir "E:\\MultiWell_swim\\08292026_CeDiv_Leva_test01" \\
        --doses control p0125 p025 --dose_mM 0 0.0125 0.025 \\
        --metric ActValS --save
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

FIGSIZE_4_3 = (8, 6)
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def load_dose_data(csv_path: str) -> dict[str, dict[float, list[float]]]:
    """Load a by_condition activity_by_time.py export CSV into
    strain -> elapsed_min -> [per-well values at that time point].
    """
    data: dict[str, dict[float, list[float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "condition" not in (reader.fieldnames or []):
            raise SystemExit(
                f"[DRCperStrain] {csv_path!r} has no 'condition' column — "
                f"re-export with activity_by_time.py --by_condition --export_csv."
            )
        for row in reader:
            strain = row["condition"]
            elapsed = float(row["elapsed_min"])
            value = float(row["value"])
            data.setdefault(strain, {}).setdefault(elapsed, []).append(value)
    return data


def plot_strain_timecourse(
    strain: str,
    doses: list,
    dose_mM: list,
    dose_data: dict,
    metric: str,
    output_dir: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_4_3)
    cmap = plt.get_cmap("tab10")

    for idx, dose in enumerate(doses):
        strain_data = dose_data[dose].get(strain)
        if not strain_data:
            print(f"[DRCperStrain] Warning: no data for strain={strain!r}, dose={dose!r}")
            continue

        elapsed_sorted = sorted(strain_data.keys())
        means = np.array([np.mean(strain_data[t]) for t in elapsed_sorted])
        stds = np.array([np.std(strain_data[t]) for t in elapsed_sorted])

        color = cmap(idx % cmap.N)
        marker = MARKER_CYCLE[idx % len(MARKER_CYCLE)]
        label = "Control" if dose_mM[idx] == 0 else f"{dose_mM[idx]:g} mM"
        ax.errorbar(elapsed_sorted, means, yerr=stds, color=color, marker=marker,
                    linewidth=2, capsize=4, elinewidth=1.5, label=label, zorder=3)

    ax.set_xlabel("Elapsed time (min)", fontsize=11)
    ax.set_ylabel(f"{metric} - Active pixels (A.U.)", fontsize=11)
    ax.set_title(strain)
    ax.legend(title="Levamisole dose", loc="best")
    plt.tight_layout()

    if output_dir:
        save_path = os.path.join(output_dir, f"activity_by_time_{strain}.png")
        fig.savefig(save_path, dpi=300)
        print(f"[DRCperStrain] Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Per-strain activity time-course across drug doses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Parent directory containing one subfolder per dose "
                             "(named per --doses), each with an "
                             "activity_by_time_<metric>_by_condition_data.csv export.")
    parser.add_argument("--doses", nargs="+", default=["control", "p0125", "p025"],
                        help="Dose subfolder names, in the order they should appear "
                             "in the legend.")
    parser.add_argument("--dose_mM", nargs="+", type=float, default=[0, 0.0125, 0.025],
                        help="Numeric dose (mM) matching each entry in --doses, one-to-one "
                             "(0 is labeled 'Control'). Used only for the legend labels.")
    parser.add_argument("--metric", default="ActValS",
                        help="Activity metric — must match the --metric used when the "
                             "CSVs were exported by activity_by_time.py.")
    parser.add_argument("--strains", nargs="+", default=None,
                        help="Strain names to plot. Defaults to every strain found "
                             "across all dose CSVs.")
    parser.add_argument("--save", action="store_true",
                        help="Save each strain's figure as a PNG (into --output_dir).")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save PNGs into. Defaults to a 'DRC_plots' "
                             "subfolder of --input_dir. Only used with --save.")
    args = parser.parse_args()

    if len(args.doses) != len(args.dose_mM):
        raise SystemExit(
            f"--doses has {len(args.doses)} entries but --dose_mM has {len(args.dose_mM)}; "
            f"provide exactly one dose (mM) per dose folder, in the same order."
        )

    csv_name = f"activity_by_time_{args.metric}_by_condition_data.csv"
    dose_data = {}
    for dose in args.doses:
        csv_path = os.path.join(args.input_dir, dose, csv_name)
        if not os.path.exists(csv_path):
            raise SystemExit(f"[DRCperStrain] Missing CSV for dose {dose!r}: {csv_path}")
        print(f"[DRCperStrain] Loading dose={dose!r}: {csv_path}")
        dose_data[dose] = load_dose_data(csv_path)

    if args.strains:
        strains = args.strains
    else:
        strains = sorted({strain for data in dose_data.values() for strain in data})
    print(f"[DRCperStrain] Plotting {len(strains)} strain(s): {strains}")

    output_dir = None
    if args.save:
        output_dir = args.output_dir or os.path.join(args.input_dir, "DRC_plots")
        os.makedirs(output_dir, exist_ok=True)

    for strain in strains:
        plot_strain_timecourse(strain, args.doses, args.dose_mM, dose_data, args.metric,
                                output_dir=output_dir)


if __name__ == "__main__":
    main()
