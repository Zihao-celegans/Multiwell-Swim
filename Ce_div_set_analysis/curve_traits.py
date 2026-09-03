"""
curve_traits.py — Quantitative traits from activity time-courses.

Reads the long-format CSVs exported by activity_by_time.py
(--by_condition --export_csv), one per drug dose (each dose recorded as a
separate session in its own subfolder), e.g.:

    <input_dir>\\control\\activity_by_time_ActValS_by_condition_data.csv
    <input_dir>\\p0125\\activity_by_time_ActValS_by_condition_data.csv
    <input_dir>\\p025\\activity_by_time_ActValS_by_condition_data.csv

The "condition" column holds the strain name; dose is encoded by which
subfolder the data came from.

Traits implemented so far:
    A0        activity at the first measured timepoint (this well's baseline)
    auc       trapezoidal area under the activity curve (A.U. * min), computed
              on the raw measured points with no smoothing, interpolation or
              baseline correction
    auc_norm  auc / A0 (minutes) — the run length a well would need to spend at
              its own baseline activity to accumulate the same area, which
              removes differences in absolute starting activity between wells

Alongside the per-well CSV it saves a grouped box plot of both AUC traits
(median/quartiles across wells, with the individual wells overlaid) for every
strain x dose combination.

Usage:
    python curve_traits.py --input_dir "E:\\MultiWell_swim\\08292026_CeDiv_Leva_test01"
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

trapezoid = getattr(np, "trapezoid", None) or np.trapz


def load_wells(csv_path: str) -> dict[str, dict[str, dict[float, float]]]:
    """Load a by_condition activity_by_time.py export into
    strain -> well -> elapsed_min -> value.
    """
    data: dict[str, dict[str, dict[float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"condition", "well", "elapsed_min", "value"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"[traits] {csv_path!r} is missing column(s) {sorted(missing)} — "
                f"re-export with activity_by_time.py --by_condition --export_csv."
            )
        for row in reader:
            (data.setdefault(row["condition"], {})
                 .setdefault(row["well"], {})[float(row["elapsed_min"])]) = float(row["value"])
    return data


def compute_auc(t: np.ndarray, y: np.ndarray) -> float:
    """Area under the activity curve by the trapezoidal rule.

    Uses the actual elapsed times, so the uneven sampling intervals are
    weighted correctly. Integration limits are the first and last measured
    timepoints of the well.
    """
    return float(trapezoid(y, t))


def compute_auc_norm(auc: float, A0: float) -> float:
    """AUC divided by the well's own baseline activity, in minutes."""
    return auc / A0 if A0 > 0 else float("nan")


def iter_wells(input_dir: str, doses: list[str], dose_mM: list[float], metric: str):
    """Yield (dose, dose_mM, strain, well, t, y) for every well, with t and y
    as float arrays sorted by elapsed time.
    """
    csv_name = f"activity_by_time_{metric}_by_condition_data.csv"
    for dose, conc in zip(doses, dose_mM):
        csv_path = os.path.join(input_dir, dose, csv_name)
        if not os.path.exists(csv_path):
            raise SystemExit(f"[traits] Missing CSV for dose {dose!r}: {csv_path}")
        print(f"[traits] Loading dose={dose!r}: {csv_path}")

        for strain, wells in sorted(load_wells(csv_path).items()):
            for well, series in sorted(wells.items()):
                times = sorted(series)
                if len(times) < 2:
                    print(f"[traits]   skipping {dose}/{strain}/{well}: only {len(times)} timepoint(s)")
                    continue
                t = np.array(times, dtype=float)
                y = np.array([series[tt] for tt in times], dtype=float)
                yield dose, conc, strain, well, t, y


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"[traits] Nothing to write to {path} — no wells were processed.")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[traits] Wrote {len(rows)} rows: {path}")


def format_dose_label(dose_mM: float) -> str:
    return "Control" if dose_mM == 0 else f"{dose_mM:g} mM"


def plot_auc_summary(rows: list[dict], metric: str, output_dir: str = None,
                     show: bool = False) -> None:
    """Grouped box plots of auc and auc_norm, with the individual wells
    overlaid so unequal replicate counts stay visible.
    """
    strains = sorted({r["strain"] for r in rows})
    doses = list(dict.fromkeys(r["dose"] for r in rows))
    dose_mM = {r["dose"]: r["dose_mM"] for r in rows}

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["strain"], row["dose"]), []).append(row)

    panels = [("auc", f"{metric} AUC (A.U. x min)"),
              ("auc_norm", f"{metric} AUC / A0 (min)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    width = 0.8 / len(doses)
    x = np.arange(len(strains))
    jitter = np.random.default_rng(0)

    for ax, (trait, ylabel) in zip(axes, panels):
        for idx, dose in enumerate(doses):
            offset = (idx - (len(doses) - 1) / 2) * width
            color = cmap(idx % cmap.N)
            data, positions = [], []
            for i, strain in enumerate(strains):
                vals = np.array([r[trait] for r in grouped.get((strain, dose), [])], dtype=float)
                vals = vals[~np.isnan(vals)]
                if not len(vals):
                    continue
                data.append(vals)
                positions.append(x[i] + offset)
                ax.scatter(x[i] + offset + jitter.uniform(-width * 0.18, width * 0.18, len(vals)),
                           vals, s=9, color="black", alpha=0.55, linewidths=0, zorder=3)

            if not data:
                continue
            # manage_ticks=False keeps the strain x-ticks set at the end intact.
            bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                            patch_artist=True, showfliers=False, manage_ticks=False,
                            medianprops={"color": "black", "linewidth": 1.4})
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for boundary in x[:-1] + 0.5:
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8,
                       alpha=0.6, zorder=1)

    axes[0].legend(handles=[Patch(facecolor=cmap(i % cmap.N), alpha=0.75, edgecolor="black",
                                  label=format_dose_label(dose_mM[d]))
                            for i, d in enumerate(doses)],
                   title="Dose", loc="best", fontsize=9)
    axes[0].set_title(f"{metric} area under the activity curve, by strain and dose")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(strains, rotation=45, ha="right")
    axes[-1].set_xlim(-0.6, len(strains) - 0.4)
    plt.tight_layout()

    if output_dir:
        save_path = os.path.join(output_dir, f"curve_traits_{metric}_auc.png")
        fig.savefig(save_path, dpi=300)
        print(f"[traits] Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative traits from activity time-courses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Parent directory containing one subfolder per dose "
                             "(named per --doses), each with an "
                             "activity_by_time_<metric>_by_condition_data.csv export.")
    parser.add_argument("--doses", nargs="+", default=["control", "p0125", "p025"],
                        help="Dose subfolder names.")
    parser.add_argument("--dose_mM", nargs="+", type=float, default=[0, 0.0125, 0.025],
                        help="Numeric dose (mM) matching each entry in --doses, one-to-one.")
    parser.add_argument("--metric", default="ActValS",
                        help="Activity metric — must match the --metric used when the "
                             "CSVs were exported by activity_by_time.py.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for the trait CSV. Defaults to a 'traits' subfolder "
                             "of --input_dir.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip the AUC summary figure.")
    parser.add_argument("--show", action="store_true",
                        help="Pop up the figure interactively as well as saving it.")
    args = parser.parse_args()

    if len(args.doses) != len(args.dose_mM):
        raise SystemExit(
            f"--doses has {len(args.doses)} entries but --dose_mM has {len(args.dose_mM)}; "
            f"provide exactly one dose (mM) per dose folder, in the same order."
        )

    rows = []
    for dose, conc, strain, well, t, y in iter_wells(
            args.input_dir, args.doses, args.dose_mM, args.metric):
        auc = compute_auc(t, y)
        A0 = float(y[0])
        rows.append({
            "dose": dose,
            "dose_mM": conc,
            "strain": strain,
            "well": well,
            "n_timepoints": len(t),
            "t_first": t[0],
            "t_last": t[-1],
            "A0": A0,
            "auc": auc,
            "auc_norm": compute_auc_norm(auc, A0),
        })

    output_dir = args.output_dir or os.path.join(args.input_dir, "traits")
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, f"curve_traits_{args.metric}_per_well.csv"), rows)

    if not args.no_plot:
        plot_auc_summary(rows, args.metric, output_dir=output_dir, show=args.show)


if __name__ == "__main__":
    main()
