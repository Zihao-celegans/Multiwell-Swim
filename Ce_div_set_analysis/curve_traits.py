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
    slope_init  early slope (A.U. per minute) from an OLS straight-line fit of
                activity vs time over the first --slope_points timepoints.
                Reported as fitted, so a decaying well is negative
    slope_final same fit over the last --final_slope_points timepoints, so a
                well that is recovering by the end of the run is positive
    range_delta (first timepoint - min) of the well's control-subtracted
                activity curve (see the control normalization below), in A.U.
    range_ratio (first timepoint - min) of the well's control-divided activity
                curve (fold of control, unitless)
    t_p10_delta, t_p50_delta, t_p90_delta, t_p10_ratio, t_p50_ratio, t_p90_ratio
                elapsed time (min) at which the linearly interpolated normalized
                curve first reaches First - p * range or below, where First is
                the curve's first timepoint and range = First - Min, for
                p = 0.1, 0.5, 0.9 (--range_fracs)

Alongside the per-well CSV it saves grouped box plots (median/quartiles across
wells, with the individual wells overlaid) for every strain x dose
combination: one figure for the AUC traits, one for the initial slope, one for
the final slope, one each for range_delta and range_ratio, and one per
normalization with a panel per --range_fracs value for the t_p<N> traits.
The plots of the post-normalization traits (range and t_p<N>) leave out the
control (0 mM) dose; the CSV still lists every well.

Separately from the traits, it also plots the control-normalized activity
time-course: each well's activity at every timepoint is normalized to the
median of the same strain's control (0 mM) wells at that timepoint, in two ways:
    delta  activity minus control median (net drug effect, control at 0)
           curve_traits_<metric>_timecourse_delta.png
    ratio  activity / control median (fold of control, control at 1)
           curve_traits_<metric>_timecourse_ratio.png
Each is shown per strain as median +/- quartiles across wells, one line per
dose. For each non-control dose it also saves a figure with all strains
overlaid (curve_traits_<metric>_timecourse_<delta|ratio>_all_strains_<dose>.png,
median +/- quartiles). Dose sessions are matched by nearest elapsed time within
--time_tol_min. All traits except range_delta / range_ratio are computed on the
raw activity; those two use each well's own control-normalized curve.

Usage:
    python curve_traits.py --input_dir "E:\\MultiWell_swim\\08292026_CeDiv_Leva_test01"
    python curve_traits.py --input_dir "E:\\MultiWell_swim\\09062026_CeDiv_Pyrantel_test01" --doses control p25 1 --dose_mM 0 0.25 1
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


def compute_slope_init(t: np.ndarray, y: np.ndarray, n_points: int) -> float:
    """OLS slope of activity vs time over the first n_points, in A.U./min.

    Straight line through the raw values, so the sign is as fitted: negative
    while activity is falling.
    """
    n = min(n_points, len(y))
    if n < 2:
        return float("nan")
    return float(np.polyfit(t[:n], y[:n], 1)[0])


def compute_slope_final(t: np.ndarray, y: np.ndarray, n_points: int) -> float:
    """OLS slope of activity vs time over the last n_points, in A.U./min.

    Same convention as compute_slope_init, so a well still falling at the end
    of the run is negative and one that has turned around is positive.
    """
    n = min(n_points, len(y))
    if n < 2:
        return float("nan")
    return float(np.polyfit(t[-n:], y[-n:], 1)[0])


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


def plot_trait_boxes(rows: list[dict], panels: list[tuple], title: str, filename: str,
                     output_dir: str = None, show: bool = False,
                     skip_control: bool = False) -> None:
    """Grouped box plots of the given (trait, ylabel) panels, one box per
    strain x dose, with the individual wells overlaid so unequal replicate
    counts stay visible. skip_control drops the 0 mM dose from the plot; the
    remaining doses keep the colors they have in the plots that include it.
    """
    all_doses = list(dict.fromkeys(r["dose"] for r in rows))
    dose_mM = {r["dose"]: r["dose_mM"] for r in rows}
    if skip_control:
        rows = [r for r in rows if r["dose_mM"] != 0]
    strains = sorted({r["strain"] for r in rows})
    doses = [d for d in all_doses if not (skip_control and dose_mM[d] == 0)]
    color_idx = {d: all_doses.index(d) for d in doses}

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["strain"], row["dose"]), []).append(row)

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 4.5 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    cmap = plt.get_cmap("tab10")
    width = 0.8 / len(doses)
    x = np.arange(len(strains))
    jitter = np.random.default_rng(0)

    for ax, (trait, ylabel) in zip(axes, panels):
        for idx, dose in enumerate(doses):
            offset = (idx - (len(doses) - 1) / 2) * width
            color = cmap(color_idx[dose] % cmap.N)
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
        if ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
            ax.axhline(0, color="black", linewidth=0.9, alpha=0.7, zorder=1)

    axes[0].legend(handles=[Patch(facecolor=cmap(color_idx[d] % cmap.N), alpha=0.75,
                                  edgecolor="black", label=format_dose_label(dose_mM[d]))
                            for d in doses],
                   title="Dose", loc="best", fontsize=9)
    if title:
        axes[0].set_title(title)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(strains, rotation=45, ha="right")
    axes[-1].set_xlim(-0.6, len(strains) - 0.4)
    plt.tight_layout()

    if output_dir:
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300)
        print(f"[traits] Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


NORMALIZATIONS = {
    # mode -> (operation, reference level of the control, y label, file tag, legend loc)
    "delta": (lambda y, c: y - c, 0.0, "{metric} minus control median (A.U.)", "delta", "lower left"),
    "ratio": (lambda y, c: y / c, 1.0, "{metric} / control median", "ratio", "best"),
}


def control_normalized_timecourses(wells: list[tuple], tol_min: float, mode: str
                                   ) -> tuple[dict[tuple, dict[float, list[float]]],
                                              dict[tuple, list[float]]]:
    """Each well's activity normalized to the median of the same strain's
    control (0 mM) wells at the same timepoint: mode "delta" subtracts it (net
    drug effect), mode "ratio" divides by it (fold of control).

    Each dose is its own session, so a well's elapsed times are matched to the
    nearest control timepoint within tol_min (unmatched points are dropped, as
    are points whose control median is 0 in ratio mode).
    Returns two dicts: (strain, dose) -> control elapsed time -> [per-well
    normalized activity], so wells from different sessions share one x value
    per timepoint; and (dose, strain, well) -> that well's normalized values in
    time order, for the per-well range traits.
    """
    op = NORMALIZATIONS[mode][0]
    pooled: dict[str, dict[float, list[float]]] = {}
    for dose, conc, strain, well, t, y in wells:
        if conc == 0:
            for tt, yy in zip(t, y):
                pooled.setdefault(strain, {}).setdefault(round(float(tt), 4), []).append(float(yy))
    if not pooled:
        raise SystemExit("[traits] No control (0 mM) dose in --dose_mM; it is needed for "
                         "the control-normalized time-course.")
    ctrl = {s: (np.array(sorted(d)), np.array([np.median(d[k]) for k in sorted(d)]))
            for s, d in pooled.items()}

    out: dict[tuple, dict[float, list[float]]] = {}
    per_well: dict[tuple, list[tuple[float, float]]] = {}
    dropped = set()
    for dose, conc, strain, well, t, y in wells:
        if strain not in ctrl:
            if strain not in dropped:
                dropped.add(strain)
                print(f"[traits]   no control wells for strain {strain!r}: not in the "
                      f"control-normalized time-course")
            continue
        ctrl_t, ctrl_med = ctrl[strain]
        nearest = np.abs(t[:, None] - ctrl_t[None, :]).argmin(axis=1)
        keep = np.abs(t - ctrl_t[nearest]) <= tol_min
        if mode == "ratio":
            keep &= ctrl_med[nearest] != 0
        if not keep.all() and (dose, strain, mode) not in dropped:
            dropped.add((dose, strain, mode))
            print(f"[traits]   {dose}/{strain}: timepoint(s) with no usable control timepoint "
                  f"within {tol_min:g} min were dropped from the {mode} time-course")
        series = out.setdefault((strain, dose), {})
        values = per_well.setdefault((dose, strain, well), [])
        for i in np.flatnonzero(keep):
            norm = float(op(y[i], ctrl_med[nearest[i]]))
            series.setdefault(float(ctrl_t[nearest[i]]), []).append(norm)
            values.append((float(t[i]), norm))
    return out, per_well


def compute_range(curve: list[tuple[float, float]]) -> float:
    """First timepoint - min of a well's normalized curve (NaN if fewer than 2 points)."""
    if len(curve) < 2:
        return float("nan")
    v = [val for _, val in curve]
    return float(v[0] - np.min(v))


def compute_time_to_fraction(curve: list[tuple[float, float]], p: float) -> float:
    """Elapsed time (min) at which the linearly interpolated curve first
    reaches First - p * (First - Min) or below, where First is the value at the
    well's first timepoint: that timepoint if it already qualifies (range 0),
    otherwise the interpolated crossing within the first segment that goes
    from above the level to at or below it. NaN if the curve has fewer than
    2 points.
    """
    if len(curve) < 2:
        return float("nan")
    times = np.array([tt for tt, _ in curve])
    v = np.array([val for _, val in curve])
    threshold = v[0] - p * (v[0] - v.min())
    hit = int(np.flatnonzero(v <= threshold)[0])
    if hit == 0:
        return float(times[0])
    frac = (v[hit - 1] - threshold) / (v[hit - 1] - v[hit])
    return float(times[hit - 1] + frac * (times[hit] - times[hit - 1]))


def plot_control_normalized_timecourse(series: dict, wells: list[tuple], metric: str, mode: str,
                                       output_dir: str = None, show: bool = False) -> None:
    """One panel per strain: median (bars = quartiles across wells) of the
    control-normalized activity vs elapsed time, one line per dose. The
    control line sits at the reference level and its bars show the baseline
    variability.
    """
    _, ref, ylabel, tag, legend_loc = NORMALIZATIONS[mode]
    strains = sorted({s for s, _ in series})
    doses = list(dict.fromkeys(w[0] for w in wells))
    dose_mM = {w[0]: w[1] for w in wells}

    ncols = min(4, len(strains))
    nrows = -(-len(strains) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, strain in zip(axes.flat, strains):
        for idx, dose in enumerate(doses):
            by_time = series.get((strain, dose))
            if not by_time:
                continue
            times = sorted(by_time)
            med = np.array([np.median(by_time[tt]) for tt in times])
            q25 = np.array([np.percentile(by_time[tt], 25) for tt in times])
            q75 = np.array([np.percentile(by_time[tt], 75) for tt in times])
            ax.errorbar(times, med, yerr=[med - q25, q75 - med], color=cmap(idx % cmap.N),
                        marker="o", markersize=4, linewidth=1.6, capsize=3, elinewidth=1.0,
                        label=format_dose_label(dose_mM[dose]))
        ax.axhline(ref, color="black", linewidth=0.9, alpha=0.7, zorder=1)
        ax.set_title(strain, fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(strains):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Elapsed time (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel.format(metric=metric))
    axes.flat[0].legend(title="Dose", loc=legend_loc, fontsize=8)
    plt.tight_layout()

    if output_dir:
        save_path = os.path.join(output_dir, f"curve_traits_{metric}_timecourse_{tag}.png")
        fig.savefig(save_path, dpi=300)
        print(f"[traits] Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_control_normalized_by_dose(series: dict, wells: list[tuple], metric: str, mode: str,
                                    output_dir: str = None, show: bool = False) -> None:
    """One figure per non-control dose with every strain overlaid: median
    control-normalized activity vs elapsed time, one line per strain.
    """
    _, ref, ylabel, tag, _ = NORMALIZATIONS[mode]
    strains = sorted({s for s, _ in series})
    dose_mM = {w[0]: w[1] for w in wells}
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for dose, conc in dose_mM.items():
        if conc == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, strain in enumerate(strains):
            by_time = series.get((strain, dose))
            if not by_time:
                continue
            times = np.array(sorted(by_time))
            med = np.array([np.median(by_time[tt]) for tt in times])
            q25 = np.array([np.percentile(by_time[tt], 25) for tt in times])
            q75 = np.array([np.percentile(by_time[tt], 75) for tt in times])
            ax.errorbar(times, med, yerr=[med - q25, q75 - med],
                        color=cmap(i % cmap.N), marker=markers[i % len(markers)],
                        markersize=5, linewidth=1.6, capsize=2.5, elinewidth=1.0,
                        label=strain)
        ax.axhline(ref, color="black", linewidth=0.9, alpha=0.7, zorder=1)
        ax.set_xlabel("Elapsed time (min)", fontsize=11)
        ax.set_ylabel(ylabel.format(metric=metric), fontsize=11)
        ax.set_title(f"{format_dose_label(conc)} — median ± quartiles across wells")
        ax.grid(alpha=0.3)
        ax.legend(title="Strain", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
        plt.tight_layout()

        if output_dir:
            save_path = os.path.join(output_dir,
                                     f"curve_traits_{metric}_timecourse_{tag}_all_strains_{dose}.png")
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
    parser.add_argument("--slope_points", type=int, default=3,
                        help="Number of leading timepoints used for the slope_init line fit.")
    parser.add_argument("--final_slope_points", type=int, default=5,
                        help="Number of trailing timepoints used for the slope_final line fit.")
    parser.add_argument("--range_fracs", nargs="+", type=float, default=[0.1, 0.5, 0.9],
                        help="Fractions p of the (first timepoint - min) range for the "
                             "t_p<N>_<delta|ratio> traits: first time the linearly interpolated "
                             "normalized curve is at or below first - p * range.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for the trait CSV. Defaults to a 'traits' subfolder "
                             "of --input_dir.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip the AUC summary figure.")
    parser.add_argument("--no_timecourse_plot", action="store_true",
                        help="Skip the control-normalized (delta and ratio) activity time-course figures.")
    parser.add_argument("--time_tol_min", type=float, default=5.0,
                        help="Max difference (min) between a well's timepoint and the control "
                             "timepoint it is matched to in the control-normalized time-course.")
    parser.add_argument("--show", action="store_true",
                        help="Pop up the figure interactively as well as saving it.")
    args = parser.parse_args()

    if len(args.doses) != len(args.dose_mM):
        raise SystemExit(
            f"--doses has {len(args.doses)} entries but --dose_mM has {len(args.dose_mM)}; "
            f"provide exactly one dose (mM) per dose folder, in the same order."
        )

    rows = []
    init_spans, final_spans = [], []
    wells = list(iter_wells(args.input_dir, args.doses, args.dose_mM, args.metric))

    # The per-well range traits need the control; without a control dose they are NaN.
    normalized = {}
    if any(w[1] == 0 for w in wells):
        normalized = {mode: control_normalized_timecourses(wells, args.time_tol_min, mode)
                      for mode in NORMALIZATIONS}
    else:
        print("[traits] No control (0 mM) dose: range_delta / range_ratio will be NaN.")

    for dose, conc, strain, well, t, y in wells:
        auc = compute_auc(t, y)
        A0 = float(y[0])
        init_spans.append(t[min(args.slope_points, len(t)) - 1] - t[0])
        final_spans.append(t[-1] - t[-min(args.final_slope_points, len(t))])
        curves = {mode: normalized[mode][1].get((dose, strain, well), []) if mode in normalized else []
                  for mode in NORMALIZATIONS}
        ranges = {mode: compute_range(curve) for mode, curve in curves.items()}
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
            "slope_init": compute_slope_init(t, y, args.slope_points),
            "slope_final": compute_slope_final(t, y, args.final_slope_points),
            "range_delta": ranges["delta"],
            "range_ratio": ranges["ratio"],
            **{f"t_p{p * 100:g}_{mode}": compute_time_to_fraction(curve, p)
               for mode, curve in curves.items() for p in args.range_fracs},
        })

    output_dir = args.output_dir or os.path.join(args.input_dir, "traits")
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, f"curve_traits_{args.metric}_per_well.csv"), rows)

    if not args.no_plot:
        plot_trait_boxes(rows,
                         [("auc", "AUC (A.U. x min)"),
                          ("auc_norm", "AUC / A0 (min)")],
                         None,
                         f"curve_traits_{args.metric}_auc.png",
                         output_dir=output_dir, show=args.show)
        plot_trait_boxes(rows,
                         [("slope_init", f"Initial slope (A.U./min), first "
                                         f"{np.median(init_spans) / 60:.0f} h")],
                         None,
                         f"curve_traits_{args.metric}_slope_init.png",
                         output_dir=output_dir, show=args.show)
        plot_trait_boxes(rows,
                         [("slope_final", f"Final slope (A.U./min), last "
                                          f"{np.median(final_spans) / 60:.0f} h")],
                         None,
                         f"curve_traits_{args.metric}_slope_final.png",
                         output_dir=output_dir, show=args.show)
        if normalized:
            plot_trait_boxes(rows,
                             [("range_delta", f"First - min of {args.metric} minus control (A.U.)")],
                             None,
                             f"curve_traits_{args.metric}_range_delta.png",
                             output_dir=output_dir, show=args.show, skip_control=True)
            plot_trait_boxes(rows,
                             [("range_ratio", f"First - min of {args.metric} / control")],
                             None,
                             f"curve_traits_{args.metric}_range_ratio.png",
                             output_dir=output_dir, show=args.show, skip_control=True)
            for mode in NORMALIZATIONS:
                plot_trait_boxes(rows,
                                 [(f"t_p{p * 100:g}_{mode}",
                                   f"Time to {args.metric} {mode} <= first - {p:g} x range (min)")
                                  for p in args.range_fracs],
                                 None,
                                 f"curve_traits_{args.metric}_time_to_fraction_{mode}.png",
                                 output_dir=output_dir, show=args.show, skip_control=True)

    if not args.no_timecourse_plot:
        for mode in NORMALIZATIONS:
            series = normalized[mode][0] if normalized else \
                control_normalized_timecourses(wells, args.time_tol_min, mode)[0]
            plot_control_normalized_timecourse(series, wells, args.metric, mode,
                                               output_dir=output_dir, show=args.show)
            plot_control_normalized_by_dose(series, wells, args.metric, mode,
                                            output_dir=output_dir, show=args.show)


if __name__ == "__main__":
    main()
