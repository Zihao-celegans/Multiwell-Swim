"""Generate a reproducible random layout for a 96-well plate.

Assigns distinct wells to each of a list of named conditions — e.g. worm
counts ("5", "10", ...), drug concentrations ("0 mM", "0.2 mM", ...), or
any other group names. Wells are labelled A1 through H12. The condition
name itself is used as the group key everywhere (layout dict key, CSV
column, plot legend) — order is preserved from the --conditions list you
pass in, not re-sorted, so pass them in the order you want them plotted
(e.g. ascending dose).

Examples:
    Print the default layout (conditions "1".. "8") using seed 123:
        python random_layout_generator.py --seed 123

    Generate a 4-condition drug dose-response layout:
        python random_layout_generator.py --seed 123 \\
            --conditions "0 mM" "0.2 mM" "2 mM" "20 mM"

    Save the layout as CSV:
        python random_layout_generator.py --seed 123 --output layout.csv

    Display a color-coded plate diagram:
        python random_layout_generator.py --seed 123 --plot

    Save the plate diagram to an image file:
        python random_layout_generator.py --seed 123 --plot-output layout.png

    Save a separate plate diagram per condition into a folder:
        python random_layout_generator.py --seed 123 --plot-dir layout_plots
"""

import argparse
import csv
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROW_LABELS = tuple("ABCDEFGH")
NUM_COLUMNS = 12
DEFAULT_CONDITIONS = tuple(str(n) for n in range(1, 9))
DEFAULT_WELLS_PER_CONDITION = 12


def all_wells() -> list[str]:
    """Return all 96 well labels in row-major order."""
    return [
        f"{row}{column}"
        for row in ROW_LABELS
        for column in range(1, NUM_COLUMNS + 1)
    ]


def _well_sort_key(well: str) -> tuple[int, int]:
    """Return the row-major sort key for a well label."""
    return ROW_LABELS.index(well[0]), int(well[1:])


def generate_layout(
    seed: int | None = None,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    wells_per_condition: int = DEFAULT_WELLS_PER_CONDITION,
) -> dict[str, list[str]]:
    """Randomly assign distinct wells to each named condition.

    Args:
        seed: Optional seed used to reproduce the same layout.
        conditions: Condition names/labels to include as experimental groups
            (e.g. worm counts as strings, drug concentrations, or any other
            group names). Order is preserved in the returned dict and used
            by downstream plotting/grouping code (e.g. legend order) — pass
            them in the order you want displayed (e.g. ascending dose).
        wells_per_condition: Number of wells assigned to each condition.

    Returns:
        A dictionary mapping each condition name to its assigned well
        labels, in the same order as `conditions`.

    Raises:
        ValueError: If the requested layout cannot fit on a 96-well plate,
            or condition names are empty/duplicated.
    """
    if not conditions:
        raise ValueError("At least one condition is required.")
    if wells_per_condition < 1:
        raise ValueError("wells_per_condition must be at least 1.")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must not contain duplicate names.")

    total_wells = len(conditions) * wells_per_condition
    plate_wells = all_wells()
    if total_wells > len(plate_wells):
        raise ValueError(
            f"The requested layout needs {total_wells} wells, "
            f"but the plate has only {len(plate_wells)}."
        )

    generator = random.Random(seed)
    selected_wells = generator.sample(plate_wells, total_wells)

    return {
        condition: sorted(
            selected_wells[index:index + wells_per_condition],
            key=_well_sort_key,
        )
        for index, condition in zip(
            range(0, total_wells, wells_per_condition), conditions
        )
    }


def write_layout_csv(layout: dict[str, list[str]], output_path: str | Path) -> None:
    """Write a generated layout to CSV, one well assignment per row."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("condition", "well"))
        for condition, wells in layout.items():
            for well in wells:
                writer.writerow((condition, well))


def plot_layout(
    layout: dict[str, list[str]],
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Draw a color-coded 8x12 plate diagram of the well layout.

    Each condition is assigned a distinct color, and unassigned wells are
    drawn in light gray.

    Args:
        layout: Mapping of condition name to assigned well labels, as
            returned by generate_layout.
        output_path: Optional path to save the figure as an image file.
        show: Whether to display the figure interactively.
    """
    well_to_condition: dict[str, str] = {
        well: condition
        for condition, wells in layout.items()
        for well in wells
    }

    conditions = list(layout.keys())
    cmap = plt.get_cmap("tab10" if len(conditions) <= 10 else "tab20")
    condition_colors = {
        condition: cmap(index % cmap.N)
        for index, condition in enumerate(conditions)
    }
    unassigned_color = "0.9"

    fig, ax = plt.subplots(figsize=(NUM_COLUMNS * 0.7, len(ROW_LABELS) * 0.7))

    for row_index, row_label in enumerate(ROW_LABELS):
        for column in range(1, NUM_COLUMNS + 1):
            well = f"{row_label}{column}"
            condition = well_to_condition.get(well)
            color = condition_colors.get(condition, unassigned_color)
            x, y = column, len(ROW_LABELS) - row_index

            circle = Circle((x, y), 0.4, facecolor=color, edgecolor="black", linewidth=0.8)
            ax.add_patch(circle)
            ax.text(x, y, well, ha="center", va="center", fontsize=7)

    ax.set_xlim(0.3, NUM_COLUMNS + 0.7)
    ax.set_ylim(0.3, len(ROW_LABELS) + 0.7)
    ax.set_xticks(range(1, NUM_COLUMNS + 1))
    ax.set_yticks(range(1, len(ROW_LABELS) + 1))
    ax.set_yticklabels(reversed(ROW_LABELS))
    ax.set_aspect("equal")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_title("96-Well Plate Layout", pad=20)

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=condition_colors[condition],
            markeredgecolor="black",
            markersize=10,
            label=condition,
        )
        for condition in conditions
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Condition",
        borderaxespad=0,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plate diagram to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _slugify(text: str) -> str:
    """Turn a condition name into a filesystem-safe slug for filenames."""
    slug = re.sub(r"[^\w.-]+", "_", text.strip())
    return slug.strip("_") or "condition"


def plot_condition_layouts(layout: dict[str, list[str]], output_dir: str | Path) -> None:
    """Save a separate plate diagram for each condition into a folder.

    Each image highlights only the wells assigned to that condition, with
    all other wells drawn in light gray.

    Args:
        layout: Mapping of condition name to assigned well labels, as
            returned by generate_layout.
        output_dir: Folder to save the per-condition images into. Created
            if it does not already exist.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for condition, wells in layout.items():
        single_condition_layout = {condition: wells}
        image_path = output_path / f"condition_{_slugify(condition)}.png"
        plot_layout(single_condition_layout, output_path=image_path, show=False)


def _print_layout(layout: dict[str, list[str]]) -> None:
    """Print a compact human-readable layout."""
    print("condition    wells")
    print("-----------  " + "-" * 31)
    for condition, wells in layout.items():
        print(f"{condition:>10}  {', '.join(wells)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible random layout for an 8 x 12 plate."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Condition names/labels, in order (e.g. --conditions \"0 mM\" \"0.2 mM\" "
             "\"2 mM\" \"20 mM\"). Can be worm counts, drug concentrations, or any "
             "other group names — order is preserved for plotting/legends. "
             "Default: \"1\" .. \"8\".",
    )
    parser.add_argument(
        "--wells-per-condition",
        type=int,
        default=DEFAULT_WELLS_PER_CONDITION,
        help="Number of wells assigned to each condition (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write the generated layout to.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a color-coded plate diagram of the layout.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional image file (e.g. PNG) to save the plate diagram to.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help=(
            "Optional folder to create and save a separate plate diagram "
            "image into, one per condition."
        ),
    )
    args = parser.parse_args()

    layout = generate_layout(
        seed=args.seed,
        conditions=tuple(args.conditions),
        wells_per_condition=args.wells_per_condition,
    )
    _print_layout(layout)
    if args.output:
        write_layout_csv(layout, args.output)
        print(f"\nSaved layout to {args.output}")

    if args.plot or args.plot_output:
        plot_layout(layout, output_path=args.plot_output, show=args.plot)

    if args.plot_dir:
        plot_condition_layouts(layout, args.plot_dir)
        print(f"Saved per-condition plate diagrams to {args.plot_dir}")


if __name__ == "__main__":
    main()