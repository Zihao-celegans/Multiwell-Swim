"""Generate a reproducible random layout for a 96-well plate.

The default layout assigns eight distinct wells to each worm-count condition
of 5, 10, ..., 40 worms. Wells are labelled A1 through H12.

Examples:
    Print a layout using seed 123:
        python random_layout_generator.py --seed 123

    Save the layout as CSV:
        python random_layout_generator.py --seed 123 --output layout.csv
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROW_LABELS = tuple("ABCDEFGH")
NUM_COLUMNS = 12
DEFAULT_WORM_COUNTS = tuple(range(5, 16, 5))
DEFAULT_WELLS_PER_CONDITION = 24


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
    worm_counts: tuple[int, ...] = DEFAULT_WORM_COUNTS,
    wells_per_condition: int = DEFAULT_WELLS_PER_CONDITION,
) -> dict[int, list[str]]:
    """Randomly assign distinct wells to each worm-count condition.

    Args:
        seed: Optional seed used to reproduce the same layout.
        worm_counts: Worm counts to include as experimental conditions.
        wells_per_condition: Number of wells assigned to each condition.

    Returns:
        A dictionary mapping each worm count to its assigned well labels.

    Raises:
        ValueError: If the requested layout cannot fit on a 96-well plate.
    """
    if not worm_counts:
        raise ValueError("At least one worm-count condition is required.")
    if wells_per_condition < 1:
        raise ValueError("wells_per_condition must be at least 1.")
    if any(worm_count < 1 for worm_count in worm_counts):
        raise ValueError("Worm counts must be positive integers.")
    if len(set(worm_counts)) != len(worm_counts):
        raise ValueError("worm_counts must not contain duplicates.")

    total_wells = len(worm_counts) * wells_per_condition
    plate_wells = all_wells()
    if total_wells > len(plate_wells):
        raise ValueError(
            f"The requested layout needs {total_wells} wells, "
            f"but the plate has only {len(plate_wells)}."
        )

    generator = random.Random(seed)
    selected_wells = generator.sample(plate_wells, total_wells)

    return {
        worm_count: sorted(
            selected_wells[index:index + wells_per_condition],
            key=_well_sort_key,
        )
        for index, worm_count in zip(
            range(0, total_wells, wells_per_condition), worm_counts
        )
    }


def write_layout_csv(layout: dict[int, list[str]], output_path: str | Path) -> None:
    """Write a generated layout to CSV, one well assignment per row."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("worm_count", "well"))
        for worm_count, wells in layout.items():
            for well in wells:
                writer.writerow((worm_count, well))


def plot_layout(
    layout: dict[int, list[str]],
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Draw a color-coded 8x12 plate diagram of the well layout.

    Each worm-count condition is assigned a distinct color, and unassigned
    wells are drawn in light gray.

    Args:
        layout: Mapping of worm count to assigned well labels, as returned
            by generate_layout.
        output_path: Optional path to save the figure as an image file.
        show: Whether to display the figure interactively.
    """
    well_to_condition: dict[str, int] = {
        well: worm_count
        for worm_count, wells in layout.items()
        for well in wells
    }

    conditions = list(layout.keys())
    cmap = plt.get_cmap("tab10" if len(conditions) <= 10 else "tab20")
    condition_colors = {
        worm_count: cmap(index % cmap.N)
        for index, worm_count in enumerate(conditions)
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
            markerfacecolor=condition_colors[worm_count],
            markeredgecolor="black",
            markersize=10,
            label=f"{worm_count} worms",
        )
        for worm_count in conditions
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


def _print_layout(layout: dict[int, list[str]]) -> None:
    """Print a compact human-readable layout."""
    print("worm_count  wells")
    print("-----------  " + "-" * 31)
    for worm_count, wells in layout.items():
        print(f"{worm_count:>10}  {', '.join(wells)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible random layout for an 8 x 12 plate."
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
    args = parser.parse_args()

    layout = generate_layout(seed=args.seed)
    _print_layout(layout)
    if args.output:
        write_layout_csv(layout, args.output)
        print(f"\nSaved layout to {args.output}")

    if args.plot or args.plot_output:
        plot_layout(layout, output_path=args.plot_output, show=args.plot)


if __name__ == "__main__":
    main()