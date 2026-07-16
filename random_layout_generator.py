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


ROW_LABELS = tuple("ABCDEFGH")
NUM_COLUMNS = 12
DEFAULT_WORM_COUNTS = tuple(range(5, 41, 5))
DEFAULT_WELLS_PER_CONDITION = 8


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
    args = parser.parse_args()

    layout = generate_layout(seed=args.seed)
    _print_layout(layout)
    if args.output:
        write_layout_csv(layout, args.output)
        print(f"\nSaved layout to {args.output}")


if __name__ == "__main__":
    main()