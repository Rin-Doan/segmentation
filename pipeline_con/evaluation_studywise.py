"""
Per-study pipeline evaluation and ranking by Dice Score.

Reuses the same validation split and metric definitions as evaluation.py, but
writes one row per study and reports the 5 studies with the lowest macro Dice
Score (dice_score).

Run on a GPU node via the project venv (from the pipeline/ directory):

    uv run evaluation_studywise.py
"""

import os

import pandas as pd

from evaluation import (
    FIRST_SLICE_CSV,
    ML_DATA_PATH,
    RANDOM_STATE,
    SEGMENTATION_PATH,
    discover_val_studies,
    evaluate,
    load_first_slices,
)

RESULTS_CSV = "./evaluation_studywise_results.csv"
TOP_N = 5


def get_lowest_dice_studies(rows: list[dict], n: int = TOP_N) -> list[dict]:
    """Return the n studies with the lowest macro Dice Score (dice_score)."""
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("dice_score", ascending=True)
    return df.head(n).to_dict("records")


def save_studywise_csv(rows: list[dict], save_path: str = RESULTS_CSV) -> None:
    """Save per-study metrics sorted by Dice Score (lowest first)."""
    df = pd.DataFrame(rows).sort_values("dice_score", ascending=True)
    df.to_csv(save_path, index=False, float_format="%.6f")


def print_lowest_studies(studies: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"TOP {len(studies)} LOWEST STUDIES BY DICE SCORE (validation set)")
    print("=" * 60)
    print(f"  {'Rank':<6} {'Study ID':<40} {'Dice':>8} {'IoU':>8}")
    for rank, row in enumerate(studies, start=1):
        print(
            f"  {rank:<6} {row['study_id']:<40}"
            f" {row['dice_score']:>8.4f} {row['iou']:>8.4f}"
        )
    print("=" * 60)


def main() -> list[dict]:
    print("=" * 60)
    print("Pipeline Study-wise Evaluation — lowest Dice Score studies")
    print("=" * 60)
    print(f"Predictions:   {ML_DATA_PATH}")
    print(f"Ground truth:  {SEGMENTATION_PATH}")
    print(f"First slices:  {FIRST_SLICE_CSV}")
    print()

    if not os.path.isfile(FIRST_SLICE_CSV):
        raise FileNotFoundError(
            f"First-slice CSV not found: {FIRST_SLICE_CSV} (run pipeline.py first)"
        )

    first_slices = load_first_slices()
    overlapping, val_studies = discover_val_studies()
    print(f"Overlapping studies: {len(overlapping)}")
    print(
        f"Validation studies:  {len(val_studies)} "
        f"(test_size=0.2, random_state={RANDOM_STATE})\n"
    )

    rows = evaluate(val_studies, first_slices)
    if not rows:
        print("No volumes evaluated; nothing to report.")
        return []

    save_studywise_csv(rows, RESULTS_CSV)
    print(f"Per-study metrics saved to: {RESULTS_CSV}")

    lowest = get_lowest_dice_studies(rows, TOP_N)
    print_lowest_studies(lowest)
    return lowest


if __name__ == "__main__":
    main()
