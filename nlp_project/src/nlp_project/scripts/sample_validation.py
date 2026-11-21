"""Randomly choose 50 kept problems for manual review.

Reads data/distractors/distractors_final.json (output of apply_review.py), picks N_SAMPLE
rows with a fixed seed, and saves those problems and their distractors to
data/samples/validation_sample.json.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IN_PATH = DATA_DIR / "distractors" / "distractors_final.json"
OUT_PATH = DATA_DIR / "samples" / "validation_sample.json"
SEED = 11
N_SAMPLE = 50


def main():
    with open(IN_PATH) as f:
        rows = json.load(f)["rows"]

    # Pick the problem ids with a fixed seed so the sample can be reproduced,
    # then sort them so the file is easy to read.
    ids = sorted(random.Random(SEED).sample([row["idx"] for row in rows], N_SAMPLE))

    # Keep only the fields needed for review.
    sample = [
        {
            "idx": row["idx"],
            "question": row["question"],
            "gold": row["gold"],
            "distractor": row["distractor"],
        }
        for row in rows
        if row["idx"] in ids
    ]

    with open(OUT_PATH, "w") as f:
        json.dump({"seed": SEED, "ids": ids, "rows": sample}, f, indent=2)

    print(f"Saved {len(sample)} of {len(rows)} problems to {OUT_PATH} (seed={SEED})")


if __name__ == "__main__":
    main()
