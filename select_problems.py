"""
select_problems.py

Picks which GSM8K test problems the project uses, ONCE, with a fixed seed,
and saves only their indices to problem_ids.json.

Only problems with at least MIN_SENTENCES sentences (counting the question)
are eligible, so the beginning, middle, and before-question positions are
always distinct. split_sentences() is the one sentence splitter for the whole
project.

Usage:
  python select_problems.py                  # writes problem_ids.json
"""

import argparse
import json
import random
import re

from datasets import load_dataset

# ---------------- Settings ---------------------------------------------
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
SPLIT = "test"
SEED = 11
N_PROBLEMS = 300
MIN_SENTENCES = 3
IDS_PATH = "problem_ids.json"
# -----------------------------------------------------------------------

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "lb.", "lbs.", "oz.",
    "vs.", "e.g.", "i.e.", "approx.", "no.",
}
TIME_ABBREVIATIONS = {"a.m.", "p.m."}


def split_sentences(text: str) -> list[str]:
    """Split a problem into sentences at '.', '!' or '?' followed by a space,
    except after common abbreviations like 'Dr.' or 'lbs.'. 'a.m.' and 'p.m.'
    end a sentence only when the next word is capitalized."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences, start = [], 0
    for m in re.finditer(r"[.!?]+(?= )", text):
        last_word = text[:m.end()].split()[-1].lower()
        if last_word in ABBREVIATIONS:
            continue
        if last_word in TIME_ABBREVIATIONS and not text[m.end() + 1:m.end() + 2].isupper():
            continue
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def load_gsm8k():
    """Load the GSM8K split from Hugging Face."""
    return load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)


def gold_answer(answer_text: str) -> str:
    """GSM8K solutions end with '#### <final answer>'. Return just that number."""
    return answer_text.split("####")[-1].strip().replace(",", "")


def eligible_indices(gsm, min_sentences: int) -> list[int]:
    """Indices of problems with at least min_sentences sentences."""
    return [i for i in range(len(gsm)) if len(split_sentences(gsm[i]["question"])) >= min_sentences]


def select_indices(n: int, seed: int, candidates: list[int]) -> list[int]:
    """Sample n distinct indices from candidates with a fixed seed (same seed -> same problems)."""
    rng = random.Random(seed)  # local RNG so nothing else can change the draw
    return sorted(rng.sample(candidates, n))


def load_selected_problems(ids_path: str = IDS_PATH) -> list[dict]:
    """
    Read problem_ids.json and return the selected problems as dicts:
      {"idx", "question", "solution", "gold"}
    """
    with open(ids_path) as f:
        meta = json.load(f)
    gsm = load_dataset(meta["dataset"], meta["config"], split=meta["split"])
    problems = []
    for i in meta["indices"]:
        row = gsm[i]
        problems.append({
            "idx": i,
            "question": row["question"],
            "solution": row["answer"],      # full worked solution from GSM8K
            "gold": gold_answer(row["answer"]),
        })
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PROBLEMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=IDS_PATH)
    args = parser.parse_args()

    gsm = load_gsm8k()
    candidates = eligible_indices(gsm, MIN_SENTENCES)
    indices = select_indices(args.n, args.seed, candidates)

    # Save the settings alongside the indices so the file documents itself.
    meta = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": SPLIT,
        "seed": args.seed,
        "n": args.n,
        "min_sentences": MIN_SENTENCES,
        "n_eligible": len(candidates),
        "n_total": len(gsm),
        "indices": indices,
    }
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(candidates)} of {len(gsm)} problems have at least {MIN_SENTENCES} sentences. "
          f"Saved {len(indices)} problem indices (seed={args.seed}) to {args.out}")


if __name__ == "__main__":
    main()
