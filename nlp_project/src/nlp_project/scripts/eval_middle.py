"""
eval_middle.py

Runs Qwen with the distractor placed before sentence n // 2 (n = number of sentences, counting the question).

Everything else (prompt, model, decoding, answer parsing, scoring)
comes from eval_baseline.py, so this condition differs from the clean run
only in the inserted sentence. Each problem uses its one reviewed distractor
from distractors_final.json, the same sentence used at every position.

Usage (from the repo root):
  python nlp_project/src/nlp_project/scripts/eval_middle.py
  python nlp_project/src/nlp_project/scripts/eval_middle.py --limit 4 --out-dir results_smoke

Tables are made separately by make_tables.py.
Results go to results/middle.jsonl. Rerunning resumes where it stopped.
"""

from eval_baseline import main

CONDITION = "middle"  # where the distractor goes; see insert_position() in eval_baseline.py

if __name__ == "__main__":
    main(CONDITION)
