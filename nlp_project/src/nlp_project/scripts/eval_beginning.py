"""
eval_beginning.py

Runs Qwen with the distractor placed before the first sentence of the problem.

Everything else (prompt, model, decoding, answer parsing, scoring)
comes from eval_baseline.py, so this condition differs from the clean run
only in the inserted sentence. Each problem uses its one reviewed distractor
from distractors_final.json, the same sentence used at every position.

Usage (from the repo root):
  python nlp_project/src/nlp_project/scripts/eval_beginning.py
  python nlp_project/src/nlp_project/scripts/eval_beginning.py --limit 4 --out-dir results_smoke

Tables are made separately by make_tables.py.
Results go to results/beginning.jsonl. Rerunning resumes where it stopped.
"""

from eval_baseline import main

CONDITION = "beginning"  # where the distractor goes; see insert_position() in eval_baseline.py

if __name__ == "__main__":
    main(CONDITION)
