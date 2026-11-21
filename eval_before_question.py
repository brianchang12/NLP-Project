"""
eval_before_question.py

Runs Qwen with the distractor placed right before the question, the last sentence (GSM-IC's placement).

Everything else (prompt, model, decoding, answer parsing, scoring)
comes from eval_baseline.py, so this condition differs from the clean run
only in the inserted sentence. Each problem uses its one reviewed distractor
from distractors_final.json, the same sentence used at every position.

Usage
  python eval_before_question.py                  # run this condition
  python eval_before_question.py --limit 4 --out-dir results_smoke   # quick test

Tables are made separately by make_tables.py.
Results go to results/before_question.jsonl. Rerunning resumes where it stopped.
"""

from eval_baseline import main

CONDITION = "before_question"  # where the distractor goes; see insert_position() in eval_baseline.py

if __name__ == "__main__":
    main(CONDITION)
