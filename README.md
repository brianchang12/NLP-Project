# NLP Project

UT Austin NLP class project. It builds a fixed set of GSM8K test problems and adds one in-topic but irrelevant sentence (a distractor) to each, for later evaluation of whether that sentence changes model answers.

The distractor must follow five rules: same topic (R1), does not change the answer (R2), mentions no person (R3), exactly one realistic number (R4), and reads naturally without earlier context (R5). Automatic checks cover R3–R5. R1, R2, and whether the number is realistic are reviewed by hand. Distractors are never edited; a review only keeps or drops a row.

## Layout

```
nlp_project/src/nlp_project/
  scripts/          select problems, generate, apply review, sample, evaluate
  data/
    problems/       problem_ids.json
    distractors/    generated rows, review output, and resume log
    samples/        validation_sample.json
notebooks/          Colab runners for the same scripts
```

Scripts locate `data/` from their own path, so they can be run from the repo root.

## Pipeline

Run these in order from the repo root.

1. **Select problems** (once). From GSM8K `main` / `test`, keep problems with at least 3 sentences and save 300 indices (seed 11).

   ```bash
   python nlp_project/src/nlp_project/scripts/select_problems.py
   ```

   Writes `nlp_project/src/nlp_project/data/problems/problem_ids.json`.

2. **Generate distractors.** One Gemma 4 12B call per problem (`google/gemma-4-12B-it`, pinned revision). Failed automatic checks are retried up to 3 times. Progress is appended row by row, so a rerun resumes.

   ```bash
   python nlp_project/src/nlp_project/scripts/generate_distractors.py
   ```

   Writes `data/distractors/distractors_progress.jsonl` and then `data/distractors/distractors.json`.

3. **Apply review.** Drops the GSM8K indices listed in `DISCARD` inside `apply_review.py` and writes the kept rows.

   ```bash
   python nlp_project/src/nlp_project/scripts/apply_review.py
   ```

   Writes `data/distractors/distractors_final.json`.

4. **Sample for manual validation.** Draws 50 kept problems (seed 11).

   ```bash
   python nlp_project/src/nlp_project/scripts/sample_validation.py
   ```

   Writes `data/samples/validation_sample.json`.

5. **Evaluate.** Run Qwen3.5-2B on the clean problems and on the same distractor at the beginning, middle, and just before the question. Each script resumes from `results/<condition>.jsonl`. Then build the tables.

   ```bash
   python nlp_project/src/nlp_project/scripts/eval_baseline.py
   python nlp_project/src/nlp_project/scripts/eval_beginning.py
   python nlp_project/src/nlp_project/scripts/eval_middle.py
   python nlp_project/src/nlp_project/scripts/eval_before_question.py
   python nlp_project/src/nlp_project/scripts/make_tables.py
   ```

   Writes `results/` and `tables/` at the repo root. `make_tables.py` only reads those result files.

## Notebooks

`notebooks/` has the Colab versions of the same steps:

- `run_select_problems.ipynb`
- `run_generate_distractors.ipynb` (GPU; set `HF_TOKEN` in Colab secrets)
- `run_review.ipynb` (apply review, then sample)
- `run_qwen_eval.ipynb` (GPU; the four eval scripts, then `make_tables.py`)

Each notebook mounts Drive, changes into the project directory, and runs the script paths above.

## Setup

Python package metadata is in `pyproject.toml`. The scripts need PyTorch, Transformers, and `datasets`. Generation also needs `accelerate` and `bitsandbytes` when loading Gemma in 4-bit, which is what the generation notebook installs:

```bash
pip install -U "transformers[torch]" datasets accelerate bitsandbytes
```
