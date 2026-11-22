"""
eval_baseline.py

Runs Qwen on the CLEAN problems (no distractor). This file also holds all the
shared code for the project's evaluation, so the three position scripts
(eval_beginning.py, eval_middle.py, eval_before_question.py) import from it.
That way every condition uses the exact same prompt, model settings, answer
parsing and scoring, and the only thing that differs is where the distractor
goes.

What it does
  1. Reads distractors_final.json (the reviewed test set).
  2. Builds the problem text for one condition (baseline here).
  3. Runs Qwen with greedy decoding and saves every output to
     results/<condition>.jsonl (it resumes if Colab disconnects).
     When the condition finishes, it also writes
     results/<condition>_YYYY-MM-DD_HH-MM-SSZ.json.

Tables and accuracy bins are made separately by make_tables.py, which only
reads the results files.

Usage (from the repo root):
  python nlp_project/src/nlp_project/scripts/eval_baseline.py
  python nlp_project/src/nlp_project/scripts/eval_baseline.py --limit 4 --out-dir results_smoke

Reads data/distractors/distractors_final.json (located from this file, not the
working directory). Results are written under results/ at the repo root.
"""

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# We reuse the project's one sentence splitter so positions are defined the
# same way as the "at least 3 sentences" filter in select_problems.py.
from select_problems import split_sentences

# ---------------- Settings ---------------------------------------------
# parents: scripts -> package -> src -> nlp_project -> repo root
_PKG = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[4]
DATA_PATH = str(_PKG / "data" / "distractors" / "distractors_final.json")
RESULTS_DIR = str(_REPO / "results")

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"  # pinned Hugging Face commit
BATCH_SIZE = 16         # lower this if Colab runs out of GPU memory
MAX_NEW_TOKENS = 2048   # room for step-by-step reasoning (512 cut off 21% of clean outputs, 2048 cut off 6%)

# The same prompt for every condition. Only {question} changes.
PROMPT = (
    "Solve the following math problem. Think step by step, then give the final "
    "answer on the last line in the form 'Final answer: <number>'.\n\n"
    "Problem: {question}"
)

# The four conditions, in table order, with their display names.
CONDITIONS = ["baseline", "beginning", "middle", "before_question"]
LABELS = {
    "baseline": "Clean (no distractor)",
    "beginning": "Beginning",
    "middle": "Middle",
    "before_question": "Before question",
}

# -----------------------------------------------------------------------


# =======================================================================
# 1. Building the problem text for each condition
# =======================================================================

def question_units(question: str) -> list[str]:
    """Split a problem into sentences. The last unit is the question.

    One special case: a trailing note in parentheses, like
    "(Round to the nearest integer.)" in problem 438, belongs to the
    question, so it is joined onto the sentence before it. Otherwise the
    "before question" distractor would land between the question and its note.
    """
    units = split_sentences(question)
    if len(units) > 1 and units[-1].startswith("("):
        units = units[:-2] + [units[-2] + " " + units[-1]]
    return units


def insert_position(condition: str, n_units: int) -> int:
    """Index of the sentence the distractor goes BEFORE.

    beginning       -> before the first sentence (index 0)
    middle          -> before sentence n // 2
    before_question -> before the last sentence, which is the question
    With at least 3 sentences these are always three different spots.
    """
    if condition == "beginning":
        return 0
    if condition == "middle":
        return n_units // 2
    if condition == "before_question":
        return n_units - 1
    raise ValueError(f"unknown condition: {condition}")


def build_problem_text(question: str, distractor: str, condition: str) -> str:
    """Return the problem text for one condition.

    The clean version is rebuilt from the same split, not taken from the raw
    question, so spacing is identical across conditions and the distractor is
    the only difference.
    """
    units = question_units(question)
    if condition == "baseline":
        return " ".join(units)
    k = insert_position(condition, len(units))
    return " ".join(units[:k] + [distractor] + units[k:])


def load_rows(path: str = DATA_PATH) -> list[dict]:
    """Load the reviewed test set and check every problem can take 3 positions."""
    with open(path) as f:
        rows = json.load(f)["rows"]
    for r in rows:
        n = len(question_units(r["question"]))
        assert n >= 3, f"problem {r['idx']} has only {n} sentences"
    return rows


# =======================================================================
# 2. Running the model
# =======================================================================

def load_model():
    """Load Qwen and its tokenizer for batched, greedy text generation."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"  # decoder-only models pad on the left for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # T4 GPUs (compute capability 7.5) have no native bfloat16, so use float16 there.
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if major >= 8 else torch.float16

    # Qwen3.5 checkpoints include a vision encoder, so they load with the
    # multimodal class. We only ever send text.
    model_cls = (getattr(transformers, "AutoModelForMultimodalLM", None)
                 or getattr(transformers, "AutoModelForImageTextToText", None)
                 or transformers.AutoModelForCausalLM)
    model = model_cls.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, problem_text: str) -> str:
    """Wrap the prompt in Qwen's chat template with thinking mode turned off."""
    messages = [{"role": "user", "content": PROMPT.format(question=problem_text)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate_batch(tokenizer, model, problem_texts: list[str]) -> list[tuple[str, bool]]:
    """Generate answers for a batch. Returns (output text, hit_token_limit) pairs."""
    import torch

    prompts = [format_prompt(tokenizer, t) for t in problem_texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                      # greedy: same input -> same output
            temperature=None, top_p=None, top_k=None,  # silence sampling warnings
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]  # drop the prompt tokens
    results = []
    for seq in new_tokens:
        n_real = int((seq != tokenizer.pad_token_id).sum())
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        results.append((text, n_real >= MAX_NEW_TOKENS))
    return results


# =======================================================================
# 3. Reading answers
# =======================================================================

NUMBER_RE = r"-?\d[\d,]*(?:\.\d+)?"


def to_number(s: str):
    """'1,450,000' -> 1450000.0. Returns None if it isn't a number."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_answer(output: str):
    """Return (predicted number, how it was found).

    Uses the last 'Final answer: <number>' in the output (markdown like
    '**Final answer:** $18' is fine). Strict scoring: if there is no such line,
    for example because the output was cut off at MAX_NEW_TOKENS, there is no
    prediction and the problem counts as wrong. Taking the last number in a
    cut-off output instead can score it correct by luck.
    """
    hits = re.findall(r"final answer[^0-9\-]{0,20}(" + NUMBER_RE + ")", output, flags=re.I)
    if hits:
        return to_number(hits[-1]), "final_answer_line"
    return None, "no_final_answer"


def is_correct(pred, gold: str) -> bool:
    """Numeric match, so '18', '18.0' and '18.00' all count."""
    return pred is not None and math.isclose(pred, float(gold), abs_tol=1e-6)


def numbers_in(text: str) -> set[float]:
    """All numbers that appear in a text, as floats."""
    return {n for n in (to_number(x) for x in re.findall(NUMBER_RE, text)) if n is not None}


def distractor_number(distractor: str):
    """The one number in the distractor ('12,500 houses' -> 12500, '15th' -> 15)."""
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", distractor)
    return to_number(nums[0]) if nums else None


# =======================================================================
# 4. Running one condition (shared by all four scripts)
# =======================================================================

def results_path(condition: str, out_dir: str = RESULTS_DIR) -> str:
    return os.path.join(out_dir, f"{condition}.jsonl")


def read_results(path: str) -> dict[int, dict]:
    """Read a results file into {problem idx: record}. Missing file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {rec["idx"]: rec for rec in (json.loads(line) for line in f if line.strip())}


def write_result_json(condition: str, out_dir: str, rows: list[dict]) -> str:
    """Write one timestamped JSON for the problems already stored in the jsonl.

    The jsonl name stays fixed so a rerun can skip finished problems. This file
    is a snapshot, named with the UTC time it was written.
    """
    questions = {r["idx"]: r["question"] for r in rows}
    recs = read_results(results_path(condition, out_dir))
    exported = [
        {
            "question": questions[idx],
            "idx": idx,
            "gold": recs[idx]["gold"],
            "prediction": recs[idx]["pred"],
            "correct": recs[idx]["correct"],
            "response": recs[idx].get("output"),
        }
        for idx in sorted(recs)
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    dest = os.path.join(out_dir, f"{condition}_{stamp}.json")
    with open(dest, "w") as f:
        json.dump(exported, f, indent=2)
        f.write("\n")
    print(f"[{condition}] wrote {dest}")
    return dest


def run_condition(condition: str, limit: int = None, out_dir: str = RESULTS_DIR):
    """Generate and score answers for one condition, appending to its results file.

    Problems already in the file are skipped, so rerunning after a Colab
    disconnect continues where it stopped. A timestamped JSON is written when
    the condition finishes, including when every problem was already done.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = load_rows()
    selected = rows[:limit] if limit else rows
    path = results_path(condition, out_dir)
    done = read_results(path)
    todo = [r for r in selected if r["idx"] not in done]
    print(f"[{condition}] {len(done)} already done, {len(todo)} to run")

    if todo:
        tokenizer, model = load_model()
        save_run_info(model, out_dir)

        with open(path, "a") as f:
            for start in range(0, len(todo), BATCH_SIZE):
                batch = todo[start:start + BATCH_SIZE]
                texts = [build_problem_text(r["question"], r["distractor"], condition) for r in batch]
                outputs = generate_batch(tokenizer, model, texts)
                for r, (output, truncated) in zip(batch, outputs):
                    pred, how = parse_answer(output)
                    rec = {
                        "idx": r["idx"],
                        "condition": condition,
                        "gold": r["gold"],
                        "pred": pred,
                        "parse": how,
                        "correct": is_correct(pred, r["gold"]),
                        "truncated": truncated,
                        "output": output,
                    }
                    f.write(json.dumps(rec) + "\n")
                f.flush()  # write each batch to disk right away
                print(f"[{condition}] {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")

    write_result_json(condition, out_dir, rows)
    recs = read_results(path)
    if recs:
        acc = 100 * sum(r["correct"] for r in recs.values()) / len(recs)
        print(f"[{condition}] accuracy {acc:.1f}% on {len(recs)} problems")


def save_run_info(model, out_dir: str):
    """Record the exact model, weights commit and settings for the paper."""
    import transformers
    info = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION or getattr(model.config, "_commit_hash", None),
        "transformers": transformers.__version__,
        "dtype": str(model.dtype),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False)",
        "thinking": "disabled (enable_thinking=False)",
        "prompt": PROMPT,
    }
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)


# =======================================================================
# 5. Command line (shared by all four scripts)
# =======================================================================

def main(condition: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only run the first N problems (for testing)")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    run_condition(condition, limit=args.limit, out_dir=args.out_dir)


if __name__ == "__main__":
    main("baseline")
