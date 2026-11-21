"""
generate_distractors.py

Generates ONE in-topic but irrelevant sentence (a "distractor") for each
selected GSM8K problem, using Gemma 4 12B.

Pipeline:
  1. Load the problems chosen by select_problems.py (data/problems/problem_ids.json).
  2. For each problem, make a fresh model call with the same fixed prompt.
  3. Check the sentence against our 5 rules with simple automatic checks.
     If it fails, retry a few times; keep the attempt with the fewest flags.
  4. Save everything to data/distractors/distractors.json (original question + distractor,
     plus gold answer, flags, and how many attempts it took).

Our 5 rules for an "in-topic but irrelevant" sentence:
  R1. About the same topic as a part of the problem, whether that be the objects,
      settings, or units of measurement.
  R2. Misleading or unnecessary for solving the problem, but doesn't change the answer.
  R3. Mentions no person: no names, no personal pronouns (he, she, they, ...), and no
      words for people (e.g. "a customer", "a friend"). It describes an object, place,
      or setting instead, so the problem's pronouns cannot refer to it.
  R4. Has exactly one number, written in digits, that is realistic for what it
      describes, similar in size to one of the problem's numbers (between half and
      double it), and different from every number in the problem and from the answer.
  R5. Avoids phrases that only work after earlier context (e.g. "as well", "more than",
      "a different"), so it reads naturally at the beginning, middle, or right before
      the question.

What the automatic checks can and can't do:
  - R3, R4, R5 are checked with heuristics (names, pronouns, person words, numbers,
    banned phrases).
  - R1, R2, and whether R4's number is realistic need a human: read the flagged rows
    and a sample of the rest.
  - Nothing is deleted automatically. Flags are for manual review.

Usage (from the repo root, after running select_problems.py):
  python nlp_project/src/nlp_project/scripts/generate_distractors.py
"""

import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from select_problems import load_selected_problems

# ---------------- Settings ---------------------------------------------
MODEL_ID = "google/gemma-4-12B-it"
REVISION = "43c9d014b231474fa7212dbb48a762566c2b255d"        # commit hash from the model page to pin the version
SEED = 11                # used only for retries (attempt 1 is greedy, so no randomness)
MAX_NEW_TOKENS = 60      # one sentence is ~20-40 tokens
MAX_ATTEMPTS = 3         # attempt 1 greedy; attempts 2+ sample at RETRY_TEMPERATURE
RETRY_TEMPERATURE = 0.7

# R4: the distractor's number must lie between RANGE_LOW x and RANGE_HIGH x
# of at least one number in the problem.
RANGE_LOW = 0.5
RANGE_HIGH = 2.0

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROGRESS_PATH = DATA_DIR / "distractors" / "distractors_progress.jsonl"  # written row by row (survives disconnects)
OUT_PATH = DATA_DIR / "distractors" / "distractors.json"                 # final file
# -----------------------------------------------------------------------

# The fixed prompt. Every problem gets exactly this text with only {question} and {forbidden} filled in.
PROMPT_TEMPLATE = """You write distractor sentences for grade-school math word problems.

Write ONE sentence that could be inserted into the problem below. It must follow ALL of these rules:
1. It is about the same topic as a part of the problem, whether that be the objects, settings, or units of measurement. For example, if a problem is about buying apples at a market, you could write about the price of pears, the hours a market is open, or the weight of a bag of apples.
2. It is unnecessary for solving the problem and does NOT change the correct answer.
3. It does not mention any person: no names, no pronouns like "he", "she", or "they", and no words for people like "a customer" or "a friend". Describe an object, place, or setting instead.
4. It contains exactly ONE number, written in digits. The number is realistic for what it describes and similar in size to one of the problem's numbers (between half and double that number). It must not be any of these numbers: {forbidden}.
5. It makes sense on its own, whether it is placed at the start of the problem, in the middle, or right before the question. Do not use phrases that depend on earlier sentences, such as "as well", "also", "too", "more than", "another", "other", "different", "separate", "the rest", or "the same".

It must be a statement, not a question.
Output only the sentence, with no quotes or explanation.

Problem:
{question}"""

# R5: phrases that only make sense after earlier context. 
CONTEXT_PHRASES = [
    "as well", "more than", "less than", "fewer than", "also", "too",
    "another", "again", "the rest", "the same", "remaining", "instead",
    "that day", "after that", "these", "those", "both",
    "different", "separate", "other",
]

# Capitalized words that are sentence starters or pronouns, not names (used for R3).
NOT_NAMES = {
    "The", "A", "An", "He", "She", "They", "It", "His", "Her", "Their", "Its",
    "If", "How", "What", "When", "Where", "Which", "Who", "Why", "Each", "Every",
    "In", "On", "At", "For", "After", "Before", "Then", "There", "This", "That",
    "One", "Some", "All", "Most", "During", "While", "Last", "Next", "Today",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "I", "We", "You", "Mr", "Mrs", "Ms", "Dr",
}

NUMBER_RE = re.compile(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")   # 12, 1,200, 3.5, $40 (the $ is ignored)

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]+\b")

PERSONAL_PRONOUN_RE = re.compile(
    r"\b(he|him|his|himself|she|her|hers|herself|they|them|their|theirs|themselves|"
    r"i|me|my|we|us|our|you|your)\b"
)

PERSON_WORDS = {
    "person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "kid", "kids", "child", "children", "baby", "babies", "friend", "friends",
    "customer", "customers", "shopper", "shoppers", "student", "students",
    "teacher", "teachers", "worker", "workers", "employee", "employees",
    "farmer", "farmers", "owner", "owners", "neighbor", "neighbors", "family",
    "mom", "dad", "mother", "father", "parent", "parents", "brother", "sister",
    "son", "daughter", "grandma", "grandpa", "someone", "everyone", "anyone",
    "somebody", "everybody", "player", "players", "team", "coach", "chef",
    "baker", "driver", "visitor", "visitors", "guest", "guests", "tourist", "tourists",
}


# ---------------------------- Checks -----------------------------------

def extract_numbers(text: str, include_words: bool = False) -> list[float]:
    """All numbers written in digits, plus simple number words ('two', 'twelve')
    when include_words is True. The prompt asks the model for digits."""
    nums = []
    for m in NUMBER_RE.findall(text):
        m = m.replace(",", "").replace(" ", "")
        if m:
            nums.append(float(m))
    if include_words:
        for w in re.findall(r"\b[a-z]+\b", text.lower()):
            if w in WORD_NUMBERS:
                nums.append(float(WORD_NUMBERS[w]))
    return nums


def extract_names(text: str) -> set[str]:
    """Rough name detector: capitalized words that aren't common starters."""
    return {w for w in CAP_WORD_RE.findall(text) if w not in NOT_NAMES}


def format_number(x: float) -> str:
    return f"{x:g}"


def forbidden_numbers(question: str, gold: str) -> list[float]:
    nums = extract_numbers(question, include_words=True)
    if gold:
        nums.append(float(gold))
    return sorted(set(nums))


def check_distractor(distractor: str, question: str, gold: str) -> list[str]:
    """Return a list of rule violations (empty list = passed every automatic check)."""
    flags = []
    d_lower = distractor.lower()

    # Basic form: one statement.
    if not distractor:
        flags.append("empty")
        return flags
    if "?" in distractor:
        flags.append("is_question")
    if re.search(r"[.!]\s+[A-Z]", distractor):
        flags.append("multiple_sentences")

    # R3: no person at all.
    d_names = extract_names(distractor)
    if d_names:
        flags.append("R3_mentions_name:" + ",".join(sorted(d_names)))
    pronouns = sorted(set(PERSONAL_PRONOUN_RE.findall(d_lower)))
    if pronouns:
        flags.append("R3_mentions_pronoun:" + ",".join(pronouns))
    person_words = sorted(set(re.findall(r"\b[a-z]+\b", d_lower)) & PERSON_WORDS)
    if person_words:
        flags.append("R3_mentions_person_word:" + ",".join(person_words))

    # R4: exactly one number, different from all problem numbers and the answer, in range.
    d_nums = extract_numbers(distractor)
    q_nums = extract_numbers(question, include_words=True)
    if len(d_nums) != 1:
        flags.append(f"R4_number_count={len(d_nums)}")
    else:
        x = d_nums[0]
        if x in q_nums:
            flags.append("R4_repeats_problem_number")
        if gold and x == float(gold):
            flags.append("R4_equals_gold_answer")   # could mislead the grader/parser
        if q_nums and not any(RANGE_LOW * q <= x <= RANGE_HIGH * q for q in q_nums):
            flags.append("R4_out_of_range:" + ",".join(format_number(q) for q in q_nums))

    # R5: no phrases that depend on earlier context.
    for phrase in CONTEXT_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", d_lower):
            flags.append(f"R5_context_phrase:{phrase}")

    return flags


# ---------------------------- Model ------------------------------------

def load_model():
    """Load Gemma 4 12B in 4-bit. T4 GPUs don't support bfloat16, so compute in float16."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, revision=REVISION, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return processor, model


def generate_once(processor, model, question: str, forbidden: str,
                  attempt: int, idx: int) -> str:
    """One fresh model call: only our prompt is in context (no chat history)."""
    prompt = PROMPT_TEMPLATE.format(question=question, forbidden=forbidden)
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    if attempt == 1:
        gen_kwargs = {"do_sample": False}   # greedy: no randomness
    else:
        # Retries need sampling (greedy would repeat the same output).
        # Seed depends on problem and attempt, so reruns try the same random draws.
        torch.manual_seed(SEED + idx * 10 + attempt)
        gen_kwargs = {"do_sample": True, "temperature": RETRY_TEMPERATURE}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **gen_kwargs)
    text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # Keep the first non-empty line, strip stray quotes.
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[0].strip('"“” ') if lines else ""


# ---------------------------- Main -------------------------------------

def main():
    problems = load_selected_problems()
    print(f"Loaded {len(problems)} problems")

    # Resume support: skip problems already in the progress file (Colab disconnects happen).
    done = {}
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            for line in f:
                row = json.loads(line)
                done[row["idx"]] = row
        print(f"Resuming: {len(done)} already done")

    processor, model = load_model()

    start = time.time()
    with open(PROGRESS_PATH, "a") as progress:
        for n, p in enumerate(problems, 1):
            if p["idx"] in done:
                continue

            # Try up to MAX_ATTEMPTS times; keep the attempt with the fewest flags.
            forbidden = ", ".join(format_number(x) for x in forbidden_numbers(p["question"], p["gold"]))
            best = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                d = generate_once(processor, model, p["question"], forbidden, attempt, p["idx"])
                flags = check_distractor(d, p["question"], p["gold"])
                if best is None or len(flags) < len(best["flags"]):
                    best = {"distractor": d, "flags": flags, "attempts": attempt}
                if not flags:
                    break

            row = {
                "idx": p["idx"],               # GSM8K test index
                "question": p["question"],     # original problem
                "gold": p["gold"],             # correct final answer
                "distractor": best["distractor"],
                "flags": best["flags"],        # [] = passed all automatic checks
                "attempts": best["attempts"],
            }
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            done[p["idx"]] = row

            if n % 20 == 0:
                print(f"{n}/{len(problems)} done, {time.time() - start:.0f}s elapsed")

    # Final file, in the same order as data/problems/problem_ids.json.
    rows = [done[p["idx"]] for p in problems]
    settings = {
        "model": MODEL_ID, "revision": REVISION, "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS, "max_attempts": MAX_ATTEMPTS,
        "retry_temperature": RETRY_TEMPERATURE, "range_low": RANGE_LOW,
        "range_high": RANGE_HIGH, "prompt": PROMPT_TEMPLATE,
    }
    with open(OUT_PATH, "w") as f:
        json.dump({"settings": settings, "rows": rows}, f, indent=2)

    n_flagged = sum(1 for r in rows if r["flags"])
    print(f"Done in {(time.time() - start) / 60:.1f} min. "
          f"{n_flagged}/{len(rows)} rows flagged for review. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
