"""
make_tables.py

Builds every results table from the four evaluation runs. It never loads the
model. It only reads results/<condition>.jsonl, so you can rerun it as often
as you like, for example after adding a new bin type.

Tables (each one is printed and written as LaTeX to tables/)
  headline   accuracy, change vs. clean, flips, answer changes and McNemar p
             per condition, over all problems
  bins_<x>   accuracy per bin for each bin type in BIN_TYPES, with the change
             from clean within the same bin in parentheses
  bin_ids    which problem ids are in each bin
  flips      problems that were right when clean and wrong with the distractor,
             marking the ones where the model used the distractor's number

It also saves each finished condition as one JSON file, results/<condition>.json
(the .jsonl files stay as the working files the eval scripts append to).

To add a bin type, write a function that takes one row of
distractors_final.json and returns a number, then add an entry to BIN_TYPES.

Usage
  python make_tables.py

Inputs needed in the same folder:
  distractors_final.json, select_problems.py, eval_baseline.py, results/
"""

import json
import math
import os
import textwrap

# Reuse the evaluation code so problems, sentence units and answer numbers
# are read exactly the same way as during the runs.
from eval_baseline import (
    CONDITIONS, LABELS, RESULTS_DIR, distractor_number, load_rows, numbers_in,
    question_units, read_results, results_path,
)

TABLES_DIR = "tables"


# =======================================================================
# 1. Problem properties (one function per bin type)
# =======================================================================
# Every property is measured on the ORIGINAL problem, so a problem sits in
# the same bin under all four conditions.

def problem_length(row: dict) -> int:
    """Sentences in the original problem, counting the question. Uses the same
    units as the distractor positions (a trailing '(Round ...)' note is part
    of the question)."""
    return len(question_units(row["question"]))


_STEPS = None  # filled the first time reasoning_steps() is called


def count_steps(solution: str) -> int:
    """Reasoning steps in a GSM8K solution = number of <<calculator>> annotations.
    A few solutions have none, so fall back to counting solution lines."""
    n = solution.count("<<")
    if n == 0:
        n = len([ln for ln in solution.split("####")[0].splitlines() if ln.strip()])
    return n


def reasoning_steps(row: dict) -> int:
    """Calculations in the GSM8K solution. distractors_final.json has no
    solutions, so GSM8K is downloaded once and the counts are cached in
    results/steps.json."""
    global _STEPS
    if _STEPS is None:
        cache = os.path.join(RESULTS_DIR, "steps.json")
        if os.path.exists(cache):
            with open(cache) as f:
                _STEPS = {int(k): v for k, v in json.load(f).items()}
        else:
            from datasets import load_dataset
            gsm = load_dataset("openai/gsm8k", "main", split="test")
            _STEPS = {}
            for r in load_rows():
                sol = gsm[r["idx"]]["answer"]
                # Sanity check: the saved gold answer must match GSM8K's.
                assert sol.split("####")[-1].strip().replace(",", "") == r["gold"], r["idx"]
                _STEPS[r["idx"]] = count_steps(sol)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(cache, "w") as f:
                json.dump(_STEPS, f)
    return _STEPS[row["idx"]]


# =======================================================================
# 2. Bin types: edit this list to add, remove or re-cut bins
# =======================================================================
# name   used for the LaTeX file name (tables/bins_<name>.tex)
# title  printed above the table
# value  function: one row of distractors_final.json -> a number
# bins   (label, lowest value, highest value), covering every problem

BIN_TYPES = [
    {
        "name": "length",
        "title": "problem length (sentences in the original problem)",
        "value": problem_length,
        "bins": [("3 sentences", 3, 3), ("4 sentences", 4, 4), ("5+ sentences", 5, 99)],
    },
    {
        "name": "steps",
        "title": "reasoning steps (calculations in the GSM8K solution)",
        "value": reasoning_steps,
        "bins": [("1-2 steps", 1, 2), ("3 steps", 3, 3), ("4 steps", 4, 4), ("5+ steps", 5, 99)],
    },
]


def bin_label(value, bins) -> str:
    """Return the label of the bin a value falls into."""
    for label, lo, hi in bins:
        if lo <= value <= hi:
            return label
    raise ValueError(f"value {value} fits no bin; widen the bins in BIN_TYPES")


# =======================================================================
# 3. Statistics
# =======================================================================

def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs
    (b = correct->wrong, c = wrong->correct). A binomial test with p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def spearman(xs, ys) -> float:
    """Spearman rank correlation (average ranks for ties), no scipy needed.
    Used to report how much two bin types overlap."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else float("nan")


# =======================================================================
# 4. Formatting helpers (text and LaTeX)
# =======================================================================

def fmt_p(p: float, tex: bool = False) -> str:
    """p-values below 0.001 are shown as <0.001 instead of 0.000."""
    if p < 0.001:
        return "$<$0.001" if tex else "<0.001"
    return f"{p:.3f}"


def fmt_drop(x: float) -> str:
    return f"{x:+.1f}"


def tex_drop(x: float) -> str:
    """LaTeX needs a real minus sign: -6.5 -> $-$6.5."""
    return ("$-$" if x < 0 else "+") + f"{abs(x):.1f}"


def ids_text(ids) -> str:
    return ", ".join(str(i) for i in sorted(ids)) if ids else "none"


def wrap(text: str) -> str:
    """Indent and wrap a long id list for printing."""
    return textwrap.fill(text, width=100, initial_indent="    ", subsequent_indent="    ")


def print_table(title: str, header: list, rows: list):
    """Print a simple aligned text table."""
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = "  ".join("-" * w for w in widths)
    print(f"\n{title}\n{line}")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print(line)
    for r in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
    print(line)


def write_tex(name: str, colspec: str, header: list, rows: list):
    """Write a booktabs tabular to tables/<name>.tex. In the paper, put it
    inside your own table environment with \\input{tables/<name>.tex} and add
    \\usepackage{booktabs}. Captions and labels go in the paper's .tex file."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    header = [h.replace("%", "\\%") for h in header]  # % starts a comment in LaTeX
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(os.path.join(TABLES_DIR, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines))


# =======================================================================
# 5. Building the tables
# =======================================================================

def load_finished_results(all_ids: list) -> dict:
    """Read each condition's results. A condition counts only when it covers
    every problem, so a half-finished run never shows up in the tables."""
    results = {}
    for c in CONDITIONS:
        recs = read_results(results_path(c))
        if recs and all(i in recs for i in all_ids):
            results[c] = recs
        elif recs:
            print(f"Skipping {c}: only {len(recs)}/{len(all_ids)} problems done")
        else:
            print(f"Skipping {c}: no results yet")
    return results


def main():
    rows = load_rows()
    all_ids = [r["idx"] for r in rows]
    row_by_id = {r["idx"]: r for r in rows}

    results = load_finished_results(all_ids)
    if "baseline" not in results:
        print("Run eval_baseline.py first: every table compares against the clean condition.")
        return
    conds = [c for c in CONDITIONS if c in results]
    positions = [c for c in conds if c != "baseline"]
    clean = results["baseline"]

    def acc(c, ids):
        """Accuracy (%) of condition c on a set of problem ids."""
        return 100 * sum(results[c][i]["correct"] for i in ids) / len(ids)

    # ---- Table 1: headline ---------------------------------------------------
    # Flips: correct when clean -> wrong with the distractor (and the reverse).
    # Answer changed: the parsed answer differs from the clean answer at all,
    # which also catches one wrong answer turning into a different wrong one.
    head = ["Condition", "n", "Accuracy (%)", "Change vs. clean", "Correct to wrong",
            "Wrong to correct", "Answer changed (%)", "McNemar p"]
    txt, tex = [], []
    for c in conds:
        a = acc(c, all_ids)
        if c == "baseline":
            txt.append([LABELS[c], len(all_ids), f"{a:.1f}"] + ["-"] * 5)
            tex.append([LABELS[c], str(len(all_ids)), f"{a:.1f}"] + ["--"] * 5)
            continue
        d = a - acc("baseline", all_ids)
        b = sum(clean[i]["correct"] and not results[c][i]["correct"] for i in all_ids)
        g = sum(not clean[i]["correct"] and results[c][i]["correct"] for i in all_ids)
        changed = 100 * sum(results[c][i]["pred"] != clean[i]["pred"] for i in all_ids) / len(all_ids)
        p = mcnemar_p(b, g)
        txt.append([LABELS[c], len(all_ids), f"{a:.1f}", fmt_drop(d), b, g,
                    f"{changed:.1f}", fmt_p(p)])
        tex.append([LABELS[c], str(len(all_ids)), f"{a:.1f}", tex_drop(d), str(b), str(g),
                    f"{changed:.1f}", fmt_p(p, tex=True)])
    print_table("Table 1. Accuracy by condition (all problems)", head, txt)
    write_tex("headline", "l" + "r" * 7, head, tex)

    # ---- One table per bin type ---------------------------------------------
    # Rows are bins, columns are conditions. Each position cell shows accuracy
    # and, in parentheses, the change from clean within the SAME bin, which
    # separates "this bin is harder" from "distractors hurt more here".
    values = {}       # bin type name -> {problem id: value}
    membership = []   # (bin type name, bin label, ids) for the id table
    for t_num, bt in enumerate(BIN_TYPES, start=2):
        values[bt["name"]] = {i: bt["value"](row_by_id[i]) for i in all_ids}
        label_of = {i: bin_label(v, bt["bins"]) for i, v in values[bt["name"]].items()}

        head = ["Bin", "n"] + [LABELS[c] for c in conds]
        txt, tex = [], []
        for label, _, _ in bt["bins"]:
            ids = [i for i in all_ids if label_of[i] == label]
            membership.append((bt["name"], label, ids))
            if not ids:
                txt.append([label, 0] + ["-"] * len(conds))
                tex.append([label, "0"] + ["--"] * len(conds))
                continue
            base = acc("baseline", ids)
            t_cells, x_cells = [f"{base:.1f}"], [f"{base:.1f}"]
            for c in positions:
                a = acc(c, ids)
                t_cells.append(f"{a:.1f} ({fmt_drop(a - base)})")
                x_cells.append(f"{a:.1f} ({tex_drop(a - base)})")
            txt.append([label, len(ids)] + t_cells)
            tex.append([label, str(len(ids))] + x_cells)
        print_table(f"Table {t_num}. Accuracy by {bt['title']}  "
                    "[accuracy %, change vs. clean in parentheses]", head, txt)
        write_tex(f"bins_{bt['name']}", "lr" + "r" * len(conds), head, tex)
    t_next = len(BIN_TYPES) + 2

    # How much the bin types overlap (e.g. long problems also needing more
    # steps). Worth one sentence in the paper.
    names = list(values)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            x, y = names[a_i], names[b_i]
            rho = spearman([values[x][i] for i in all_ids], [values[y][i] for i in all_ids])
            print(f"\nSpearman correlation between {x} and {y}: {rho:.2f}")

    # ---- Which problems are in each bin --------------------------------------
    print(f"\nTable {t_next}. Problem ids in each bin")
    for name, label, ids in membership:
        print(f"  {name} / {label} (n={len(ids)}):\n{wrap(ids_text(ids))}")
    write_tex("bin_ids", "llrp{0.62\\linewidth}", ["Bin type", "Bin", "n", "Problem ids"],
              [[name.capitalize(), label, str(len(ids)), ids_text(ids)]
               for name, label, ids in membership])

    # ---- Flips, and whether the model used the distractor's number -----------
    # "Used the distractor number": the distractor's number shows up in the
    # model's reasoning for that condition but NOT in its clean reasoning, so
    # the model likely calculated with it rather than getting derailed some
    # other way.
    if positions:
        txt, tex, lists = [], [], []
        for c in positions:
            flips = [i for i in all_ids if clean[i]["correct"] and not results[c][i]["correct"]]
            used = []
            for i in flips:
                num = distractor_number(row_by_id[i]["distractor"])
                if (num is not None and num in numbers_in(results[c][i]["output"])
                        and num not in numbers_in(clean[i]["output"])):
                    used.append(i)
            marked = ", ".join(f"{i}*" if i in used else str(i) for i in flips) or "none"
            txt.append([LABELS[c], len(flips), len(used)])
            tex.append([LABELS[c], str(len(flips)), str(len(used)), marked])
            lists.append((LABELS[c], len(flips), marked))
        head = ["Condition", "Correct to wrong", "Used distractor number", "Problem ids"]
        print_table(f"Table {t_next + 1}. Flips (correct when clean, wrong with the distractor)",
                    head[:3], txt)
        for label, n, marked in lists:
            print(f"  {label} (n={n}, * = used distractor number):\n{wrap(marked)}")
        write_tex("flips", "lrrp{0.5\\linewidth}", head, tex)

    # ---- Answer parsing check -------------------------------------------------
    # Outputs without a 'Final answer' line or cut off at the token limit are
    # worth reading before trusting the numbers.
    for c in conds:
        fb = [i for i in all_ids if results[c][i]["parse"] != "final_answer_line"]
        tr = [i for i in all_ids if results[c][i]["truncated"]]
        if fb or tr:
            print(f"\n[{c}] no 'Final answer' line: {ids_text(fb)}; hit token limit: {ids_text(tr)}")

    # ---- Final results as plain JSON ------------------------------------------
    # The .jsonl files are the working files (append-friendly, so runs can
    # resume). Once a condition is complete, also save it as one JSON file,
    # sorted by problem id, with the run settings and summary numbers on top.
    run_info_path = os.path.join(RESULTS_DIR, "run_info.json")
    run_info = json.load(open(run_info_path)) if os.path.exists(run_info_path) else {}
    for c in conds:
        recs = [results[c][i] for i in sorted(all_ids)]
        out = {
            "condition": c,
            "run_info": run_info,
            "n": len(recs),
            "accuracy": round(acc(c, all_ids), 2),
            "results": recs,
        }
        with open(os.path.join(RESULTS_DIR, f"{c}.json"), "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nFinal results saved as JSON: "
          + ", ".join(f"{RESULTS_DIR}/{c}.json" for c in conds))

    # ---- One row per problem, for your own slicing ----------------------------
    with open(os.path.join(RESULTS_DIR, "per_problem.csv"), "w") as f:
        f.write(",".join(["idx"] + names + [f"correct_{c}" for c in conds]) + "\n")
        for i in all_ids:
            f.write(",".join([str(i)] + [str(values[n][i]) for n in names]
                             + [str(int(results[c][i]["correct"])) for c in conds]) + "\n")
    print(f"\nLaTeX tables written to {TABLES_DIR}/, per-problem results to "
          f"{RESULTS_DIR}/per_problem.csv")


if __name__ == "__main__":
    main()
