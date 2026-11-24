"""
check_snapshots.py

Checks that every snapshot in data/results/snapshots matches the jsonl it was
exported from. The snapshot name starts with the condition
(baseline, beginning, middle, before_question), and that condition's
results/<condition>.jsonl is the base file.

These fields must agree for every problem id:
  snapshot idx          == jsonl idx
  snapshot gold         == jsonl gold
  snapshot prediction   == jsonl pred
  snapshot correct      == jsonl correct
  snapshot response     == jsonl output

Usage (from the repo root):
  python nlp_project/src/nlp_project/scripts/check_snapshots.py
"""

import json
import sys
from pathlib import Path

# Same names as CONDITIONS in eval_baseline.py. Listed here so this script
# does not import the evaluation stack.
CONDITIONS = ["baseline", "beginning", "middle", "before_question"]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"

# Fields copied from the jsonl into the snapshot by write_result_json.
FIELD_PAIRS = (
    ("gold", "gold"),
    ("prediction", "pred"),
    ("correct", "correct"),
    ("response", "output"),
)


def condition_of(path: Path) -> str | None:
    """Condition encoded in a snapshot name, longest match first."""
    stem = path.stem
    for condition in sorted(CONDITIONS, key=len, reverse=True):
        if stem == condition or stem.startswith(condition + "_"):
            return condition
    return None


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows = []
    errors = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON ({exc})")
    return rows, errors


def index_rows(rows: list[dict], label: str) -> tuple[dict[int, dict], list[str]]:
    by_idx = {}
    errors = []
    for row in rows:
        idx = row.get("idx")
        if not isinstance(idx, int):
            errors.append(f"{label}: record has no integer idx ({idx!r})")
            continue
        if idx in by_idx:
            errors.append(f"{label}: duplicate idx {idx}")
            continue
        by_idx[idx] = row
    return by_idx, errors


def compare(snapshot_path: Path, base_path: Path) -> list[str]:
    """Return a list of mismatch descriptions. Empty means the files agree."""
    problems = []
    if not base_path.exists():
        return [f"base file missing: {base_path}"]

    try:
        snapshot_rows = json.loads(snapshot_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid JSON ({exc})"]
    if not isinstance(snapshot_rows, list):
        return ["snapshot is not a JSON list"]

    base_rows, errors = read_jsonl(base_path)
    problems.extend(errors)

    snap, snap_errors = index_rows(snapshot_rows, snapshot_path.name)
    base, base_errors = index_rows(base_rows, base_path.name)
    problems.extend(snap_errors)
    problems.extend(base_errors)

    missing = sorted(set(base) - set(snap))
    extra = sorted(set(snap) - set(base))
    if missing:
        problems.append(f"missing from snapshot ({len(missing)}): {missing}")
    if extra:
        problems.append(f"not in base file ({len(extra)}): {extra}")

    for idx in sorted(set(snap) & set(base)):
        for snap_key, base_key in FIELD_PAIRS:
            snap_val = snap[idx].get(snap_key)
            base_val = base[idx].get(base_key)
            if snap_val != base_val:
                problems.append(
                    f"idx {idx}: {snap_key}={snap_val!r} but {base_path.name} "
                    f"{base_key}={base_val!r}"
                )
    return problems


def main() -> int:
    if not SNAPSHOTS_DIR.is_dir():
        print(f"No snapshots directory: {SNAPSHOTS_DIR}")
        return 1

    snapshots = sorted(SNAPSHOTS_DIR.glob("*.json"))
    if not snapshots:
        print(f"No snapshots in {SNAPSHOTS_DIR}")
        return 1

    failed = 0
    for path in snapshots:
        condition = condition_of(path)
        if condition is None:
            print(f"FAIL  {path.name}: name does not start with a known condition")
            failed += 1
            continue
        base_path = RESULTS_DIR / f"{condition}.jsonl"
        problems = compare(path, base_path)
        if problems:
            failed += 1
            print(f"FAIL  {path.name} vs {base_path.name} ({len(problems)} problems)")
            for problem in problems[:20]:
                print(f"      {problem}")
            if len(problems) > 20:
                print(f"      ... {len(problems) - 20} more")
        else:
            n = len(json.loads(path.read_text()))
            print(f"OK    {path.name} vs {base_path.name} ({n} records)")

    print(f"\n{len(snapshots) - failed}/{len(snapshots)} snapshots match their base file")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
