"""Apply the manual review decisions to the generated distractors.

Reads data/distractors/distractors.json (output of generate_distractors.py), removes every row
listed in the DISCARD dict below, and writes only the kept rows to
data/distractors/distractors_final.json. The evaluation script uses this file as is.

Review rule: a distractor is discarded if it breaks any of the 5 rules
(R1 to R5) in generate_distractors.py. Distractors are never edited by hand,
so the review cannot make them easier or harder for the model. The DISCARD
dict is the record of what was dropped and why.
"""

import json
from collections import Counter
from pathlib import Path

# Input file from generate_distractors.py, and the output file for evaluation.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IN_PATH = DATA_DIR / "distractors" / "distractors.json"
OUT_PATH = DATA_DIR / "distractors" / "distractors_final.json"

# Problems to drop, keyed by GSM8K test index (the "idx" field in the JSON).
# Start each reason with the rule that was broken, followed by a colon,
# so the discards can be counted by rule. Every idx not listed here is kept.
DISCARD = {
    # 121: "R2: could be counted in the fruit total (changes the answer)",
    # R1: not about same topic as original problem
    92: "R1: house and rooms never mentioned",
    167: "R1: cake and candles never mentioned",
    182: "R1: library and bookshelves never mentioned",
    290: "R1: stadium not implied by counting medals",
    429: "R1: mailbox and inches never mentioned",
    496: "R1: cake and candles never mentioned",
    536: "R1: cake and candles never mentioned",
    578: "R1: an actual train never mentioned",
    687: "R1: nursery and toys never mentioned",
    957: "R1: chairs and room never mentioned",
    1068: "R1: bakery and ovens not implied by sub sandwiches",
    1181: "R1: library and bookshelves never mentioned",
    1292: "R1: garden and fences never mentioned",
    1318: "R1: kitchen and ovens never mentioned",
    # R2: could change the answer
    85: "R2: 12-row quilt, but 11 years means 11 rows",
    121: "R2: basket's bananas could be counted in fruit total",
    187: "R2: savings account interest rate could change the answer",
    198: "R2: the 40 rare cards could be part of Elaine's collection",
    229: "R2: melons becomes ambiguous",
    407: "R2: 1,500 steps could change the answer",
    481: "R2: copper penny could be one of the two pennies",
    483: "R2: the storage device could be read as the portable media player",
    529: "R2: headphone set price does not make sense",
    731: "R2: the box can't have 10 crackers",
    750: "R2: rental shop can't have 85 chairs but allow 100 chairs to be rented",
    831: "R2: shelves could be read as more items to install",
    863: "R2: the box's 45 coins could contribute to the total",
    980: "R2: the 35 colorful beads could be counted in bead total",
    1015: "R2: 12 rows of sweets could change the answer",
    1039: "R2: the comforter can't be 10 feet long",
    1093: "R2: a 15 feet wide ceiling is incompatible",
    1168: "R2: a green rope is a fourth rope",
    1311: "R2: the dock can't be both 200 feet and 120 feet",
    # R3: mentioned people
    586: "R3: mentioned students",
    # R4: no number 
    1157: "R4",
    # R4: number out of range automatic check
    2: "R4",
    15: "R4",
    36: "R4",
    58: "R4",
    76: "R4",
    104: "R4",
    112: "R4",
    118: "R4",
    173: "R4",
    190: "R4",
    208: "R4",
    214: "R4",
    282: "R4",
    340: "R4",
    343: "R4",
    372: "R4",
    444: "R4",
    478: "R4",
    501: "R4",
    533: "R4",
    537: "R4",
    541: "R4",
    601: "R4",
    608: "R4",
    631: "R4",
    641: "R4",
    644: "R4",
    658: "R4",
    697: "R4",
    841: "R4",
    851: "R4",
    920: "R4",
    981: "R4",
    998: "R4",
    1119: "R4",
    1133: "R4",
    1167: "R4",
    1184: "R4",
    1209: "R4",
    1238: "R4",
    1280: "R4",
    1316: "R4",
    # R4: nonsense or unrealistic
    28: "R4: bike has 32 wheels",
    38: "R4: track is 40 miles long",
    103: "R4: ocean is 15 miles wide",
    183: "R4: 120 meters wide track",
    228: "R4: 180 pages long puzzle box",
    252: "R4: forest has 22 trees",
    323: "R4: coffee pot holds 80 milliliters",
    328: "R4: battery with capacity of 40 percent",
    597: "R4: triangle with 4 sides",
    610: "R4: 200 meters wide street",
    682: "R4: sewing machine with 30 stitches per minute",
    713: "R4: 18 shelves in such a small space",
    1055: "R4: insurance company with 150 policies",
    1066: "R4: pizza oven holds 35 pizzas",
    1160: "R4: 1000 filing cabinets in an office",
    1186: "R4: 10 mile long track",
    1230: "R4: a tennis court with 3 nets",
}


def main():
    # Load the generated distractors. The file holds the generation settings
    # plus a list of rows, one per problem.
    with open(IN_PATH) as f:
        data = json.load(f)

    rows = data["rows"]
    ids = {row["idx"] for row in rows}

    # Catch typos: every idx in DISCARD must exist in the input file.
    unknown = set(DISCARD) - ids
    if unknown:
        raise ValueError(f"Not in {IN_PATH}: {sorted(unknown)}")

    # Keep only the rows that were not discarded.
    data["rows"] = [row for row in rows if row["idx"] not in DISCARD]

    # Save the settings and the kept rows to the final file.
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Print counts so you can check that at least 200 problems are kept,
    # and report how many were discarded under each rule.
    print(f"kept: {len(data['rows'])}, discarded: {len(DISCARD)}")
    by_rule = Counter(reason.split(":")[0] for reason in DISCARD.values())
    for rule, count in sorted(by_rule.items()):
        print(f"  {rule}: {count}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
