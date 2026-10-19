'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_inputs, print_table, save_json

# The short codes used in the released id -> gender maps.
GENDER_LABELS = {"M": "male", "F": "female"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spoken QA accuracy per group from judge verdicts (step 2 of 2). "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Judged SQA result files, directories or globs.",
    )
    parser.add_argument(
        "--id_to_gender",
        default=None,
        help="LibriSQA only: path to id_2_gender_part1.json. When given, "
        "answers are grouped by the speaker's gold gender instead of by "
        "dialect.",
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    id_to_gender = {}
    if args.id_to_gender:
        with open(args.id_to_gender) as f:
            raw = json.load(f)
        id_to_gender = {
            key: GENDER_LABELS.get(code, code) for key, code in raw.items()
        }
        print(f"Loaded {len(id_to_gender)} gold genders from {args.id_to_gender}")

    group_column = "gender" if id_to_gender else "dialect"

    # (model, group) -> counters
    counts = defaultdict(lambda: {"n": 0, "correct": 0, "unparsed": 0})
    unjudged = 0
    unmapped = 0

    for _path, items in load_inputs(args.inputs):
        for item in items:
            if id_to_gender:
                group = id_to_gender.get(item.get("id"))
                if group is None:
                    unmapped += 1
                    continue
            else:
                group = None  # taken per entry below

            for model, entry in item.get("generations", {}).items():
                if "output" not in entry:
                    continue
                if "evaluation_verdict" not in entry:
                    unjudged += 1
                    continue

                key = group if id_to_gender else entry.get("dialect", "unknown")
                c = counts[(model, key)]
                verdict = entry["evaluation_verdict"]
                c["n"] += 1
                if verdict is None:
                    c["unparsed"] += 1
                else:
                    c["correct"] += int(verdict)

    if unjudged:
        print(
            f"[warn] skipped {unjudged} answers without an evaluation_verdict; "
            "run evaluation/sqa_judge.py over them first"
        )
    if unmapped:
        print(f"[warn] skipped {unmapped} items whose id is not in the gender map")
    if not counts:
        print("No judged SQA results found.")
        return

    rows = [
        {
            "model": model,
            group_column: group,
            "n": c["n"],
            "correct": c["correct"],
            "unparsed": c["unparsed"],
            "accuracy": 100.0 * c["correct"] / c["n"] if c["n"] else None,
        }
        for (model, group), c in sorted(counts.items())
    ]

    print_table(
        rows,
        ["model", group_column, "n", "correct", "unparsed", "accuracy"],
        title=f"Spoken QA accuracy per {group_column} (LLM-as-judge)",
    )

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    summary = []
    for model, model_rows in sorted(by_model.items()):
        accuracies = [r["accuracy"] for r in model_rows if r["accuracy"] is not None]
        best = max(model_rows, key=lambda r: r["accuracy"])
        worst = min(model_rows, key=lambda r: r["accuracy"])
        summary.append(
            {
                "model": model,
                "groups": len(model_rows),
                "n": sum(r["n"] for r in model_rows),
                "overall_accuracy": (
                    100.0 * sum(r["correct"] for r in model_rows)
                    / sum(r["n"] for r in model_rows)
                    if sum(r["n"] for r in model_rows)
                    else None
                ),
                "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
                "best": f"{best[group_column]} ({best['accuracy']:.1f})",
                "worst": f"{worst[group_column]} ({worst['accuracy']:.1f})",
                "gap": best["accuracy"] - worst["accuracy"],
            }
        )

    print_table(
        summary,
        ["model", "groups", "n", "overall_accuracy", "mean_accuracy",
         "best", "worst", "gap"],
        title=f"Mean over {group_column}s, and the gap between best and worst",
    )

    save_json({"per_group": rows, "summary": summary}, args.output_json)


if __name__ == "__main__":
    main()
