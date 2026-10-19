'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_inputs, print_table, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Age personalization accuracy from judge verdicts (step 2 of 2). "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Judged result files, directories or globs.",
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # (model, condition, speaker group) -> counters
    counts = defaultdict(
        lambda: {"n": 0, "correct": 0, "child": 0, "adult": 0, "unparsed": 0}
    )
    unjudged = 0

    for _path, items in load_inputs(args.inputs):
        for item in items:
            for model, entry in item.get("generations", {}).items():
                if "output" not in entry:
                    continue
                if "judge_choice" not in entry:
                    unjudged += 1
                    continue

                group = entry["speaker_group"]
                c = counts[(model, entry["condition"], group)]

                choice = entry["judge_choice"]
                c["n"] += 1
                if choice in ("child", "adult"):
                    c[choice] += 1
                else:
                    c["unparsed"] += 1
                c["correct"] += int(choice == group)

    if unjudged:
        print(
            f"[warn] skipped {unjudged} generations without a judge_choice; "
            "run evaluation/age_personalization_judge.py over them first"
        )
    if not counts:
        print("No judged age-personalization results found.")
        return

    rows = []
    for (model, condition, group), c in sorted(counts.items()):
        rows.append(
            {
                "model": model,
                "condition": condition,
                "speaker_group": group,
                "n": c["n"],
                "accuracy": 100.0 * c["correct"] / c["n"],
                "chose_child": 100.0 * c["child"] / c["n"],
                "chose_adult": 100.0 * c["adult"] / c["n"],
                "unparsed": c["unparsed"],
            }
        )

    print_table(
        rows,
        ["model", "condition", "speaker_group", "n", "accuracy", "chose_child",
         "chose_adult", "unparsed"],
        title="Persona accuracy per speaker group (LLM-as-judge)",
    )

    # Mean over the two age groups -- one number per (model, condition).
    by_condition = defaultdict(dict)
    for row in rows:
        by_condition[(row["model"], row["condition"])][row["speaker_group"]] = row

    summary = []
    for (model, condition), per_group in sorted(by_condition.items()):
        accuracies = [r["accuracy"] for r in per_group.values()]
        summary.append(
            {
                "model": model,
                "condition": condition,
                "n": sum(r["n"] for r in per_group.values()),
                "adult": per_group.get("adult", {}).get("accuracy"),
                "child": per_group.get("child", {}).get("accuracy"),
                "mean_accuracy": sum(accuracies) / len(accuracies),
            }
        )

    print_table(
        summary,
        ["model", "condition", "n", "adult", "child", "mean_accuracy"],
        title="Mean over age groups",
    )

    save_json({"per_group": rows, "summary": summary}, args.output_json)


if __name__ == "__main__":
    main()
