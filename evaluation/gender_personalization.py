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
        description="Gender personalization accuracy (BASIS-Health / BASIS-Product). "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result files, directories or globs.",
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # (model, condition, speaker gender) -> counters
    counts = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "unparsed": 0,
            "selected_male": 0,
            "selected_female": 0,
            "explicit_n": 0,
            "follows_signal": 0,
        }
    )

    for _path, items in load_inputs(args.inputs):
        for item in items:
            for model, per_order in item.get("evaluation", {}).items():
                for _order, record in per_order.items():
                    key = (model, record["condition"], record["speaker_gender"])
                    c = counts[key]

                    c["n"] += 1
                    c["correct"] += int(record["correct"])

                    selected = record["option_selected"]
                    if selected is None:
                        c["unparsed"] += 1
                    else:
                        c[f"selected_{selected}"] += 1

                    if record["signal"] == "explicit":
                        c["explicit_n"] += 1
                        c["follows_signal"] += int(record.get("follows_signal", False))

    if not counts:
        print("No gender-personalization results found.")
        return

    rows = []
    for (model, condition, gender), c in sorted(counts.items()):
        rows.append(
            {
                "model": model,
                "condition": condition,
                "speaker_gender": gender,
                "n": c["n"],
                "accuracy": 100.0 * c["correct"] / c["n"],
                "female_pct": 100.0 * c["selected_female"] / c["n"],
                "unparsed": c["unparsed"],
                "follows_signal": (
                    100.0 * c["follows_signal"] / c["explicit_n"]
                    if c["explicit_n"]
                    else None
                ),
            }
        )

    print_table(
        rows,
        ["model", "condition", "speaker_gender", "n", "accuracy", "female_pct",
         "unparsed", "follows_signal"],
        title="Accuracy per speaker gender (each option order counted separately)",
    )

    # Mean over the two speaker genders -- one number per (model, condition).
    by_condition = defaultdict(dict)
    for row in rows:
        by_condition[(row["model"], row["condition"])][row["speaker_gender"]] = row

    summary = []
    for (model, condition), per_gender in sorted(by_condition.items()):
        accuracies = [r["accuracy"] for r in per_gender.values()]
        summary.append(
            {
                "model": model,
                "condition": condition,
                "n": sum(r["n"] for r in per_gender.values()),
                "female": per_gender.get("female", {}).get("accuracy"),
                "male": per_gender.get("male", {}).get("accuracy"),
                "mean_accuracy": sum(accuracies) / len(accuracies),
            }
        )

    print_table(
        summary,
        ["model", "condition", "n", "female", "male", "mean_accuracy"],
        title="Mean over speaker genders",
    )

    save_json({"per_gender": rows, "summary": summary}, args.output_json)


if __name__ == "__main__":
    main()
