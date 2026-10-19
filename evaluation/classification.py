'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.classification import TASK_CONFIG
from utils import load_inputs, print_table, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demographic classification prediction counts, per input file. "
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

    # (file, model, task) -> Counter of predictions
    per_file = defaultdict(Counter)
    # (model, task, gold group) -> Counter of predictions
    per_gold = defaultdict(Counter)

    for path, items in load_inputs(args.inputs):
        name = os.path.basename(path)
        for item in items:
            for model, entry in item.get("generations", {}).items():
                for task, result in entry.items():
                    # Skip anything that is not a classification result
                    # (e.g. an ASR or personalization generation).
                    if not isinstance(result, dict) or "prediction" not in result:
                        continue

                    prediction = result["prediction"]
                    per_file[(name, model, task)][prediction] += 1
                    if result.get("gold"):
                        per_gold[(model, task, result["gold"])][prediction] += 1

    if not per_file:
        print("No classification results found.")
        return

    # ---- counts per input file, one table per task ----
    file_rows = []
    for task in sorted({key[2] for key in per_file}):
        classes = TASK_CONFIG[task]["classes"] if task in TASK_CONFIG else []
        rows = []
        for (name, model, task_name), counter in sorted(per_file.items()):
            if task_name != task:
                continue
            total = sum(counter.values())
            row = {"file": name, "model": model, "task": task, "n": total}
            for cls in classes:
                row[cls] = counter.get(cls, 0)
            row["unknown"] = sum(v for k, v in counter.items() if k not in classes)
            rows.append(row)

        print_table(
            rows,
            ["file", "model", "task", "n"] + list(classes) + ["unknown"],
            title=f"Predictions per input file [{task}]",
        )
        file_rows.extend(rows)

    # ---- the same counts broken down by the speaker's true group ----
    gold_rows = []
    for (model, task, gold), counter in sorted(per_gold.items()):
        classes = TASK_CONFIG[task]["classes"] if task in TASK_CONFIG else []
        total = sum(counter.values())
        row = {
            "model": model,
            "task": task,
            "gold": gold,
            "n": total,
            "correct": counter.get(gold, 0),
            "accuracy": 100.0 * counter.get(gold, 0) / total if total else None,
        }
        for cls in classes:
            row[cls] = counter.get(cls, 0)
        row["unknown"] = sum(v for k, v in counter.items() if k not in classes)
        gold_rows.append(row)

    if gold_rows:
        all_classes = []
        for row in gold_rows:
            for cls in TASK_CONFIG.get(row["task"], {}).get("classes", []):
                if cls not in all_classes:
                    all_classes.append(cls)
        print_table(
            gold_rows,
            ["model", "task", "gold", "n"] + all_classes + ["unknown", "accuracy"],
            title="Predictions by the speaker's true group (accuracy in %)",
        )
    else:
        print(
            "\nNo gold labels recorded, so no per-group breakdown. Pass --gold "
            "to generation/classification.py to get one."
        )

    save_json({"per_file": file_rows, "per_gold_group": gold_rows}, args.output_json)


if __name__ == "__main__":
    main()
