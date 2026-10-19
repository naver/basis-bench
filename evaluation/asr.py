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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Word error rate, grouped by a field recorded at generation time. "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result files, directories or globs.",
    )
    parser.add_argument(
        "--gold_key",
        default="context",
        help="Item field holding the reference transcript. Default `context` "
        "suits the Spoken StereoSet+ ASR split; pass `question` for SD-QA.",
    )
    parser.add_argument(
        "--group_key",
        default="age_group",
        help="Generation-entry field to group by. Default `age_group` suits "
        "the Spoken StereoSet+ ASR split; pass `dialect` for SD-QA.",
    )
    parser.add_argument("--wer_cap", type=float, default=100.0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--write_back",
        action="store_true",
        help="Also write per-utterance wer/edits/ref_len back into the inputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from jiwer import process_words
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as error:
        raise SystemExit(
            "evaluation/asr.py needs jiwer and whisper-normalizer:\n"
            f"    pip install jiwer whisper-normalizer\n({error})"
        )

    normalize = EnglishTextNormalizer()

    # (model, group) -> totals
    totals = defaultdict(lambda: {"n": 0, "edits": 0, "ref_len": 0, "wers": []})
    empty_references = 0

    for path, items in load_inputs(args.inputs):
        for item in items:
            if args.gold_key not in item:
                continue
            reference = normalize(str(item[args.gold_key]))

            for model, entry in item.get("generations", {}).items():
                if "output" not in entry:
                    continue

                if not reference.strip():
                    empty_references += 1
                    entry["wer"] = None
                    continue

                hypothesis = normalize(str(entry["output"]))
                out = process_words(reference, hypothesis)
                edits = out.substitutions + out.insertions + out.deletions
                ref_len = out.substitutions + out.deletions + out.hits
                wer = 100.0 * edits / ref_len if ref_len else 0.0

                entry["wer"] = round(wer, 1)
                entry["edits"] = edits
                entry["ref_len"] = ref_len

                t = totals[(model, entry.get(args.group_key, "unknown"))]
                t["n"] += 1
                t["edits"] += edits
                t["ref_len"] += ref_len
                t["wers"].append(min(wer, args.wer_cap))

        if args.write_back:
            with open(path, "w") as f:
                json.dump(items, f, indent=4, ensure_ascii=False)

    if empty_references:
        print(f"[warn] {empty_references} items have an empty reference transcript")
    if not totals:
        print("No ASR results found.")
        return

    # The group column is named after --group_key, so the table reads
    # "age_group" for Spoken StereoSet+ and "dialect" for SD-QA.
    group_column = args.group_key

    rows = []
    for (model, group), t in sorted(totals.items()):
        mean_wer = sum(t["wers"]) / len(t["wers"])
        rows.append(
            {
                "model": model,
                group_column: group,
                "n": t["n"],
                "corpus_wer": 100.0 * t["edits"] / t["ref_len"] if t["ref_len"] else None,
                "mean_wer": mean_wer,
                "accuracy": args.wer_cap - mean_wer,
            }
        )

    print_table(
        rows,
        ["model", group_column, "n", "corpus_wer", "mean_wer", "accuracy"],
        title=f"Word error rate per {group_column} "
        f"(accuracy = 100 - capped mean WER)",
    )

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    summary = []
    for model, model_rows in sorted(by_model.items()):
        accuracies = [r["accuracy"] for r in model_rows]
        best = max(model_rows, key=lambda r: r["accuracy"])
        worst = min(model_rows, key=lambda r: r["accuracy"])
        summary.append(
            {
                "model": model,
                "groups": ", ".join(r[group_column] for r in model_rows),
                "n": sum(r["n"] for r in model_rows),
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "best": f"{best[group_column]} ({best['accuracy']:.1f})",
                "worst": f"{worst[group_column]} ({worst['accuracy']:.1f})",
                "gap": best["accuracy"] - worst["accuracy"],
            }
        )

    print_table(
        summary,
        ["model", "groups", "n", "mean_accuracy", "best", "worst", "gap"],
        title=f"Mean over {group_column}s, and the gap between best and worst",
    )

    save_json({"per_group": rows, "summary": summary}, args.output_json)


if __name__ == "__main__":
    main()
