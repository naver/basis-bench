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

# Demographic fields on each item that can be grouped over.
DEMOGRAPHIC_FIELDS = ["gender", "first_language", "age", "ethnicity"]
DEFAULT_GROUP_BY = ["gender", "first_language"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fair-Speech word error rate by speaker demographic. "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Fair-Speech ASR result files, directories or globs.",
    )
    parser.add_argument(
        "--group_by",
        nargs="+",
        default=DEFAULT_GROUP_BY,
        choices=DEMOGRAPHIC_FIELDS,
        help=f"Demographic fields to aggregate over (default: "
        f"{' '.join(DEFAULT_GROUP_BY)}).",
    )
    parser.add_argument(
        "--gold_key",
        default="transcript",
        help="Item field holding the reference transcript (default: transcript).",
    )
    parser.add_argument(
        "--raw_language",
        action="store_true",
        help="Report all 30 first_language values instead of bucketing them "
        "into English / non-English.",
    )
    parser.add_argument("--wer_cap", type=float, default=100.0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--write_back",
        action="store_true",
        help="Also write per-utterance wer/edits/ref_len back into the inputs.",
    )
    return parser.parse_args()


def language_bucket(value, raw):
    """English / non-English, unless --raw_language."""
    if raw:
        return value
    return "English" if str(value).strip().lower() == "english" else "non-English"


def main():
    args = parse_args()

    try:
        from jiwer import process_words
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as error:
        raise SystemExit(
            "evaluation/asr_fairspeech.py needs jiwer and whisper-normalizer:\n"
            f"    pip install jiwer whisper-normalizer\n({error})"
        )

    normalize = EnglishTextNormalizer()

    # (model, field, group value) -> totals, plus ("", "overall", "all")
    totals = defaultdict(lambda: {"n": 0, "edits": 0, "ref_len": 0, "wers": []})
    empty_references = 0
    scored = 0

    for path, items in load_inputs(args.inputs):
        touched = False
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
                    touched = True
                    continue

                hypothesis = normalize(str(entry["output"]))
                out = process_words(reference, hypothesis)
                edits = out.substitutions + out.insertions + out.deletions
                ref_len = out.substitutions + out.deletions + out.hits
                wer = 100.0 * edits / ref_len if ref_len else 0.0

                entry["wer"] = round(wer, 1)
                entry["edits"] = edits
                entry["ref_len"] = ref_len
                touched = True
                scored += 1

                buckets = [(model, "overall", "all")]
                for field in args.group_by:
                    value = item.get(field)
                    if value is None:
                        continue
                    if field == "first_language":
                        value = language_bucket(value, args.raw_language)
                    buckets.append((model, field, value))

                for key in buckets:
                    t = totals[key]
                    t["n"] += 1
                    t["edits"] += edits
                    t["ref_len"] += ref_len
                    t["wers"].append(min(wer, args.wer_cap))

        if touched and args.write_back:
            with open(path, "w") as f:
                json.dump(items, f, indent=4, ensure_ascii=False)

    if empty_references:
        print(f"[warn] skipped {empty_references} items with a blank reference")
    if not totals:
        print("No Fair-Speech ASR results found.")
        return
    print(f"Scored {scored} utterances.")

    rows = []
    for (model, field, group), t in totals.items():
        mean_wer = sum(t["wers"]) / len(t["wers"])
        rows.append(
            {
                "model": model,
                "grouping": field,
                "group": group,
                "n": t["n"],
                "corpus_wer": 100.0 * t["edits"] / t["ref_len"] if t["ref_len"] else None,
                "mean_wer": mean_wer,
                "accuracy": args.wer_cap - mean_wer,
            }
        )

    # One table per grouping, overall first, then in --group_by order.
    order = ["overall"] + list(args.group_by)
    columns = ["model", "group", "n", "corpus_wer", "mean_wer", "accuracy"]
    for field in order:
        field_rows = sorted(
            (r for r in rows if r["grouping"] == field),
            key=lambda r: (r["model"], -r["n"]),
        )
        if field_rows:
            title = "Overall" if field == "overall" else f"WER by {field}"
            print_table(field_rows, columns, title=title)

    # The disparity between the largest and smallest group, per grouping.
    gaps = []
    for field in args.group_by:
        by_model = defaultdict(list)
        for row in rows:
            if row["grouping"] == field:
                by_model[row["model"]].append(row)
        for model, model_rows in sorted(by_model.items()):
            if len(model_rows) < 2:
                continue
            best = min(model_rows, key=lambda r: r["corpus_wer"])
            worst = max(model_rows, key=lambda r: r["corpus_wer"])
            gaps.append(
                {
                    "model": model,
                    "grouping": field,
                    "best": f"{best['group']} ({best['corpus_wer']:.1f})",
                    "worst": f"{worst['group']} ({worst['corpus_wer']:.1f})",
                    "gap": worst["corpus_wer"] - best["corpus_wer"],
                }
            )

    if gaps:
        print_table(
            gaps,
            ["model", "grouping", "best", "worst", "gap"],
            title="Corpus WER gap between groups (lower is fairer)",
        )

    save_json(
        {"per_group": sorted(rows, key=lambda r: (r["model"], r["grouping"], r["group"])),
         "gaps": gaps},
        args.output_json,
    )


if __name__ == "__main__":
    main()
