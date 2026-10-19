#!/usr/bin/env python
# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

"""Build the Fair-Speech ASR input JSON from the released TSV + audio folder.

Fair-Speech ships as two pieces -- a folder of wavs and a metadata TSV keyed
by the wav's hash name -- which this joins into the item format the rest of
the pipeline expects:

    {
      "audio_path": "audios/asr_fairness_audio/<hash>.wav",
      "gender": "female",
      "age": "18 - 22",
      "first_language": "English",
      "ethnicity": "White",
      "transcript": "hey facebook answer the call"
    }

`audio_path` is relative to the Fair-Speech root, so generation is run with
`--audio_root data/FairSpeech`, the same way Spoken StereoSet+ works.

Example
-------
    python prepare_fairspeech.py --fairspeech_root data/FairSpeech
"""

import argparse
import csv
import json
import os
from collections import Counter

# Columns copied from the TSV into each item. `transcription` is renamed to
# `transcript`, which is what evaluation/asr_fairspeech.py scores against.
# The TSV also carries `socioeconomic_bkgd`, which the paper does not use.
METADATA_COLUMNS = ["gender", "age", "first_language", "ethnicity"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fairspeech_root",
        required=True,
        help="Folder holding metadata.tsv and audios/.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.tsv (default: <fairspeech_root>/metadata.tsv).",
    )
    parser.add_argument(
        "--audio_subdir",
        default="audios/asr_fairness_audio",
        help="Wav folder, relative to --fairspeech_root "
        "(default: audios/asr_fairness_audio).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Where to write the JSON "
        "(default: <fairspeech_root>/FairSpeech_ASR_data.json).",
    )
    parser.add_argument(
        "--keep_missing_audio",
        action="store_true",
        help="Keep rows whose wav is absent. They are dropped by default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = args.fairspeech_root
    metadata = args.metadata or os.path.join(root, "metadata.tsv")
    output_json = args.output_json or os.path.join(root, "FairSpeech_ASR_data.json")
    audio_dir = os.path.join(root, args.audio_subdir)

    for path, what in ((metadata, "metadata TSV"), (audio_dir, "audio folder")):
        if not os.path.exists(path):
            raise SystemExit(
                f"{what} not found: {path}\n"
                "See the Fair-Speech section of the README for how to obtain it."
            )

    with open(metadata, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"Read {len(rows)} rows from {metadata}")

    missing_columns = [
        c for c in METADATA_COLUMNS + ["hash_name", "transcription"]
        if c not in rows[0]
    ]
    if missing_columns:
        raise SystemExit(
            f"metadata TSV is missing columns {missing_columns}; "
            f"found {list(rows[0])}"
        )

    data = []
    missing_audio = 0
    empty_transcript = 0
    for row in rows:
        relative = os.path.join(args.audio_subdir, f"{row['hash_name']}.wav")
        if not os.path.isfile(os.path.join(root, relative)):
            missing_audio += 1
            if not args.keep_missing_audio:
                continue

        # A blank reference cannot be scored, so those rows are dropped.
        if not (row["transcription"] or "").strip():
            empty_transcript += 1
            continue

        item = {"audio_path": relative}
        item.update({column: row[column] for column in METADATA_COLUMNS})
        item["transcript"] = row["transcription"]
        data.append(item)

    if missing_audio:
        kept = "kept" if args.keep_missing_audio else "dropped"
        print(f"[warn] {missing_audio} rows have no wav in {audio_dir} ({kept})")
    if empty_transcript:
        print(f"[warn] dropped {empty_transcript} rows with a blank transcript")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nWrote {len(data)} items to {output_json}")
    for column in METADATA_COLUMNS:
        counts = Counter(item[column] for item in data)
        shown = ", ".join(f"{k}={v}" for k, v in counts.most_common(4))
        extra = f", +{len(counts) - 4} more" if len(counts) > 4 else ""
        print(f"  {column:<15} {len(counts):>3} values  ({shown}{extra})")


if __name__ == "__main__":
    main()
