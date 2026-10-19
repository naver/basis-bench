'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.models import generate, load_model, prepare_conversation
# The 11 SD-QA dialects, so --dialect is validated against the same list.
from generation.sqa import DIALECTS
from prompts import asr as P


def run_inference(model, processor, data, args):
    print(f"Preparing {len(data)} inputs...")
    conversations = [
        prepare_conversation(
            args.system_prompt,
            args.text_prompt,
            os.path.join(args.audio_root, item["audio_path"]),
        )
        for item in data
    ]

    print(f"\nPrompt example: {conversations[0]}\n")

    all_texts = generate(
        model,
        processor,
        conversations,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        desc="Running batched inference [asr]",
    )

    for item, text in zip(data, all_texts):
        entry = {
            "task": "asr",
            "output": text,
            "model": args.model,
            "condition": "asr",
        }
        # Whichever group label this dataset is scored by. Fair-Speech needs
        # neither: it is grouped by the demographic fields on each item.
        if args.age_group:
            entry["age_group"] = args.age_group
        if args.dialect:
            entry["dialect"] = args.dialect
        item.setdefault("generations", {})[args.model] = entry

    print(f"\n===== Summary [asr] [{args.model}] =====")
    print(f"Transcripts:       {len(all_texts)}")
    print(f"Empty transcripts: {sum(1 for t in all_texts if not t)}")

    return data


def load_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def save_data(data, output_folder, input_json_path):
    """Write the results, merging into an existing file by item index."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(input_json_path))

    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for old_item, new_item in zip(existing, data):
            merged = old_item.get("generations", {})
            merged.update(new_item.get("generations", {}))
            new_item["generations"] = merged

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR generation (Spoken StereoSet+ ASR split, Fair-Speech, SD-QA). "
        "See generation/README.md for what this does and how to run it."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument(
        "--audio_root",
        required=True,
        help="Base the audio_path values are relative to: the repository root "
        "for Spoken StereoSet+ (whose paths already include audio/), the "
        "Fair-Speech root for Fair-Speech.",
    )
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--model",
        default="audio_flamingo_3",
        choices=["audio_flamingo_3", "audio_flamingo_next"],
    )
    parser.add_argument("--model_path", default=None, help="Override the checkpoint.")
    parser.add_argument(
        "--age_group",
        choices=["child", "young", "elderly"],
        default=None,
        help="The age group of this input file, recorded with the outputs "
        "and used to group the Spoken StereoSet+ scores. Omit for Fair-Speech, "
        "which is grouped by the demographic fields on each item instead.",
    )

    parser.add_argument(
        "--dialect",
        choices=DIALECTS,
        default=None,
        help="The SD-QA dialect of this input file, recorded with the outputs "
        "and used to group the SD-QA scores.",
    )

    parser.add_argument("--text_prompt", default=P.TEXT_PROMPT)
    parser.add_argument("--system_prompt", default=P.SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--debug", type=int, default=None, help="Only process this many examples."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Condition: asr | age_group={args.age_group}")

    data = load_data(args.input_json)
    if args.debug is not None:
        data = data[: args.debug]
        print(f"Debug mode: {args.debug} examples only")

    model, processor = load_model(args.model, args.model_path)
    data = run_inference(model, processor, data, args)
    save_data(data, args.output_folder, args.input_json)
