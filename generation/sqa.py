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
from prompts import sqa as P

# The 11 dialects SD-QA is released in.
DIALECTS = [
    "aus", "gbr", "ind_n", "ind_s", "irl", "kenya",
    "nga", "nzl", "phl", "usa", "zaf",
]


def run_inference(model, processor, data, args):
    label = f"sqa | {args.dataset}" + (f" | {args.dialect}" if args.dialect else "")

    print(f"Preparing {len(data)} inputs...")
    conversations = [
        prepare_conversation(
            args.system_prompt,
            P.build_prompt(args.dataset, args.text_prompt, item),
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
        desc=f"Running batched inference [{label}]",
    )

    for item, text in zip(data, all_texts):
        entry = {
            "task": "sqa",
            "output": text,
            "model": args.model,
            "condition": "sqa",
            "dataset": args.dataset,
        }
        # SD-QA is scored per dialect. LibriSQA is scored per speaker gender,
        # which evaluation/sqa.py looks up in the released id -> gender map.
        if args.dialect:
            entry["dialect"] = args.dialect
        item.setdefault("generations", {})[args.model] = entry

    print(f"\n===== Summary [{label}] [{args.model}] =====")
    print(f"Answers:       {len(all_texts)}")
    print(f"Empty answers: {sum(1 for t in all_texts if not t)}")
    print("Score with evaluation/sqa_judge.py, then evaluation/sqa.py")

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
        description="Spoken question answering generation (SD-QA, LibriSQA Part I). "
        "See generation/README.md for what this does and how to run it."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument(
        "--audio_root",
        required=True,
        help="Base the audio_path values are relative to: the SD-QA root "
        "(paths start with the dialect folder) or the LibriSQA root (paths "
        "start with audio/).",
    )
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--model",
        default="audio_flamingo_3",
        choices=["audio_flamingo_3", "audio_flamingo_next"],
    )
    parser.add_argument("--model_path", default=None, help="Override the checkpoint.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(P.DATASETS),
        help="Which dataset this input file is: sdqa (spoken question, text "
        "passage) or librisqa (spoken passage, text question).",
    )
    parser.add_argument(
        "--dialect",
        choices=DIALECTS,
        default=None,
        help="The dialect of this input file. Required for --dataset sdqa, "
        "not used for librisqa.",
    )

    parser.add_argument(
        "--text_prompt",
        default=None,
        help="Instruction sent in the user turn. Defaults to the dataset's "
        "own prompt from prompts/sqa.py.",
    )
    parser.add_argument("--system_prompt", default=P.SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--debug", type=int, default=None, help="Only process this many examples."
    )

    args = parser.parse_args()
    if args.dataset == "sdqa" and not args.dialect:
        parser.error("--dialect is required with --dataset sdqa")
    if args.dataset == "librisqa" and args.dialect:
        parser.error("--dialect does not apply to --dataset librisqa")
    if args.text_prompt is None:
        args.text_prompt = P.DATASETS[args.dataset]["text_prompt"]
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Condition: sqa | dataset={args.dataset}"
          + (f" | dialect={args.dialect}" if args.dialect else ""))

    data = load_data(args.input_json)
    if args.debug is not None:
        data = data[: args.debug]
        print(f"Debug mode: {args.debug} examples only")

    model, processor = load_model(args.model, args.model_path)
    data = run_inference(model, processor, data, args)
    save_data(data, args.output_folder, args.input_json)
