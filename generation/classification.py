'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.models import generate, load_model, prepare_conversation
from prompts.classification import TASK_CONFIG, TASKS


def normalize_prediction(raw_output, classes):
    """Map raw model text onto one of the valid class labels.

    Returns the matched class label, or 'unknown' if nothing matches.
    """
    text = raw_output.strip().lower()

    # 1) Exact match wins.
    for cls in classes:
        if text == cls.lower():
            return cls

    # 2) Whole-word match. Check longer labels first so that e.g. "female"
    #    is not shadowed by "male" (which is a substring of "female").
    for cls in sorted(classes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(cls.lower())}\b", text):
            return cls

    return "unknown"


def run_inference(model, processor, data, args):
    cfg = TASK_CONFIG[args.task]
    classes = cfg["classes"]

    print(f"Preparing {len(data)} inputs...")
    conversations = [
        prepare_conversation(
            cfg["system_prompt"],
            cfg["task_prompt"],
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
        desc=f"Running batched inference [{args.task}]",
    )

    for item, raw in zip(data, all_texts):
        # Different tasks can be stored side by side for the same model.
        entry = item.setdefault("generations", {}).setdefault(args.model, {})
        entry[args.task] = {
            "task": args.task,
            "prediction": normalize_prediction(raw, classes),
            "raw_output": raw,
            "gold": args.gold,
            "model": args.model,
        }

    counter = Counter(
        item["generations"][args.model][args.task]["prediction"] for item in data
    )
    total = len(data)
    print("\n" + "=" * 50)
    print(f"Classification stats [{args.task}] (total: {total})")
    print("=" * 50)
    for cls in list(classes) + [k for k in counter if k not in classes]:
        count = counter.get(cls, 0)
        pct = (100.0 * count / total) if total else 0.0
        print(f"  {cls:<12}: {count:>6}  ({pct:5.1f}%)")
    print("=" * 50)

    return data


def load_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def save_data(data, output_folder, input_json_path):
    """Write the results, merging into an existing file by item index.

    Merged per model as well as per item, so that running a second task
    (or a second model) into the same file accumulates rather than replaces.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(input_json_path))

    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for old_item, new_item in zip(existing, data):
            merged = old_item.get("generations", {})
            for model_key, model_gen in new_item.get("generations", {}).items():
                merged.setdefault(model_key, {})
                merged[model_key].update(model_gen)
            new_item["generations"] = merged

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demographic classification generation. "
        "See generation/README.md for what this does and how to run it."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument(
        "--audio_root",
        required=True,
        help="Base the audio_path values are relative to: the split folder "
        "for BASIS, audio/ for ELIP-Pair, the repository root for "
        "Spoken StereoSet+.",
    )
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--model",
        default="audio_flamingo_3",
        choices=["audio_flamingo_3", "audio_flamingo_next"],
    )
    parser.add_argument("--model_path", default=None, help="Override the checkpoint.")

    parser.add_argument(
        "--task",
        required=True,
        choices=TASKS,
        help="Which classification to run. The age variants differ only in "
        "their classes: young/elderly for Spoken StereoSet+, child/adult "
        "for ELIP-Pair and BASIS.",
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="The speaker group of this input file (e.g. 'female'), recorded "
        "with every prediction. Must be one of the task's classes, so the "
        "input file has to hold a single group.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--debug", type=int, default=None, help="Only process this many examples."
    )

    args = parser.parse_args()
    classes = TASK_CONFIG[args.task]["classes"]
    if args.gold not in classes:
        parser.error(
            f"--gold must be one of {classes} for task {args.task!r}, "
            f"got {args.gold!r}"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Task: {args.task} -> classes: {TASK_CONFIG[args.task]['classes']}")

    data = load_data(args.input_json)
    if args.debug is not None:
        data = data[: args.debug]
        print(f"Debug mode: {args.debug} examples only")

    model, processor = load_model(args.model, args.model_path)
    data = run_inference(model, processor, data, args)
    save_data(data, args.output_folder, args.input_json)
