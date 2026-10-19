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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.models import generate, load_model, prepare_conversation
from prompts import age_personalization as P


def normalize_question(question):
    """Drop bracketed stage directions and collapse whitespace.

    "(laughs) Hello  there" -> "Hello there"
    """
    question = re.sub(r"\([^)]*\)", "", question or "")
    return re.sub(r"\s+", " ", question).strip()


def run_inference(model, processor, data, args):
    condition = "text_baseline" if args.baseline else f"{args.modality}_{args.signal}"
    system_prompt = P.system_prompt(args.modality, baseline=args.baseline)

    print(f"Preparing {len(data)} inputs...")
    conversations = []

    for item in data:
        audio_path = os.path.join(args.audio_root, item["audio_path"])
        text = P.build_prompt(
            normalize_question(item["question"]),
            modality=args.modality,
            signal=args.signal,
            targeted_user=args.targeted_user,
        )
        conversations.append(
            prepare_conversation(
                system_prompt,
                text,
                audio_path if args.modality == "speech" else None,
            )
        )

    print(f"\nPrompt example: {conversations[0]}\n")

    all_texts = generate(
        model,
        processor,
        conversations,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        desc=f"Running batched inference [{condition}]",
    )

    for item, text in zip(data, all_texts):
        item.setdefault("generations", {})[args.model] = {
            "output": text,
            "model": args.model,
            "condition": condition,
            "modality": args.modality,
            "signal": args.signal,
            "speaker_group": args.targeted_user,
            "targeted_user": args.targeted_user if args.signal == "explicit" else None,
        }

    lengths = [len(t.split()) for t in all_texts]
    print(f"\n===== Summary [{condition}] [{args.model}] =====")
    print(f"Generations:         {len(all_texts)}")
    print(f"Empty generations:   {sum(1 for t in all_texts if not t)}")
    print(f"Mean length (words): {sum(lengths) / len(lengths):.1f}")

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
        description="Age personalization generation (ELIP-Pair). "
        "See generation/README.md for what this does and how to run it."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument(
        "--audio_root",
        required=True,
        help="audio_path values are relative to data/ELIP_Pair/audio.",
    )
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--model",
        default="audio_flamingo_3",
        choices=["audio_flamingo_3", "audio_flamingo_next"],
    )
    parser.add_argument("--model_path", default=None, help="Override the checkpoint.")

    parser.add_argument("--modality", choices=["speech", "text"], required=True)
    parser.add_argument("--signal", choices=["implicit", "explicit"], required=True)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="text-baseline condition: send the default-behaviour system "
        "prompt, which specifies no response style.",
    )
    parser.add_argument(
        "--targeted_user",
        choices=["child", "adult"],
        required=True,
        help="The speaker's age group for this input file. Stated in the "
        "prompt in the explicit conditions, recorded either way.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--debug", type=int, default=None, help="Only process this many examples."
    )

    args = parser.parse_args()
    if args.baseline and (args.modality != "text" or args.signal != "implicit"):
        parser.error("--baseline requires --modality text --signal implicit")
    return args


if __name__ == "__main__":
    args = parse_args()
    condition = "text_baseline" if args.baseline else f"{args.modality}_{args.signal}"
    print(f"Condition: {condition} | speaker_group={args.targeted_user}")

    data = load_data(args.input_json)
    if args.debug is not None:
        data = data[: args.debug]
        print(f"Debug mode: {args.debug} examples only")

    model, processor = load_model(args.model, args.model_path)
    data = run_inference(model, processor, data, args)
    save_data(data, args.output_folder, args.input_json)
