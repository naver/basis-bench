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
from prompts import gender_personalization as P


def parse_choice(text):
    """Extract a single A/B choice, robust to common perturbations.

    Returns "A", "B", or None. None also covers the ambiguous case where both
    letters appear as candidate answers.
    """
    if not text:
        return None

    t = text.strip().upper()
    t = t.replace("*", "").replace("`", "").replace("#", "")

    # "ANSWER: A", "THE ANSWER IS B", "OPTION A"
    m = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*[\(\[]?\s*([AB])\b", t)
    if m:
        return m.group(1)

    m = re.search(r"[\(\[]\s*([AB])\s*[\)\]]", t)  # (A) / [B]
    if m:
        return m.group(1)

    m = re.match(r"^\s*([AB])\b\s*[\.\):-]", t)  # "A." / "B)" at the start
    if m:
        return m.group(1)

    # Standalone A / B tokens; reject if both appear.
    unique = set(re.findall(r"\b([AB])\b", t))
    if unique == {"A"}:
        return "A"
    if unique == {"B"}:
        return "B"

    return None


def run_inference(model, processor, data, args):
    """Run one condition over the whole dataset.

    Each (item, order) pair is its own datapoint. Per datapoint we record:
      - option_selected: "male" / "female" / None (unparsed)
      - correct:         did the model pick the dataset's correct_option
      - follows_signal:  explicit conditions only -- did the model pick the
                         option matching the injected target gender
    """
    condition = f"{args.modality}_{args.signal}"

    # ---- Build every conversation up front ----
    print(f"Preparing {len(data) * len(P.ORDERS)} inputs...")
    conversations = []
    meta = []  # (item, order, opt_a, opt_b)

    for item in data:
        audio_path = os.path.join(args.audio_root, item["audio_path"])
        for order in P.ORDERS:
            opt_a, opt_b = P.get_options(item, order)
            text = P.build_prompt(
                item["question"],
                opt_a,
                opt_b,
                modality=args.modality,
                signal=args.signal,
                targeted_user=args.targeted_user,
            )
            conversations.append(
                prepare_conversation(
                    P.system_prompt(args.modality),
                    text,
                    audio_path if args.modality == "speech" else None,
                )
            )
            meta.append((item, order, opt_a, opt_b))

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

    # ---- Record one result per (item, order) ----
    for (item, order, opt_a, opt_b), raw in zip(meta, all_texts):
        choice = parse_choice(raw)
        selected = opt_a if choice == "A" else (opt_b if choice == "B" else None)

        # In "male_first"   A = male,   B = female.
        # In "female_first" A = female, B = male.
        option_a_gender, option_b_gender = P.option_genders(order)

        if selected is None:
            option_selected = None
        elif selected == item["male_option"]:
            option_selected = "male"
        else:
            option_selected = "female"

        correct_gender = (
            "male" if item["correct_option"] == item["male_option"] else "female"
        )

        record = {
            "raw_output": raw,
            "choice": choice,
            "option_A": option_a_gender,
            "option_B": option_b_gender,
            "option_selected": option_selected,
            "correct": option_selected == correct_gender,
            "speaker_gender": correct_gender,
            "model": args.model,
            "condition": condition,
            "modality": args.modality,
            "signal": args.signal,
            "targeted_user": args.targeted_user if args.signal == "explicit" else None,
        }
        if args.signal == "explicit":
            record["follows_signal"] = option_selected == args.targeted_user

        item.setdefault("evaluation", {}).setdefault(args.model, {})[order] = record

    # ---- Summary: each (item, order) counted separately ----
    total = correct = unparsed = male_selected = female_selected = 0
    for item in data:
        for order in P.ORDERS:
            record = item["evaluation"][args.model][order]
            total += 1
            correct += int(record["correct"])
            if record["option_selected"] is None:
                unparsed += 1
            elif record["option_selected"] == "male":
                male_selected += 1
            else:
                female_selected += 1

    print(f"\n===== Summary [{condition}] [{args.model}] =====")
    print(f"Total datapoints (2x examples): {total}")
    print(f"Male option selected:           {male_selected}")
    print(f"Female option selected:         {female_selected}")
    print(f"Unparsed:                       {unparsed}")
    print(f"Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}")

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
            merged = old_item.get("evaluation", {})
            merged.update(new_item.get("evaluation", {}))
            new_item["evaluation"] = merged

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gender personalization generation (BASIS-Health / BASIS-Product). "
        "See generation/README.md for what this does and how to run it."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument(
        "--audio_root",
        required=True,
        help="audio_path values are relative to the split folder "
        "(data/BASIS_health_product/health or .../product).",
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
        "--targeted_user",
        choices=["male", "female"],
        required=True,
        help="The speaker's gender for this input file. Stated in the prompt "
        "in the explicit conditions, recorded either way.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--debug", type=int, default=None, help="Only process this many examples."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Condition: {args.modality}_{args.signal} | targeted_user={args.targeted_user}")

    data = load_data(args.input_json)
    if args.debug is not None:
        data = data[: args.debug]
        print(f"Debug mode: {args.debug} examples only")

    model, processor = load_model(args.model, args.model_path)
    data = run_inference(model, processor, data, args)
    save_data(data, args.output_folder, args.input_json)
