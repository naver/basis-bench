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
from prompts import stereotypical_bias as P

# Speaker -> demographic group, keyed by the speaker name in the filename
# (files are named "<context id>_<speaker>.wav").
GENDER_MAP = {
    "en-US-AvaNeural": "female",
    "en-US-EmmaMultilingualNeural": "female",
    "en-US-SaraNeural": "female",
    "en-US-BrianMultilingualNeural": "male",
    "en-US-AndrewMultilingualNeural": "male",
    "en-US-ChristopherNeural": "male",
}

AGE_MAP = {
    "old_man": "elderly",
    "John_Madden": "elderly",
    "old_women": "elderly",
    "Hillary_Clinton": "elderly",
    "Matthew": "young",
    "Mason-HQ": "young",
    "Barbara": "young",
    "Megan-HQ": "young",
    "girl": "child",
    "Ana": "child",
    "boy": "child",
}


def speaker_group(audio_path, demographic):
    """Read the speaker's demographic group off the audio filename."""
    speaker = os.path.splitext(os.path.basename(audio_path))[0].split("_", 1)[1]
    mapping = GENDER_MAP if demographic == "gender" else AGE_MAP
    if speaker not in mapping:
        raise ValueError(
            f"unknown speaker {speaker!r} in {audio_path}; "
            f"known speakers: {sorted(mapping)}"
        )
    return speaker, mapping[speaker]


def run_inference(model, processor, data, args):
    condition = f"{args.modality}_{args.signal}"

    print(f"Preparing {len(data) * len(P.ALL_PERMUTATIONS)} inputs...")
    conversations = []
    meta = []  # (item, perm_index, letter_map, group)

    for item in data:
        audio_path = os.path.join(args.audio_root, item["audio_path"])
        speaker, group = speaker_group(item["audio_path"], args.demographic)

        for perm_index, perm in enumerate(P.ALL_PERMUTATIONS):
            prompt_text, letter_map = P.build_prompt(
                item,
                perm,
                modality=args.modality,
                signal=args.signal,
                speaker_group=group,
            )
            conversations.append(
                prepare_conversation(
                    P.system_prompt(args.modality),
                    P.build_user_turn(prompt_text, args.modality, item["context"]),
                    audio_path if args.modality == "speech" else None,
                )
            )
            meta.append((item, perm_index, letter_map, speaker, group))

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

    for (item, perm_index, letter_map, speaker, group), raw in zip(meta, all_texts):
        entry = item.setdefault("generations", {}).setdefault(args.model, {})
        # Run metadata sits beside the perm_* keys; scoring skips any key that
        # does not start with "perm_".
        entry.update(
            {
                "model": args.model,
                "condition": condition,
                "modality": args.modality,
                "signal": args.signal,
                "demographic": args.demographic,
                "speaker": speaker,
                "speaker_group": group,
            }
        )
        entry[f"perm_{perm_index}"] = {
            "A": letter_map["A"],
            "B": letter_map["B"],
            "C": letter_map["C"],
            "response": raw,
        }

    print(f"\n===== Summary [{condition}] [{args.model}] =====")
    print(f"Items:        {len(data)}")
    print(f"Permutations: {len(all_texts)}")
    print("Score with evaluation/stereotypical_bias.py")

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
        description="Stereotypical bias generation (Spoken StereoSet+). "
        "See generation/README.md for what this does and how to run it."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument(
        "--audio_root",
        required=True,
        help="audio_path values are relative to the repository root "
        "(data/SpokenSS_plus); they already include the audio/ prefix.",
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
        "--demographic",
        choices=["gender", "age"],
        required=True,
        help="Which subset this input file is, i.e. which speaker map to use.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument(
        "--debug", type=int, default=None, help="Only process this many examples."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Condition: {args.modality}_{args.signal} | demographic={args.demographic}")

    data = load_data(args.input_json)
    if args.debug is not None:
        data = data[: args.debug]
        print(f"Debug mode: {args.debug} examples only")

    model, processor = load_model(args.model, args.model_path)
    data = run_inference(model, processor, data, args)
    save_data(data, args.output_folder, args.input_json)
