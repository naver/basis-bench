'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

from tqdm import tqdm

# torch and transformers are imported inside the functions below, so that the
# prompts can be inspected on a machine without them installed.

MODEL_PATHS = {
    "audio_flamingo_3": "nvidia/audio-flamingo-3-hf",
    "audio_flamingo_next": "nvidia/audio-flamingo-next-hf",
}

SAMPLING_PARAMETERS = {
    "do_sample": False,
    "top_p": 1.0,
    "temperature": 0.0,
}


def load_model(model_name, model_path=None):
    import torch

    model_path = model_path or MODEL_PATHS[model_name]
    print(f"Loading model: {model_path}")

    if model_name == "audio_flamingo_3":
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )

    elif model_name == "audio_flamingo_next":
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    else:
        raise ValueError(
            f"unknown model {model_name!r}; expected one of {list(MODEL_PATHS)}"
        )

    return model, processor


def prepare_conversation(system_prompt, text, audio_path=None):
    """Build one conversation in the HF chat-template format.

    An audio part is attached only in the speech conditions; the text
    conditions send no audio at all.
    """
    conversation = []

    if system_prompt:
        conversation.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )

    content = [{"type": "text", "text": text}]
    if audio_path:
        content.append({"type": "audio", "path": audio_path})

    conversation.append({"role": "user", "content": content})
    return conversation


def generate(
    model,
    processor,
    conversations,
    model_name,
    max_new_tokens,
    batch_size,
    desc="Running batched inference",
):
    """Generate for every conversation, in batches, preserving input order."""
    import torch

    sampling_parameters = dict(SAMPLING_PARAMETERS, max_new_tokens=max_new_tokens)
    all_texts = []

    for i in tqdm(range(0, len(conversations), batch_size), desc=desc):
        batch = conversations[i : i + batch_size]

        if model_name == "audio_flamingo_3":
            inputs = processor.apply_chat_template(
                batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Cast ONLY the floating-point tensors (the audio features) to the
            # model dtype; input_ids / attention_mask must stay integral.
            inputs = {
                k: v.to(
                    device=model.device,
                    dtype=model.dtype if v.is_floating_point() else v.dtype,
                )
                for k, v in inputs.items()
            }
            decode_kwargs = {"skip_special_tokens": True}

        else:  # audio_flamingo_next
            inputs = processor.apply_chat_template(
                batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)
            decode_kwargs = {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }

        with torch.inference_mode():
            outputs = model.generate(**inputs, **sampling_parameters)

        prompt_length = inputs["input_ids"].shape[1]
        all_texts.extend(
            processor.batch_decode(outputs[:, prompt_length:], **decode_kwargs)
        )

    return [text.strip() for text in all_texts]
