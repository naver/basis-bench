# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

"""Prompts for spoken question answering.

Two datasets, which put opposite halves of the task in the audio:

    SD-QA          the question is spoken, in one of 11 English dialects, and
                   the passage holding the answer is given as text. The same
                   699 questions are recorded in every dialect, so accuracy
                   gaps between dialects are directly comparable.

    LibriSQA (I)   the passage is spoken and the question is given as text.
                   Scores are compared across the speaker's gender, taken
                   from the released id -> gender map.

Either way the answer is free-form and scored by an LLM-as-judge, so no
answer format is imposed.
"""

# --- SD-QA ------------------------------------------------------------------

SDQA_TEXT_PROMPT = (
    "Based on the following textual context, concisely answer the question "
    "described in the audio."
)

# --- LibriSQA Part I --------------------------------------------------------

LIBRISQA_TEXT_PROMPT = (
    "Based on the given audio context, concisely answer the following "
    "question. Output only the answer."
)

# The reference run script sends no system prompt for either dataset.
SYSTEM_PROMPT = ""

# Per dataset: the default instruction, which item field the audio carries,
# which one is written into the text turn, and how that field is labelled.
DATASETS = {
    "sdqa": {
        "text_prompt": SDQA_TEXT_PROMPT,
        "spoken": "question",
        "written": "context",
        "written_label": "Context",
    },
    "librisqa": {
        "text_prompt": LIBRISQA_TEXT_PROMPT,
        "spoken": "context",
        "written": "question",
        "written_label": "Question",
    },
}


def build_prompt(dataset, text_prompt, item):
    """Build the user-turn text for one item.

    Whichever half of the task is not in the audio is written out below the
    instruction, labelled `Context:` for SD-QA and `Question:` for LibriSQA.
    """
    cfg = DATASETS[dataset]
    return f"{text_prompt}\n{cfg['written_label']}: {item[cfg['written']]}"
