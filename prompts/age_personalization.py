# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

"""Prompts for age personalization (ELIP-Pair).

Open-ended generation. The model answers a spoken question and should adopt
the register matching the speaker: playful and non-expert for a child, formal
and expert for an adult.

Five conditions:

    speech + implicit   question as audio, no age in the prompt
    speech + explicit   question as audio, age stated in the prompt
    text   + implicit   question as text, no age in the prompt; the system
                        prompt still describes both styles, so this measures
                        which profile the model prefers by default
    text   + explicit   question as text, age stated in the prompt
    text-baseline       question as text, and no response style specified at
                        all -- the model's default register

text-implicit and text-baseline send the same user turn; they differ only in
the system prompt.
"""

# --- personas ---------------------------------------------------------------
#
# Not injected into the generation prompts (the system prompt already
# describes both styles). These are the canonical wordings, reused by the
# LLM-as-judge in evaluation/age_personalization_judge.py.

CHILD_PROFILE = (
    "- Expertise : Prefers a response that can be easily understood by a child.\n"
    "- Style : Prefers a response that is friendly, witty, funny, and humorous\n"
)

ADULT_PROFILE = (
    "- Expertise : Prefers a response that is tailored for an expert audience.\n"
    "- Style : Prefers a response that answers in a cold and impersonal tone.\n"
)

# How each persona is named inside the explicit prompts.
USER_PROFILES = {"child": "a child", "adult": "an adult person"}

# --- system prompts ---------------------------------------------------------
#
# The trailing spaces after "styles: " and "paragraph. " are verbatim from the
# run scripts.

SYSTEM_PROMPT_SPEECH = (
    "You are a voice assistant capable of adapting your response based on the "
    "demographic characteristics perceived from the speaker voice. "
    "Depending on the speaker, you should adopt one or two interaction styles: \n"
    "1) Adult speaker: respond in formal style, assuming an expert audience.\n"
    "2) Child speaker: respond in a friendly, witty, funny and humorous tone, "
    "assuming a non-expert audience.\n"
    "In both cases, your response should range from a single sentence to a "
    "very short paragraph. \n"
)

SYSTEM_PROMPT_TEXT = (
    "You are an assistant capable of adapting your response based on the "
    "demographic characteristics defined in the question. "
    "Depending on the user, you should adopt one or two interaction styles: \n"
    "1) Adult speaker: respond in formal style, assuming an expert audience.\n"
    "2) Child speaker: respond in a friendly, witty, funny and humorous tone, "
    "assuming a non-expert audience.\n"
    "In both cases, your response should range from a single sentence to a "
    "very short paragraph. \n"
)

# text-baseline: no response style is specified.
SYSTEM_PROMPT_BASELINE = (
    "Please answer the given question. Your response should range from a "
    "single sentence to a very short paragraph."
)

# --- user turn --------------------------------------------------------------

SPEECH_IMPLICIT_TEMPLATE = (
    "Answer the question considering the demographic characteristics perceived "
    "from the speaker voice. "
    "Do not include any introduction, preamble, explanation or conclusion — "
    "only the direct response to the question.\n"
)

SPEECH_EXPLICIT_TEMPLATE = (
    "Answer the question considering the demographic characteristics perceived "
    "from the speaker voice. "
    "Do not include any introduction, preamble, explanation or conclusion — "
    "only the direct response to the question.\n"
    "The speaker is {user}.\n"
)

TEXT_IMPLICIT_TEMPLATE = (
    "You are given a question. "
    "Your job is to write a response to the question. Your response should "
    "range from a single sentence to a very short paragraph. "
    "Do not include any introduction, preamble, explanation or conclusion — "
    "only the direct response to the question.\n"
    "Here is the question:\n"
    "{question}\n"
)

TEXT_EXPLICIT_TEMPLATE = (
    "You are given a question. Your job is to write a response using the tone "
    "and style that matches the preferences of the user. "
    "Your response should range from a single sentence to a very short paragraph. "
    "Do not include any introduction, preamble, explanation or conclusion — "
    "only the direct response to the question.\n"
    "The speaker is {user}.\n"
    "Here is the question:\n"
    "{question}\n"
)


def system_prompt(modality, baseline=False):
    if baseline:
        return SYSTEM_PROMPT_BASELINE
    return SYSTEM_PROMPT_SPEECH if modality == "speech" else SYSTEM_PROMPT_TEXT


def build_prompt(question, modality, signal, targeted_user=None):
    """Build the user-turn text for one ELIP-Pair datapoint.

    In the speech modality the question is never written out: it only lives in
    the audio.
    """
    if modality == "text":
        if signal == "explicit":
            return TEXT_EXPLICIT_TEMPLATE.format(
                user=USER_PROFILES[targeted_user], question=question
            )
        return TEXT_IMPLICIT_TEMPLATE.format(question=question)

    if signal == "explicit":
        return SPEECH_EXPLICIT_TEMPLATE.format(user=USER_PROFILES[targeted_user])
    return SPEECH_IMPLICIT_TEMPLATE
