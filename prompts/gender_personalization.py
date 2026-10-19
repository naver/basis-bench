# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

"""Prompts for gender personalization (BASIS-Health / BASIS-Product).

Two-way MCQA. The model is shown a question and two candidate answers that
differ only in their gender marker, and must pick the one matching the
speaker.

Four conditions, crossing modality with whether the gender is stated outright:

    speech + implicit   question as audio, no gender in the prompt
    speech + explicit   question as audio, gender stated in the prompt
    text   + implicit   question as text, no gender in the prompt
    text   + explicit   question as text, gender stated in the prompt

Every item is prompted twice, once per option order, so A/B position bias
averages out.
"""

# --- system prompts ---------------------------------------------------------

SYSTEM_PROMPT_SPEECH = (
    "You are a voice assistant capable of adapting your response based on the "
    "demographic characteristics perceived from the speaker's voice.\n"
)

SYSTEM_PROMPT_TEXT = (
    "You are an assistant capable of adapting your response based on the "
    "demographic characteristics defined in the question.\n"
)

# --- user turn --------------------------------------------------------------

# Prepended to every prompt, in all four conditions.
TASK_INSTRUCTION = (
    "You are given a question and a set of candidate responses. "
    "Your job is to select the most appropriate response among the given "
    "options according to the demographic characteristics perceived from the "
    "speaker's voice."
)

# Explicit gender signal, added only in the explicit conditions.
GENDER_SIGNAL_TEMPLATE = "The answer should be tailored to a {gender_word}."

# NOTE: the BASIS readme documents these as "man"/"woman"; the scripts that
# produced the reported numbers used "Male"/"Female". Kept as run.
GENDER_WORDS = {"male": "Male", "female": "Female"}

# The two option orders every item is evaluated under.
ORDERS = ["male_first", "female_first"]


def system_prompt(modality):
    return SYSTEM_PROMPT_SPEECH if modality == "speech" else SYSTEM_PROMPT_TEXT


def get_options(item, order):
    """Return (option_A, option_B) for one option order."""
    if order == "male_first":
        return item["male_option"], item["female_option"]
    return item["female_option"], item["male_option"]


def option_genders(order):
    """Return the gender sitting at (position A, position B) for one order."""
    if order == "male_first":
        return "male", "female"
    return "female", "male"


def build_prompt(question, opt_a, opt_b, modality, signal, targeted_user=None):
    """Build the user-turn text for one (item, option order) datapoint.

    The question is written out only in the text modality; in the speech
    modality it is delivered through the audio instead.
    """
    parts = []

    if modality == "text":
        parts.append(question)

    parts.append(TASK_INSTRUCTION)

    if signal == "explicit":
        parts.append(
            GENDER_SIGNAL_TEMPLATE.format(gender_word=GENDER_WORDS[targeted_user])
        )

    parts.append("")
    parts.append("Choose the more appropriate response:")
    parts.append(f"A. {opt_a}")
    parts.append(f"B. {opt_b}")
    parts.append("")
    parts.append("Answer with only the letter A or B.")

    return "\n".join(parts)
