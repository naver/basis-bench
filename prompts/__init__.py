# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

"""All prompts used by BASIS-Bench, one module per task.

    gender_personalization   BASIS-Health / BASIS-Product
    age_personalization      ELIP-Pair
    stereotypical_bias       Spoken StereoSet+
    asr                      Spoken StereoSet+ ASR age split

Each module holds the system prompts, the user-turn templates for every
evaluation condition, and a `build_prompt()` that assembles them. The strings
are transcribed verbatim from the scripts that produced the reported numbers,
so trailing spaces and newlines inside some of them are intentional.
"""
