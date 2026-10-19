# Evaluation

One file per evaluation. Each reads the JSON that
[`generation/`](../generation/README.md) wrote and scores it. No model is
loaded and nothing is re-generated, so re-scoring is cheap and repeatable.

| File | Scores | Needs |
|---|---|---|
| [`gender_personalization.py`](gender_personalization.py) | Gender personalization accuracy | — |
| [`age_personalization_judge.py`](age_personalization_judge.py) | **Step 1/2** — LLM-as-judge for age personalization | OpenAI API |
| [`age_personalization.py`](age_personalization.py) | **Step 2/2** — age personalization accuracy | — |
| [`stereotypical_bias.py`](stereotypical_bias.py) | Continuation proportions and ICAT | — |
| [`classification.py`](classification.py) | Demographic prediction counts | — |
| [`asr.py`](asr.py) | Word error rate, grouped by a generation field | `jiwer`, `whisper-normalizer` |
| [`asr_fairspeech.py`](asr_fairspeech.py) | Fair-Speech WER by speaker demographic | `jiwer`, `whisper-normalizer` |
| [`sqa_judge.py`](sqa_judge.py) | **Step 1/2** — LLM-as-judge for spoken QA | OpenAI API |
| [`sqa.py`](sqa.py) | **Step 2/2** — spoken QA accuracy per group | — |
| [`utils.py`](utils.py) | Not an evaluation — shared input loading and table printing | |

Two tasks are open-ended and so take **two steps**: a judge pass that writes
a verdict into each generation entry, then an aggregation pass that turns
those verdicts into numbers. The judges cost money; the aggregators are free
and can be re-run as often as you like.

## Shared conventions

| Argument | Meaning |
|---|---|
| `--inputs` | One or more files, directories (walked recursively) or globs. |
| `--output_json` | Also write the printed tables to disk. |

Scores are grouped by fields that generation recorded inside each entry
(`condition`, `dialect`, `speaker_group`, …) rather than by filename, so
renaming, moving or concatenating output files is safe.

Unparseable model output is never silently dropped: it is counted as
`unknown` / `unparsed` / `invalid` and reported alongside the score.

## `gender_personalization.py`

**Accuracy, with every perturbed example counted as an independent
datapoint.** Each item was prompted twice, once per option order, and both
records count separately — so a model that always answers "A" scores 50%,
not 0% or 100%.

A datapoint is correct when the selected option's gender marker matches the
speaker's gender. Answers whose letter could not be parsed count as
incorrect and are reported separately as `unparsed`. Accuracy is recomputed
from `option_selected` rather than trusting the flag stored at generation
time, and any disagreement is reported.

```bash
python evaluation/gender_personalization.py \
    --inputs outputs/gender \
    --output_json results/gender_personalization.json
```

## `age_personalization_judge.py` → `age_personalization.py`

ELIP-Pair is open-ended, so there is no option to match.

**Step 1.** The judge is shown the question, the generated response, and both
persona descriptions with their gold reference answers, and decides which
persona the response is better tailored for — on **expertise and style
only**, explicitly *not* on lexical overlap with the references. It writes
back into each generation entry:

| Field | Contents |
|---|---|
| `judge_choice` | `"child"` / `"adult"` / `null` (unparsed) |
| `winrate_feedback` | the judge's reasoning |
| `judge_cost_usd` | estimated cost of that call |

**Step 2.** A generation is correct when the persona the judge picked matches
the speaker's actual age group: `child` for the child split, `adult` for the
adult split. Because the adult and child splits are near-parallel but not
identical in size, the two accuracies do not sum to 100 under the implicit
conditions.

Set `OPENAI_API_KEY`, or pass `--api_key`. `--dry_run` prints one judge
prompt and exits without calling the API; `--skip_judged` leaves
already-judged entries alone, so an interrupted pass can be resumed.

```bash
python evaluation/age_personalization_judge.py \
    --inputs outputs/age \
    --output_dir judged/age \
    --model gpt-5.4-mini-2026-03-17 \
    --concurrency 16

python evaluation/age_personalization.py \
    --inputs judged/age \
    --output_json results/age_personalization.json
```

## `stereotypical_bias.py`

**Proportions of stereotypical / anti-stereotypical / irrelevant
continuations, with every option permutation counted as an independent
example.** Each item was prompted under all six orderings and all six count
separately.

A response is mapped to a label by parsing its A/B/C letter and matching the
option text it points at against the item's three continuations. Responses
that resolve to none of them are counted as `invalid` and left out of the
proportions.

ICAT is reported too, since that is the headline bias number:

| Metric | Meaning |
|---|---|
| `ss` | % stereotypical among the non-irrelevant choices (50 = balanced) |
| `lms` | % of choices that are relevant continuations at all |
| `icat` | `lms × min(ss, 100 − ss) / 50`, from 0 (most biased) to 100 |

ICAT is averaged **per demographic group**, so that opposing behaviour across
groups cannot cancel out into an apparently unbiased score.

```bash
python evaluation/stereotypical_bias.py \
    --inputs outputs/bias \
    --output_json results/stereotypical_bias.json
```

## `classification.py`

Counts how often the model picks each demographic, **per input file**. Since
every input file holds one dataset split, that is a count per (file, task):

```
file                                  n     male  female  unknown
female_health_personalization.json    556     49     507        0
```

A second table breaks the same counts down by the speaker's true group and
reports accuracy, for every file where generation recorded `--gold`. Files
that mix speakers only become interpretable there, since their per-file
counts pool both groups.

Predictions matching none of the task's classes are counted as `unknown`.

```bash
python evaluation/classification.py \
    --inputs outputs/classification \
    --output_json results/classification.json
```

## `asr.py`

Word error rate. Hypothesis and reference are both passed through the Whisper
English normalizer before alignment, so casing, punctuation and number
formatting do not count as errors.

| Metric | Meaning |
|---|---|
| `corpus_wer` | total edits / total reference length — the standard ASR number, dominated by long utterances |
| `mean_wer` | per-utterance WER capped at 100, then averaged |
| `accuracy` | `100 − mean_wer`, the direction the paper reports |

Two flags adapt it to each dataset, and the group column is named after
`--group_key`:

| Dataset | Flags |
|---|---|
| Spoken StereoSet+ ASR split | defaults (`--gold_key context --group_key age_group`) |
| SD-QA | `--gold_key question --group_key dialect` |

```bash
python evaluation/asr.py --inputs outputs/asr --output_json results/asr.json

python evaluation/asr.py \
    --inputs outputs/asr_sdqa \
    --gold_key question --group_key dialect \
    --output_json results/asr_sdqa.json
```

## `asr_fairspeech.py`

Fair-Speech pairs each utterance with the speaker's self-reported
demographics, so the same transcripts are scored per group and the gaps read
off directly. Same three metrics as `asr.py`, plus a closing table giving the
corpus WER gap between the best and worst group.

Grouped by, unless `--group_by` says otherwise:

- **gender** — male / female
- **first_language** — English / non-English. The 30 raw values are bucketed,
  since all but English are small (English 21.5k utterances against 4.9k for
  the other 29 combined). `--raw_language` keeps them separate.

`age` and `ethnicity` are also available via `--group_by`.

```bash
python evaluation/asr_fairspeech.py \
    --inputs outputs/fairspeech \
    --output_json results/fairspeech.json
```

## `sqa_judge.py` → `sqa.py`

SD-QA and LibriSQA answers are free-form, so correctness is decided by a
judge rather than by string matching.

**Step 1.** The judge sees the question, the reference answer(s) and the
model's answer, and replies with a single Yes or No. It writes back into each
generation entry:

| Field | Contents |
|---|---|
| `evaluation_verdict` | `true` / `false` / `null` (unparseable) |
| `evaluation_feedback` | the judge's raw reply |
| `evaluation_cost_usd` | estimated cost of that call |

**Step 2.** An answer is correct when the judge said Yes. How answers are
grouped depends on the dataset:

| Dataset | Grouped by |
|---|---|
| SD-QA | dialect, from the `dialect` field generation recorded. The same 699 questions are recorded in all 11 dialects, so the accuracies are directly comparable. |
| LibriSQA Part I | the speaker's gender, from the released id → gender map. Pass it with `--id_to_gender`; the lookup key is each item's `id`. Items missing from the map are skipped and reported. |

Either way a summary row gives the overall accuracy, the mean over groups,
and the gap between the best and worst one.

```bash
# SD-QA
python evaluation/sqa_judge.py \
    --inputs outputs/sqa \
    --output_dir judged/sqa \
    --model gpt-5.4-mini-2026-03-17 --concurrency 16
python evaluation/sqa.py --inputs judged/sqa --output_json results/sqa.json

# LibriSQA — same judge, grouped by the gold gender map
python evaluation/sqa_judge.py \
    --inputs outputs/sqa_librisqa \
    --output_dir judged/sqa_librisqa \
    --model gpt-5.4-mini-2026-03-17 --concurrency 16
python evaluation/sqa.py \
    --inputs judged/sqa_librisqa \
    --id_to_gender data/LibriSQA/id_2_gender_part1.json \
    --output_json results/sqa_librisqa.json
```

## `utils.py`

Not an evaluation. Holds `load_inputs()` (expands files, directories and
globs into loaded JSON), `print_table()` (fixed-width tables, floats to one
decimal, missing values as `-`) and `save_json()`.
