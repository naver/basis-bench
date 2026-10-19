# Generation

One script per task. Each loads a model, builds the prompts for one
condition, and writes the model's raw output into JSON. Nothing is scored
here — that is [`evaluation/`](../evaluation/README.md).

| File | Task | Data |
|---|---|---|
| [`gender_personalization.py`](gender_personalization.py) | Gender personalization, 2-way MCQA | BASIS-Health / BASIS-Product |
| [`age_personalization.py`](age_personalization.py) | Age personalization, open-ended | ELIP-Pair |
| [`stereotypical_bias.py`](stereotypical_bias.py) | Stereotypical bias, 3-way MCQA | Spoken StereoSet+ |
| [`classification.py`](classification.py) | Demographic classification | all three of the above |
| [`asr.py`](asr.py) | Transcription | Spoken StereoSet+ ASR split, Fair-Speech, SD-QA |
| [`sqa.py`](sqa.py) | Spoken question answering | SD-QA, LibriSQA Part I |
| [`models.py`](models.py) | Not a task — shared model loading and batching | |

## Shared conventions

Every script takes the same core arguments:

| Argument | Meaning |
|---|---|
| `--input_json` | One dataset split. Scripts run one file at a time; the driver loops. |
| `--audio_root` | What the released `audio_path` values are relative to. **Differs per dataset** — see the table below. |
| `--output_folder` | Results land at `<output_folder>/<input file name>`. |
| `--model` | `audio_flamingo_3` or `audio_flamingo_next`. |
| `--model_path` | Override the checkpoint (defaults per model in `models.py`). |
| `--batch_size`, `--max_new_tokens` | Inference knobs. |
| `--debug N` | Only process the first N items. |

`--audio_root` per dataset, because each repository stores its paths
relative to a different base:

| Dataset | `--audio_root` |
|---|---|
| BASIS-Health / BASIS-Product | the split folder, `data/BASIS_health_product/health` or `.../product` |
| ELIP-Pair | `data/ELIP_Pair/audio` |
| Spoken StereoSet+ | `data/SpokenSS_plus` (its paths already include `audio/`) |
| Fair-Speech | `data/FairSpeech` |
| SD-QA | `data/SDQA` (its paths start with the dialect folder) |
| LibriSQA | `data/LibriSQA` (its paths start with `audio/`) |

**Output.** Each file is the input JSON with a results dict added per item,
keyed by model name. Most tasks write under `generations`; gender
personalization writes under `evaluation`, with one record per option order.
Re-running a different condition into the same folder **merges** into the
existing file by item index rather than replacing it, which is how the four
evaluation conditions end up side by side.

**Decoding** is greedy everywhere (`do_sample=False, temperature=0.0,
top_p=1.0`), set once in `models.py`.

## `gender_personalization.py`

Two-way MCQA: pick the candidate answer whose gender marker matches the
speaker. One of the four conditions per run, from two orthogonal arguments:

- `--modality {speech,text}` — how the question is delivered
- `--signal {implicit,explicit}` — whether the speaker's gender is in the prompt

Every item is prompted **twice**, once per option order (`male_first` /
`female_first`), so A/B position bias is measurable. Both records are kept
and count as independent datapoints at scoring time.

`--targeted_user {male,female}` is the speaker gender of the input file;
it goes into the prompt in the explicit conditions and is recorded either way.

```bash
python generation/gender_personalization.py \
    --input_json data/BASIS_health_product/health/female_health_personalization.json \
    --audio_root data/BASIS_health_product/health \
    --output_folder outputs/gender/speech_implicit \
    --model audio_flamingo_3 \
    --modality speech --signal implicit \
    --targeted_user female
```

## `age_personalization.py`

Open-ended generation: answer in the register matching the speaker's age.
Five conditions:

- `--modality {speech,text}` and `--signal {implicit,explicit}` as above
- `--baseline` — the text-baseline control, in which no response style is
  specified at all. Use with `--modality text --signal implicit`.

Scoring is a separate LLM-as-judge step; see
[`evaluation/age_personalization_judge.py`](../evaluation/age_personalization_judge.py).

```bash
python generation/age_personalization.py \
    --input_json data/ELIP_Pair/data/child_data.json \
    --audio_root data/ELIP_Pair/audio \
    --output_folder outputs/age/speech_implicit \
    --model audio_flamingo_3 \
    --modality speech --signal implicit \
    --targeted_user child
```

## `stereotypical_bias.py`

Three-way MCQA following the StereoSet context association test. **All six
orderings** of the three continuations are prompted per item, which is what
mitigates positional bias; each permutation counts separately at scoring
time.

`--demographic {gender,age}` says which subset the input file is, i.e. which
speaker map to use. In the explicit conditions the speaker's group is read
off the audio filename with the `GENDER_MAP` / `AGE_MAP` in this file, so
both groups in a subset are handled in one run.

```bash
python generation/stereotypical_bias.py \
    --input_json "data/SpokenSS_plus/SpokenStereoSet+/gender.json" \
    --audio_root data/SpokenSS_plus \
    --output_folder outputs/bias/speech_implicit \
    --model audio_flamingo_3 \
    --modality speech --signal implicit \
    --demographic gender
```

## `classification.py`

Can the model perceive the demographic cue at all? This is what separates
"ignored the cue" from "never heard it" when reading the other two axes. It
runs on exactly the same audio, so it takes the same input files.

`--task` picks the classes, which differ per dataset:

| `--task` | Classes | Use on |
|---|---|---|
| `gender` | male / female | all datasets |
| `age_young_elderly` | young / elderly | Spoken StereoSet+ |
| `age_child_adult` | child / adult | ELIP-Pair, BASIS-Health/Product |

`--gold` is the speaker group of the input file, recorded with every
prediction so the evaluation can break the counts down by it. It is
**required and never inferred**, so the input file has to hold a single
group. The two Spoken StereoSet+ files mix both groups and need splitting
per speaker group first.

```bash
python generation/classification.py \
    --input_json data/BASIS_health_product/health/female_health_personalization.json \
    --audio_root data/BASIS_health_product/health \
    --output_folder outputs/classification \
    --model audio_flamingo_3 \
    --task gender --gold female
```

## `asr.py`

Audio in, transcript out. Serves all three ASR datasets; they differ only in
which group label is recorded, so the evaluation knows how to aggregate:

| Dataset | Flag | Notes |
|---|---|---|
| Spoken StereoSet+ ASR split | `--age_group {child,young,elderly}` | reference transcript is `context` |
| SD-QA | `--dialect <one of 11>` | reference transcript is `question`; the only ASR run that sends a system prompt |
| Fair-Speech | neither | grouped by the demographic fields on each item instead |

```bash
# Spoken StereoSet+ ASR split
python generation/asr.py \
    --input_json data/SpokenSS_plus/ASR_split/elderly_data.json \
    --audio_root data/SpokenSS_plus \
    --output_folder outputs/asr \
    --model audio_flamingo_3 \
    --age_group elderly

# Fair-Speech
python generation/asr.py \
    --input_json data/FairSpeech/FairSpeech_ASR_data.json \
    --audio_root data/FairSpeech \
    --output_folder outputs/fairspeech \
    --model audio_flamingo_3

# SD-QA
python generation/asr.py \
    --input_json data/SDQA/aus/sdqa_aus.json \
    --audio_root data/SDQA \
    --output_folder outputs/asr_sdqa \
    --model audio_flamingo_3 \
    --dialect aus \
    --system_prompt "Transcribe the input speech."
```

## `sqa.py`

Spoken question answering. The two datasets put **opposite halves** of the
task in the audio, which `--dataset` swaps:

| `--dataset` | Audio carries | Text turn carries | Files |
|---|---|---|---|
| `sdqa` | the question, in one of 11 dialects | the passage | one per dialect, so `--dialect` is required |
| `librisqa` | the passage | the question | one, the whole Part I test set |

Answers are free-form and scored by an LLM-as-judge; see
[`evaluation/sqa_judge.py`](../evaluation/sqa_judge.py).

```bash
# SD-QA, Australian dialect
python generation/sqa.py \
    --input_json data/SDQA/aus/sdqa_aus.json \
    --audio_root data/SDQA \
    --output_folder outputs/sqa \
    --model audio_flamingo_3 \
    --dataset sdqa --dialect aus

# LibriSQA Part I
python generation/sqa.py \
    --input_json data/LibriSQA/LibriSQA-PartI-test.json \
    --audio_root data/LibriSQA \
    --output_folder outputs/sqa_librisqa \
    --model audio_flamingo_3 \
    --dataset librisqa
```

## `models.py`

Not a task. Holds `load_model()`, `prepare_conversation()` and `generate()`
for the two Audio Flamingo models, which are used through their Hugging Face
integrations and take the same conversation format — only the loading call
and the way the batch tensors are cast differ. `torch` and `transformers`
are imported inside the functions, so the prompts stay inspectable on a
machine without them installed.
