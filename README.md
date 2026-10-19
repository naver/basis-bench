<div align="center">

# BASIS-Bench

_**B**ias **A**nd per**S**onalization **I**n **S**peech LLMs **Bench**mark_

**Personalization Without Stereotyping: Do Speech LLMs Know When to Turn a Deaf Ear?**

Kshitij Ambilduke, Caroline Brun, Thibaut Thonet, Vassilina Nikoulina, Marcely Zanon Boito

**[Interactive results webpage](https://mzboito.github.io/basis-bench-webpage/)**


</div>

---

Speech carries demographic cues (age, gender) that text does not. A good
speech assistant should use them when the user asks for a tailored answer,
and ignore them otherwise. **BASIS-Bench** separates these two behaviors,
treating demographic personalization as desirable *only when explicitly
requested*, along three axes:

<img src="assets/basis_bench_overview.svg" alt="BASIS-Bench overview: the three evaluation axes and their datasets" width="750">


Bias and personalization are run on speech and on text inputs, with the
demographic either left implicit or stated explicitly (four conditions).
This lets us trace demographic-dependent behavior to the **acoustic**
encoder, the language **backbone**, or their **interaction**. Demographic
robustness is evaluated on speech input only.

## This repository

Generation and evaluation code for BASIS-Bench. It reads the released data
repositories, prompts a speech LLM over them, and scores the results.
Generation only writes model outputs to JSON; every metric is computed
afterwards from those files.

| Data | Axis | Task |
|---|---|---|
| BASIS-Health / BASIS-Product | Personalization (gender) | 2-way MCQA |
| ELIP-Pair | Personalization (age) | Open-ended, LLM-as-judge |
| Spoken StereoSet+ | Stereotypical bias | 3-way MCQA (CAT / ICAT) |
| Spoken StereoSet+ (ASR split) | Demographic robustness | ASR / WER |
| all three of the above | Demographic robustness | Age / gender classification |
| Fair-Speech | Demographic robustness | ASR / WER by speaker demographic |
| SD-QA | Demographic robustness | Spoken QA by dialect, LLM-as-judge |
| SD-QA | Demographic robustness | ASR / WER by dialect |
| LibriSQA (Part I) | Demographic robustness | Spoken QA by gender, LLM-as-judge |

BASIS-Health, BASIS-Product, ELIP-Pair and Spoken StereoSet+ are released
with the paper; Fair-Speech, SD-QA and LibriSQA are external (see
[Setup](#setup)).

The paper evaluates eight speech LLMs. This release currently ships loaders
for `audio_flamingo_3` and `audio_flamingo_next`; others can be added in
[`generation/models.py`](generation/models.py).

## Layout

```
clone_data.sh          fetch the three data repositories into ./data
prepare_fairspeech.py  join the Fair-Speech TSV + audio folder into one JSON
prompts/               every prompt, for every condition, one file per task
generation/            one script per task, plus models.py (loading + batching)
evaluation/            one script per evaluation (the two judges write
                       verdicts, their partner script aggregates them)
scripts/               Slurm driver for generation, shell driver for evaluation
```

Each of those two folders has its own README describing what every file in
it does, its arguments and an example command:
[`generation/README.md`](generation/README.md) and
[`evaluation/README.md`](evaluation/README.md).

## Setup

```bash
./clone_data.sh                     # -> data/{BASIS_health_product,ELIP_Pair,SpokenSS_plus}
pip install -r requirements.txt
```

The audio is stored with git-lfs, so `git lfs install` must have been run once
on the machine. Generation and evaluation can live in separate environments:
only generation needs `torch` / `transformers`.

### Fair-Speech

Fair-Speech is not in those repositories and is not redistributable, so it
has to be downloaded separately from Meta, who gate it behind a data
agreement: <https://ai.meta.com/datasets/speech-fairness-dataset/>.

It arrives as two pieces, a folder of wavs and a metadata TSV keyed by each
wav's hash name. Put them together as:

```
data/FairSpeech/
├── metadata.tsv
└── audios/
    └── asr_fairness_audio/
        ├── 0003b9543cad190a9f9217200a768a23.wav
        └── ...
```

Then join them into the item format the pipeline reads:

```bash
python prepare_fairspeech.py --fairspeech_root data/FairSpeech
```

That writes `data/FairSpeech/FairSpeech_ASR_data.json` -- one entry per
utterance with its transcript and the speaker's self-reported demographics:

```json
{
    "audio_path": "audios/asr_fairness_audio/ca1920051b7fdc279c45b5211b261cf0.wav",
    "gender": "female",
    "age": "18 - 22",
    "first_language": "English",
    "ethnicity": "White",
    "transcript": "hey facebook answer the call"
}
```

Rows whose wav is missing, or whose transcript is blank, are dropped and
reported. The released TSV has 26,471 rows and one blank transcript, so the
result is 26,470 utterances. `scripts/run_generation.sbatch` runs this step
itself if the JSON is not there yet, and skips Fair-Speech entirely if
`data/FairSpeech` does not exist.

### SD-QA

SD-QA is also downloaded separately, from Hugging Face:
<https://huggingface.co/datasets/KshitijAmbilduke/SD-QA>. It needs no preparation
step. It already ships one JSON per dialect, in the item format the
pipeline reads. Put it inside the `data/` folder as:

```
data/SDQA/
├── aus/
│   ├── sdqa_aus.json
│   └── audio/
├── gbr/
└── ...            # ind_n, ind_s, irl, kenya, nga, nzl, phl, usa, zaf
```

The `audio_path` values are relative to `data/SDQA` and start with the
dialect folder, so generation is run with `--audio_root data/SDQA`. Each file
holds the same 699 questions in a different dialect (7,689 utterances over
the 11 dialects), which is what makes the per-dialect numbers comparable.
SD-QA feeds two tasks, `sqa` (answer the spoken question) and `asr_sdqa`
(transcribe it). `scripts/run_generation.sbatch` skips both if `data/SDQA`
does not exist.

### LibriSQA

LibriSQA Part I is downloaded separately, from Hugging Face:
<https://huggingface.co/datasets/KshitijAmbilduke/LibriSQA-Part1>. Only Part I is
used. Put it inside the `data/` folder as:

```
data/LibriSQA/
├── LibriSQA-PartI-test.json   # 2,620 items
├── id_2_gender_part1.json     # id -> "M" / "F", the gold labels
└── audio/                     # <id>.wav, the spoken passages
```

No preparation step is needed. The `audio_path` values are relative to
`data/LibriSQA` and start with `audio/`, so generation is run with
`--audio_root data/LibriSQA`.

LibriSQA is the mirror image of SD-QA: here the **passage** is spoken and the
**question** is given as text, so `generation/sqa.py` takes
`--dataset librisqa` to swap which half goes into the audio. Scores are
grouped by the speaker's gender, which `evaluation/sqa.py` looks up in
`id_2_gender_part1.json` by each item's `id` (1,389 female, 1,231 male).
`scripts/run_generation.sbatch` skips LibriSQA if `data/LibriSQA` does not
exist.

## Conditions

Only Bias and Personalization run under the four conditions of the paper,
crossing modality with whether the speaker's demographic is stated outright:

| Condition | `--modality` | `--signal` | Demographic |
|---|---|---|---|
| speech implicit | `speech` | `implicit` | only in the voice — the target setting |
| speech explicit | `speech` | `explicit` | also stated in the prompt |
| text implicit | `text` | `implicit` | absent — isolates the language backbone |
| text explicit | `text` | `explicit` | stated in the prompt |

Demographic robustness (ASR, spoken QA and demographic classification) has
no conditions: it is always speech in, with a fixed prompt per task.

Age personalization adds a fifth, `--baseline` (with `--modality text
--signal implicit`): no response style is specified at all, which measures
the model's default behavior.

## Generation

Run everything for one model:

```bash
mkdir -p logs
sbatch scripts/run_generation.sbatch audio_flamingo_3 af3
sbatch scripts/run_generation.sbatch audio_flamingo_next af_next
```

The second argument is the conda environment to activate. Outputs go to
`outputs/<model>/<task>/<condition>/<input file name>`.

Or a single condition by hand:

```bash
# gender personalization, speech implicit, female speakers, BASIS-Health
python generation/gender_personalization.py \
    --input_json data/BASIS_health_product/health/female_health_personalization.json \
    --audio_root data/BASIS_health_product/health \
    --output_folder outputs/audio_flamingo_3/gender/speech_implicit \
    --model audio_flamingo_3 \
    --modality speech --signal implicit \
    --targeted_user female

# age personalization, speech explicit, child speakers
python generation/age_personalization.py \
    --input_json data/ELIP_Pair/data/child_data.json \
    --audio_root data/ELIP_Pair/audio \
    --output_folder outputs/audio_flamingo_3/age/speech_explicit \
    --model audio_flamingo_3 \
    --modality speech --signal explicit \
    --targeted_user child

# stereotypical bias, text explicit, age subset
python generation/stereotypical_bias.py \
    --input_json "data/SpokenSS_plus/SpokenStereoSet+/age.json" \
    --audio_root data/SpokenSS_plus \
    --output_folder outputs/audio_flamingo_3/bias/text_explicit \
    --model audio_flamingo_3 \
    --modality text --signal explicit \
    --demographic age

# ASR, elderly speakers
python generation/asr.py \
    --input_json data/SpokenSS_plus/ASR_split/elderly_data.json \
    --audio_root data/SpokenSS_plus \
    --output_folder outputs/audio_flamingo_3/asr \
    --model audio_flamingo_3 \
    --age_group elderly

# demographic classification, gender, female BASIS-Health speakers
python generation/classification.py \
    --input_json data/BASIS_health_product/health/female_health_personalization.json \
    --audio_root data/BASIS_health_product/health \
    --output_folder outputs/audio_flamingo_3/classification/basis_health \
    --model audio_flamingo_3 \
    --task gender --gold female

# Fair-Speech ASR (same script; no --age_group, it is grouped per item instead)
python generation/asr.py \
    --input_json data/FairSpeech/FairSpeech_ASR_data.json \
    --audio_root data/FairSpeech \
    --output_folder outputs/audio_flamingo_3/fairspeech \
    --model audio_flamingo_3

# SD-QA spoken question answering, Australian dialect
python generation/sqa.py \
    --input_json data/SDQA/aus/sdqa_aus.json \
    --audio_root data/SDQA \
    --output_folder outputs/audio_flamingo_3/sqa \
    --model audio_flamingo_3 \
    --dataset sdqa --dialect aus

# LibriSQA Part I spoken question answering (spoken passage, text question)
python generation/sqa.py \
    --input_json data/LibriSQA/LibriSQA-PartI-test.json \
    --audio_root data/LibriSQA \
    --output_folder outputs/audio_flamingo_3/sqa_librisqa \
    --model audio_flamingo_3 \
    --dataset librisqa

# SD-QA ASR, same files, same script as the other ASR runs
python generation/asr.py \
    --input_json data/SDQA/aus/sdqa_aus.json \
    --audio_root data/SDQA \
    --output_folder outputs/audio_flamingo_3/asr_sdqa \
    --model audio_flamingo_3 \
    --dialect aus \
    --system_prompt "Transcribe the input speech."
```

Demographic classification runs on the same audio as the axes above, so it
takes the same input files. Which `--task` to use depends on the dataset,
because the age classes differ:

| Input file | `--task` | Classes | `--gold` |
|---|---|---|---|
| BASIS `{female,male}_{health,product}_personalization.json` | `gender` | male / female | `female` / `male` |
| ELIP-Pair `{adult,child}_data.json` | `age_child_adult` | child / adult | `adult` / `child` |
| Spoken StereoSet+ `gender.json` | `gender` | male / female | see below |
| Spoken StereoSet+ `age.json` | `age_young_elderly` | young / elderly | see below |

`--gold` is the speaker group of the input file. It is required and never
inferred, and is recorded alongside each prediction so the evaluation can
break the counts down by it. The input file therefore has to hold a single
group.

The two Spoken StereoSet+ files each mix both groups, so they need splitting
per speaker group before classification can run on them. The speaker is the
part of the audio filename after the context id
(`00002_en-US-AvaNeural.wav`), and `generation/stereotypical_bias.py` holds
the speaker to group maps. `scripts/run_generation.sbatch` therefore covers
BASIS and ELIP-Pair only.

`--audio_root` differs per repository because the released `audio_path`
values are relative to different bases: the split folder for BASIS
(`health/` or `product/`), `audio/` for ELIP-Pair, and the repository root
for Spoken StereoSet+ (whose paths already include `audio/`).

Every script also takes `--debug N` to process only the first N items,
`--batch_size`, `--max_new_tokens` and `--model_path` (to override the
checkpoint). Re-running a different condition into the same output folder
merges into the existing file rather than replacing it.

## Evaluation

```bash
./scripts/run_evaluation.sh audio_flamingo_3       # all four, writes results/
```

Or one at a time:

```bash
python evaluation/gender_personalization.py --inputs outputs/audio_flamingo_3/gender
python evaluation/stereotypical_bias.py     --inputs outputs/audio_flamingo_3/bias
python evaluation/asr.py                    --inputs outputs/audio_flamingo_3/asr
python evaluation/asr.py --inputs outputs/audio_flamingo_3/asr_sdqa \
                        --gold_key question --group_key dialect
python evaluation/classification.py         --inputs outputs/audio_flamingo_3/classification
python evaluation/asr_fairspeech.py         --inputs outputs/audio_flamingo_3/fairspeech

# age needs the judge first
export OPENAI_API_KEY=...
python evaluation/age_personalization_judge.py \
    --inputs outputs/audio_flamingo_3/age \
    --output_dir judged/audio_flamingo_3/age \
    --model gpt-5.4-mini-2026-03-17 --concurrency 16
python evaluation/age_personalization.py --inputs judged/audio_flamingo_3/age

# SD-QA also needs the judge first
python evaluation/sqa_judge.py \
    --inputs outputs/audio_flamingo_3/sqa \
    --output_dir judged/audio_flamingo_3/sqa \
    --model gpt-5.4-mini-2026-03-17 --concurrency 16
python evaluation/sqa.py --inputs judged/audio_flamingo_3/sqa

# LibriSQA: same judge, then grouped by the gold gender map
python evaluation/sqa_judge.py \
    --inputs outputs/audio_flamingo_3/sqa_librisqa \
    --output_dir judged/audio_flamingo_3/sqa_librisqa \
    --model gpt-5.4-mini-2026-03-17 --concurrency 16
python evaluation/sqa.py \
    --inputs judged/audio_flamingo_3/sqa_librisqa \
    --id_to_gender data/LibriSQA/id_2_gender_part1.json
```

`--inputs` takes files, directories or globs. All scripts accept
`--output_json` to also write the tables to disk.

**Scoring rules.**

- *Gender personalization* — accuracy, with every perturbed example counted
  independently: each item was prompted twice, once per option order, and
  both count. A model that always answers "A" scores 50%.
- *Age personalization* — the judge decides which persona each response is
  tailored for, on expertise and style only; a response is correct when that
  matches the speaker's age group.
- *Stereotypical bias* — proportions of stereotypical / anti-stereotypical /
  irrelevant continuations, with every option permutation counted
  independently (six per item). ICAT is reported alongside, per demographic
  group and then averaged.
- *Demographic classification* — how often each demographic is predicted,
  counted per input file. Where generation recorded `--gold`, the same
  counts are also broken down by the speaker's true group, with accuracy.
- *SD-QA* — spoken QA accuracy per dialect. The answers are free-form, so
  a judge is shown the question, the reference answer(s) and the model's
  answer and replies Yes or No; accuracy is the share of Yes verdicts.
  A summary row gives the mean over the 11 dialects and the gap between
  the best and worst one.
- *Fair-Speech* — word error rate per speaker demographic. Grouped by
  gender and by first language (bucketed English / non-English, since all
  29 non-English values together are a fifth of the English ones; pass
  `--raw_language` to keep them separate). `--group_by` also accepts `age`
  and `ethnicity`. A final table gives the corpus WER gap between the best
  and worst group.
- *LibriSQA* — the same judge as SD-QA, but grouped by the speaker's gold
  gender from `id_2_gender_part1.json` rather than by dialect. Items whose
  `id` is absent from the map are skipped and reported.
- *SD-QA ASR* — the same scorer as below, with `--gold_key question` (the
  reference is the spoken question, not the passage) and `--group_key
  dialect`. The summary row gives the mean over dialects and the
  best-to-worst gap.
- *ASR* — word error rate after Whisper normalization, reported both as
  corpus WER and as `100 − WER` with the per-utterance WER capped at 100.

`evaluation/age_personalization_judge.py --dry_run` prints one judge prompt
and exits without calling the API.

## Prompts

Everything lives in [`prompts/`](prompts/), one module per task, holding the
system prompts, the user-turn templates for every condition, and a
`build_prompt()` that assembles them. The strings are transcribed verbatim
from the scripts that produced the reported numbers, so a few trailing spaces
and newlines inside them are intentional.

## Output format

Each output file is the input JSON with a results dict added per item, keyed
by model name.

Gender personalization writes under `evaluation`, one record per option order:

```json
"evaluation": {
  "audio_flamingo_3": {
    "male_first":   {"raw_output": "B", "choice": "B",
                     "option_A": "male", "option_B": "female",
                     "option_selected": "female", "correct": true,
                     "speaker_gender": "female", "condition": "speech_implicit",
                     "modality": "speech", "signal": "implicit"},
    "female_first": {"...": "..."}
  }
}
```

Age personalization and ASR write one entry under `generations` with an
`output` field; the judge adds `judge_choice`, `winrate_feedback` and
`judge_cost_usd`. Stereotypical bias writes the run metadata plus one
`perm_<i>` entry per option ordering, each holding the A/B/C option texts and
the raw response.

## Citation

```bibtex
@misc{basisbench2026,
  title         = {Personalization Without Stereotyping: Do Speech {LLMs} Know When to Turn a Deaf Ear?},
  author        = {Ambilduke, Kshitij and Brun, Caroline and Thonet, Thibaut and Nikoulina, Vassilina and Boito, Marcely Zanon},
  year          = {2026},
  eprint        = {2610.13550},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.13550}
}
```

## Flaticon Icons
The icons from our image above come from [flaticon.com](https://www.flaticon.com/). These icons are not covered by the repository's Apache-2.0 license and remain subject to the Flaticon [terms of use](https://media.flaticon.com/license/license.pdf). 

The attribution for the three number icons: <a href="https://www.flaticon.com/fr/icones-gratuites/nombres" title="Number one">number one (Abrrakhlaed2626 - Flaticon)</a>, <a href="https://www.flaticon.com/fr/icones-gratuites/numero-2" title="numéro 2 icônes">number two (Abrrakhlaed2626 - Flaticon)</a>, <a href="https://www.flaticon.com/free-icons/number-3" title="number 3 icons">number three (Abrrakhlaed2626 - Flaticon)</a>.


The attribution for the three other icons, from top to bottom: <a href="https://www.flaticon.com/free-icons/participation" title="First icon"> first icon (Magnific - Flaticon)</a>, <a href="https://www.flaticon.com/free-animated-icons/comparison" title="Second icon">second icon (Magnific - Flaticon)</a>, <a href="https://www.flaticon.com/free-icons/target-audience" title="Third icon">third icon (Magnific - Flaticon)</a>.

