#!/bin/bash
# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

# Generation for one model, over all four tasks and all conditions.
#
# Submit from the repository root:
#
#     mkdir -p logs
#     sbatch scripts/run_generation.sbatch audio_flamingo_3 af3
#     sbatch scripts/run_generation.sbatch audio_flamingo_next af_next
#
#     $1  model       audio_flamingo_3 | audio_flamingo_next
#     $2  conda env   the environment to activate for that model
#
# Environment overrides: DATA (default ./data), OUT (default ./outputs/$MODEL),
# TASKS (default "gender age bias asr classification fairspeech sqa asr_sdqa
#                 sqa_librisqa"),
# LIBRISQA (default $DATA/LibriSQA; skipped if absent),
# SDQA (default $DATA/SDQA; skipped if absent),
# FAIRSPEECH (default $DATA/FairSpeech; skipped if absent),
# DEBUG (e.g. DEBUG="--debug 20").

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint="gpu_a100&gpu_80g"
#SBATCH --job-name=basis_bench_gen
#SBATCH --output=logs/basis_bench_gen_%j.out

set -euo pipefail

MODEL="${1:-audio_flamingo_3}"
CONDA_ENV="${2:-af3}"

DATA="${DATA:-data}"
OUT="${OUT:-outputs/$MODEL}"
TASKS="${TASKS:-gender age bias asr classification fairspeech sqa asr_sdqa sqa_librisqa}"
DEBUG="${DEBUG:-}"

CONDITIONS=("speech implicit" "speech explicit" "text implicit" "text explicit")

source ~/.bashrc
conda activate "$CONDA_ENV"
nvidia-smi || true

echo "model=$MODEL  data=$DATA  out=$OUT  tasks=$TASKS"

for task in $TASKS; do
echo
echo "##################### $task #####################"
case "$task" in

  gender)
    BASIS="$DATA/BASIS_health_product"
    for domain in health product; do
      for gender in female male; do
        for cond in "${CONDITIONS[@]}"; do
          modality="${cond% *}"; signal="${cond#* }"
          echo "--- gender | $domain | $gender | ${modality}_${signal} ---"
          python generation/gender_personalization.py \
            --input_json "$BASIS/$domain/${gender}_${domain}_personalization.json" \
            --audio_root "$BASIS/$domain" \
            --output_folder "$OUT/gender/${modality}_${signal}" \
            --model "$MODEL" \
            --modality "$modality" --signal "$signal" \
            --targeted_user "$gender" \
            $DEBUG
        done
      done
    done
    ;;

  age)
    ELIP="$DATA/ELIP_Pair"
    for group in adult child; do
      for cond in "${CONDITIONS[@]}"; do
        modality="${cond% *}"; signal="${cond#* }"
        echo "--- age | $group | ${modality}_${signal} ---"
        python generation/age_personalization.py \
          --input_json "$ELIP/data/${group}_data.json" \
          --audio_root "$ELIP/audio" \
          --output_folder "$OUT/age/${modality}_${signal}" \
          --model "$MODEL" \
          --modality "$modality" --signal "$signal" \
          --targeted_user "$group" \
          $DEBUG
      done

      # text-baseline: default behaviour, no response style specified.
      echo "--- age | $group | text_baseline ---"
      python generation/age_personalization.py \
        --input_json "$ELIP/data/${group}_data.json" \
        --audio_root "$ELIP/audio" \
        --output_folder "$OUT/age/text_baseline" \
        --model "$MODEL" \
        --modality text --signal implicit --baseline \
        --targeted_user "$group" \
        $DEBUG
    done
    ;;

  bias)
    SS="$DATA/SpokenSS_plus"
    for demographic in gender age; do
      for cond in "${CONDITIONS[@]}"; do
        modality="${cond% *}"; signal="${cond#* }"
        echo "--- bias | $demographic | ${modality}_${signal} ---"
        python generation/stereotypical_bias.py \
          --input_json "$SS/SpokenStereoSet+/${demographic}.json" \
          --audio_root "$SS" \
          --output_folder "$OUT/bias/${modality}_${signal}" \
          --model "$MODEL" \
          --modality "$modality" --signal "$signal" \
          --demographic "$demographic" \
          $DEBUG
      done
    done
    ;;

  asr)
    SS="$DATA/SpokenSS_plus"
    for group in child young elderly; do
      echo "--- asr | $group ---"
      python generation/asr.py \
        --input_json "$SS/ASR_split/${group}_data.json" \
        --audio_root "$SS" \
        --output_folder "$OUT/asr" \
        --model "$MODEL" \
        --age_group "$group" \
        $DEBUG
    done
    ;;

  classification)
    # Demographic classification, on the same audio as the axes above.
    BASIS="$DATA/BASIS_health_product"
    ELIP="$DATA/ELIP_Pair"
    SS="$DATA/SpokenSS_plus"
    CLS_OUT="$OUT/classification"

    # BASIS-Health / BASIS-Product: gender, one group per file.
    for domain in health product; do
      for gender in female male; do
        echo "--- classification | basis $domain | gender | $gender ---"
        python generation/classification.py \
          --input_json "$BASIS/$domain/${gender}_${domain}_personalization.json" \
          --audio_root "$BASIS/$domain" \
          --output_folder "$CLS_OUT/basis_$domain" \
          --model "$MODEL" \
          --task gender --gold "$gender" \
          $DEBUG
      done
    done

    # ELIP-Pair: age as child/adult, one group per file.
    for group in adult child; do
      echo "--- classification | elip | age_child_adult | $group ---"
      python generation/classification.py \
        --input_json "$ELIP/data/${group}_data.json" \
        --audio_root "$ELIP/audio" \
        --output_folder "$CLS_OUT/elip_pair" \
        --model "$MODEL" \
        --task age_child_adult --gold "$group" \
        $DEBUG
    done

    # Spoken StereoSet+ is not run here: gender.json and age.json each mix
    # both groups in one file, and --gold is a single required label. Split
    # them per speaker group first, then run e.g.
    #
    #   python generation/classification.py \
    #     --input_json <female speakers only>.json --audio_root "$SS" \
    #     --output_folder "$CLS_OUT/spoken_stereoset" --model "$MODEL" \
    #     --task gender --gold female
    #
    # The speaker is the part of the audio filename after the context id
    # ("00002_en-US-AvaNeural.wav"); generation/stereotypical_bias.py has the
    # speaker -> group maps.
    ;;

  fairspeech)
    # Fair-Speech is downloaded separately (see the README) and joined into
    # one JSON by prepare_fairspeech.py before it can be transcribed.
    FS="${FAIRSPEECH:-$DATA/FairSpeech}"
    if [[ ! -d "$FS" ]]; then
      echo "Fair-Speech not found at $FS -- skipping."
      echo "See the Fair-Speech section of the README to download it, or set"
      echo "FAIRSPEECH=/path/to/FairSpeech."
    else
      if [[ ! -f "$FS/FairSpeech_ASR_data.json" ]]; then
        echo "--- fairspeech | building FairSpeech_ASR_data.json ---"
        python prepare_fairspeech.py --fairspeech_root "$FS"
      fi
      echo "--- fairspeech | transcribing ---"
      python generation/asr.py \
        --input_json "$FS/FairSpeech_ASR_data.json" \
        --audio_root "$FS" \
        --output_folder "$OUT/fairspeech" \
        --model "$MODEL" \
        $DEBUG
    fi
    ;;

  sqa)
    # SD-QA spoken question answering: one file per dialect, all covering the
    # same 699 questions. Downloaded separately, see the README.
    SDQA="${SDQA:-$DATA/SDQA}"
    if [[ ! -d "$SDQA" ]]; then
      echo "SD-QA not found at $SDQA -- skipping."
      echo "See the SD-QA section of the README to download it, or set"
      echo "SDQA=/path/to/SDQA."
    else
      for dialect in aus gbr ind_n ind_s irl kenya nga nzl phl usa zaf; do
        echo "--- sqa | $dialect ---"
        python generation/sqa.py \
          --input_json "$SDQA/$dialect/sdqa_${dialect}.json" \
          --audio_root "$SDQA" \
          --output_folder "$OUT/sqa" \
          --model "$MODEL" \
          --dataset sdqa --dialect "$dialect" \
          $DEBUG
      done
    fi
    ;;

  asr_sdqa)
    # SD-QA ASR: transcribe the spoken questions, to read WER per dialect.
    # Unlike the other ASR runs this one sends a system prompt.
    SDQA="${SDQA:-$DATA/SDQA}"
    if [[ ! -d "$SDQA" ]]; then
      echo "SD-QA not found at $SDQA -- skipping."
      echo "See the SD-QA section of the README to download it, or set"
      echo "SDQA=/path/to/SDQA."
    else
      for dialect in aus gbr ind_n ind_s irl kenya nga nzl phl usa zaf; do
        echo "--- asr_sdqa | $dialect ---"
        python generation/asr.py \
          --input_json "$SDQA/$dialect/sdqa_${dialect}.json" \
          --audio_root "$SDQA" \
          --output_folder "$OUT/asr_sdqa" \
          --model "$MODEL" \
          --dialect "$dialect" \
          --system_prompt "Transcribe the input speech." \
          $DEBUG
      done
    fi
    ;;

  sqa_librisqa)
    # LibriSQA Part I: the passage is spoken and the question is text.
    # Downloaded separately, see the README. Scored per speaker gender.
    LIBRISQA="${LIBRISQA:-$DATA/LibriSQA}"
    if [[ ! -d "$LIBRISQA" ]]; then
      echo "LibriSQA not found at $LIBRISQA -- skipping."
      echo "See the LibriSQA section of the README to download it, or set"
      echo "LIBRISQA=/path/to/LibriSQA."
    else
      echo "--- sqa_librisqa | part I test ---"
      python generation/sqa.py \
        --input_json "$LIBRISQA/LibriSQA-PartI-test.json" \
        --audio_root "$LIBRISQA" \
        --output_folder "$OUT/sqa_librisqa" \
        --model "$MODEL" \
        --dataset librisqa \
        $DEBUG
    fi
    ;;

  *)
    echo "ERROR: unknown task '$task' (expected: gender age bias asr classification fairspeech sqa asr_sdqa sqa_librisqa)" >&2
    exit 1
    ;;
esac
done

echo
echo "Done. Generations are in $OUT"
