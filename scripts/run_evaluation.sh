#!/bin/bash
# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

# Score everything under $OUT. CPU only, no GPU needed.
#
#     ./scripts/run_evaluation.sh                        # audio_flamingo_3
#     ./scripts/run_evaluation.sh audio_flamingo_next
#
# The age axis needs two steps: an LLM-as-judge pass, then aggregation. The
# judge costs money, so it only runs when $OPENAI_API_KEY is set (or set
# RUN_JUDGE=0 to skip it explicitly).
#
# Environment overrides: OUT, JUDGED, RESULTS, JUDGE_MODEL, CONCURRENCY,
# RUN_JUDGE, DATA, LIBRISQA, JUDGED_LIBRISQA,
# TASKS (default "gender age bias asr classification fairspeech sqa asr_sdqa
#                 sqa_librisqa").

set -euo pipefail

MODEL="${1:-audio_flamingo_3}"
OUT="${OUT:-outputs/$MODEL}"
JUDGED="${JUDGED:-judged/$MODEL/age}"
JUDGED_SQA="${JUDGED_SQA:-judged/$MODEL/sqa}"
RESULTS="${RESULTS:-results/$MODEL}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"
CONCURRENCY="${CONCURRENCY:-16}"
DATA="${DATA:-data}"
LIBRISQA="${LIBRISQA:-$DATA/LibriSQA}"
JUDGED_LIBRISQA="${JUDGED_LIBRISQA:-judged/$MODEL/sqa_librisqa}"
TASKS="${TASKS:-gender age bias asr classification fairspeech sqa asr_sdqa sqa_librisqa}"

if [[ -z "${RUN_JUDGE:-}" ]]; then
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then RUN_JUDGE=1; else RUN_JUDGE=0; fi
fi

mkdir -p "$RESULTS"

for task in $TASKS; do
echo
echo "##################### $task #####################"
case "$task" in

  gender)
    python evaluation/gender_personalization.py \
      --inputs "$OUT/gender" \
      --output_json "$RESULTS/gender_personalization.json"
    ;;

  bias)
    python evaluation/stereotypical_bias.py \
      --inputs "$OUT/bias" \
      --output_json "$RESULTS/stereotypical_bias.json"
    ;;

  asr)
    python evaluation/asr.py \
      --inputs "$OUT/asr" \
      --output_json "$RESULTS/asr.json"
    ;;

  age)
    if [[ "$RUN_JUDGE" == "1" ]]; then
      echo "--- LLM-as-judge ($JUDGE_MODEL) ---"
      python evaluation/age_personalization_judge.py \
        --inputs "$OUT/age" \
        --output_dir "$JUDGED" \
        --model "$JUDGE_MODEL" \
        --concurrency "$CONCURRENCY" \
        --skip_judged
    else
      echo "--- judge skipped (no \$OPENAI_API_KEY, or RUN_JUDGE=0) ---"
    fi

    if [[ -d "$JUDGED" ]]; then
      python evaluation/age_personalization.py \
        --inputs "$JUDGED" \
        --output_json "$RESULTS/age_personalization.json"
    else
      echo "No judged outputs in $JUDGED; nothing to aggregate."
    fi
    ;;

  classification)
    python evaluation/classification.py \
      --inputs "$OUT/classification" \
      --output_json "$RESULTS/classification.json"
    ;;

  fairspeech)
    if [[ -d "$OUT/fairspeech" ]]; then
      python evaluation/asr_fairspeech.py \
        --inputs "$OUT/fairspeech" \
        --output_json "$RESULTS/fairspeech.json"
    else
      echo "No Fair-Speech transcripts in $OUT/fairspeech -- skipping."
    fi
    ;;

  sqa)
    if [[ "$RUN_JUDGE" == "1" ]] && [[ -d "$OUT/sqa" ]]; then
      echo "--- LLM-as-judge ($JUDGE_MODEL) ---"
      python evaluation/sqa_judge.py \
        --inputs "$OUT/sqa" \
        --output_dir "$JUDGED_SQA" \
        --model "$JUDGE_MODEL" \
        --concurrency "$CONCURRENCY" \
        --skip_judged
    elif [[ "$RUN_JUDGE" != "1" ]]; then
      echo "--- judge skipped (no \$OPENAI_API_KEY, or RUN_JUDGE=0) ---"
    else
      echo "No SQA answers in $OUT/sqa -- skipping."
    fi

    if [[ -d "$JUDGED_SQA" ]]; then
      python evaluation/sqa.py \
        --inputs "$JUDGED_SQA" \
        --output_json "$RESULTS/sqa.json"
    else
      echo "No judged answers in $JUDGED_SQA; nothing to aggregate."
    fi
    ;;

  asr_sdqa)
    if [[ -d "$OUT/asr_sdqa" ]]; then
      python evaluation/asr.py \
        --inputs "$OUT/asr_sdqa" \
        --gold_key question --group_key dialect \
        --output_json "$RESULTS/asr_sdqa.json"
    else
      echo "No SD-QA transcripts in $OUT/asr_sdqa -- skipping."
    fi
    ;;

  sqa_librisqa)
    if [[ "$RUN_JUDGE" == "1" ]] && [[ -d "$OUT/sqa_librisqa" ]]; then
      echo "--- LLM-as-judge ($JUDGE_MODEL) ---"
      python evaluation/sqa_judge.py \
        --inputs "$OUT/sqa_librisqa" \
        --output_dir "$JUDGED_LIBRISQA" \
        --model "$JUDGE_MODEL" \
        --concurrency "$CONCURRENCY" \
        --skip_judged
    elif [[ "$RUN_JUDGE" != "1" ]]; then
      echo "--- judge skipped (no \$OPENAI_API_KEY, or RUN_JUDGE=0) ---"
    else
      echo "No LibriSQA answers in $OUT/sqa_librisqa -- skipping."
    fi

    if [[ -d "$JUDGED_LIBRISQA" ]] && [[ -f "$LIBRISQA/id_2_gender_part1.json" ]]; then
      python evaluation/sqa.py \
        --inputs "$JUDGED_LIBRISQA" \
        --id_to_gender "$LIBRISQA/id_2_gender_part1.json" \
        --output_json "$RESULTS/sqa_librisqa.json"
    else
      echo "Need judged answers in $JUDGED_LIBRISQA and"
      echo "$LIBRISQA/id_2_gender_part1.json to aggregate -- skipping."
    fi
    ;;

  *)
    echo "ERROR: unknown task '$task' (expected: gender age bias asr classification fairspeech sqa asr_sdqa sqa_librisqa)" >&2
    exit 1
    ;;
esac
done

echo
echo "Done. Results are in $RESULTS"
