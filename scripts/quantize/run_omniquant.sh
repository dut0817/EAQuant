#!/usr/bin/env bash
set -euo pipefail

: "${EAQUANT_SYSTEM:?Set EAQUANT_SYSTEM to medmix_baseline or eaquant.}"
: "${EAQUANT_MODEL_PATH:?Set EAQUANT_MODEL_PATH to the FP16 model directory.}"
: "${EAQUANT_MODEL_KEY:?Set EAQUANT_MODEL_KEY to the OmniQuant --net value.}"
: "${EAQUANT_OUTPUT_DIR:?Set EAQUANT_OUTPUT_DIR for the checkpoint and logs.}"
: "${EAQUANT_ACT_SCALES:?Set EAQUANT_ACT_SCALES to the seed-matched .pt file.}"
: "${EAQUANT_ACT_SHIFTS:?Set EAQUANT_ACT_SHIFTS to the seed-matched .pt file.}"

case "${EAQUANT_SYSTEM}" in
  medmix_baseline|eaquant) ;;
  *)
    echo "EAQUANT_SYSTEM must be medmix_baseline or eaquant." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BACKEND_ROOT="${PROJECT_ROOT}/backends/omniquant"

export PYTHONPATH="${PROJECT_ROOT}/src:${BACKEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${EAQUANT_OUTPUT_DIR}"

ARGS=(
  --model "${EAQUANT_MODEL_PATH}"
  --net "${EAQUANT_MODEL_KEY}"
  --output_dir "${EAQUANT_OUTPUT_DIR}"
  --cache_dir "${EAQUANT_OMNI_CACHE_DIR:-${BACKEND_ROOT}/cache}"
  --calib_dataset medmix
  --nsamples 128
  --seqlen 512
  --seed "${EAQUANT_SEED:-0}"
  --batch_size 1
  --epochs 20
  --wbits 4
  --abits 4
  --lwc
  --let
  --aug_loss
  --attn_implementation eager
  --act-scales "${EAQUANT_ACT_SCALES}"
  --act-shifts "${EAQUANT_ACT_SHIFTS}"
)

if [[ "${EAQUANT_SYSTEM}" == "eaquant" ]]; then
  : "${EAQUANT_CACHE_PATH:?Set EAQUANT_CACHE_PATH to the Omni effect cache.}"
  ARGS+=(
    --explanation_loss_enabled
    --faithfulness_cache_path "${EAQUANT_CACHE_PATH}"
    --faithfulness_cache_limit 0
    --faithfulness_max_seq_len "${EAQUANT_FAITH_MAX_SEQ_LEN:-640}"
    --faithfulness_batch_size 4
    --faithfulness_every_n_steps 32
    --faithfulness_loss_type logit_kl
    --faithfulness_answer_scoring_mode letter_option_mean_logprob
    --faithfulness_weight_source omni_quant
    --explanation_token_loss_weight 0.1
    --explanation_recovery_loss_weight 0.1
  )
fi

ARGS+=("$@")
cd "${BACKEND_ROOT}"
exec "${PYTHON:-python}" main.py "${ARGS[@]}"
