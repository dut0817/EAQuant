#!/usr/bin/env bash
set -euo pipefail

: "${EAQUANT_SYSTEM:?Set EAQUANT_SYSTEM to medmix_baseline or eaquant.}"
: "${EAQUANT_MODEL_PATH:?Set EAQUANT_MODEL_PATH to the FP16 model directory.}"
: "${EAQUANT_OUTPUT_DIR:?Set EAQUANT_OUTPUT_DIR for logs and trainer outputs.}"
: "${EAQUANT_SAVE_QMODEL_PATH:?Set EAQUANT_SAVE_QMODEL_PATH for the checkpoint.}"

case "${EAQUANT_SYSTEM}" in
  medmix_baseline|eaquant) ;;
  *)
    echo "EAQUANT_SYSTEM must be medmix_baseline or eaquant." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BACKEND_ROOT="${PROJECT_ROOT}/backends/ostquant"

export PYTHONPATH="${PROJECT_ROOT}/src:${BACKEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
NPROC="${EAQUANT_NPROC_PER_NODE:-${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}}"
if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid process count: ${NPROC}. Set EAQUANT_NPROC_PER_NODE to a positive integer." >&2
  exit 2
fi

mkdir -p "${EAQUANT_OUTPUT_DIR}" "$(dirname -- "${EAQUANT_SAVE_QMODEL_PATH}")"

ARGS=(
  --output_dir "${EAQUANT_OUTPUT_DIR}"
  --model "${EAQUANT_MODEL_PATH}"
  --seed "${EAQUANT_SEED:-0}"
  --seqlen 512
  --nsamples 128
  --train_dataset medmix
  --cal_dataset medmix
  --eval_dataset wikitext2
  --loss_type kl_top
  --post_attn=True
  --rotate_ov=True
  --rotate_post_rope=False
  --online_qk_hadamard=True
  --smooth_qk=True
  --smooth_ov=True
  --smooth_up_down=True
  --smooth_norm_linear=True
  --bf16=True
  --lm_eval True
  --max_steps 100
  --a_bits 4
  --v_bits 4
  --k_bits 4
  --down_bits 4
  --train_enable_wquant False
  --sub_mean False
  --distribute True
  --bsz 4
  --use_klt True
  --save_qmodel_path "${EAQUANT_SAVE_QMODEL_PATH}"
)

if [[ "${EAQUANT_SYSTEM}" == "medmix_baseline" ]]; then
  ARGS+=(--per_device_train_batch_size 4)
else
  LOSS_VARIANT="${EAQUANT_LOSS_VARIANT:-both}"
  ARGS+=(
    --rotate_down_dim 1
    --w_clip True
    --lm_eval_batch_size 4
    --per_device_train_batch_size 1
  )

  case "${LOSS_VARIANT}" in
    none) ;;
    token|recovery|both)
      : "${EAQUANT_CACHE_PATH:?Set EAQUANT_CACHE_PATH for EAQuant losses.}"
      TOKEN_WEIGHT="${EAQUANT_TOKEN_WEIGHT:-0.1}"
      RECOVERY_WEIGHT="${EAQUANT_RECOVERY_WEIGHT:-0.1}"
      if [[ "${LOSS_VARIANT}" == "token" ]]; then
        RECOVERY_WEIGHT=0.0
      elif [[ "${LOSS_VARIANT}" == "recovery" ]]; then
        TOKEN_WEIGHT=0.0
      fi
      ARGS+=(
        --explanation_loss_enabled True
        --remove_unused_columns False
        --faithfulness_cache_path "${EAQUANT_CACHE_PATH}"
        --faithfulness_answer_scoring_mode letter_option_mean_logprob
        --faithfulness_max_seq_len "${EAQUANT_FAITH_MAX_SEQ_LEN:-640}"
        --explanation_token_loss_weight "${TOKEN_WEIGHT}"
        --explanation_recovery_loss_weight "${RECOVERY_WEIGHT}"
      )
      ;;
    *)
      echo "EAQUANT_LOSS_VARIANT must be none, token, recovery, or both." >&2
      exit 2
      ;;
  esac
fi

ARGS+=("$@")
cd "${BACKEND_ROOT}"

if (( NPROC > 1 )); then
  exec "${TORCHRUN:-torchrun}" \
    --nnodes 1 \
    --nproc_per_node "${NPROC}" \
    --master_addr "${EAQUANT_MASTER_ADDR:-127.0.0.1}" \
    --master_port "${EAQUANT_MASTER_PORT:-29500}" \
    main.py "${ARGS[@]}"
fi

exec "${PYTHON:-python}" main.py "${ARGS[@]}"
