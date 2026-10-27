#!/usr/bin/env bash
# Single-node, eight-device XPU DMD2 distillation with HSDP shard=8.
# Additional --dotted.key value arguments override the YAML defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/FastWan2.2-TI2V-5B-FullAttn-Diffusers}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Wan2.2-Syn-121x704x1280_subset400}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/wan22_dmd_xpu_hsdp8}"

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    echo "Missing local model: ${MODEL_PATH}/model_index.json" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Missing local dataset: ${DATA_DIR}" >&2
    exit 1
fi

# Set the topology explicitly; the generic launcher otherwise probes nvidia-smi.
export NUM_GPUS=8 NNODES=1 NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29650}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

exec bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/dmd2_t2v_lora_hsdp8.yaml" \
    --models.student.init_from "${MODEL_PATH}" \
    --models.teacher.init_from "${MODEL_PATH}" \
    --models.critic.init_from "${MODEL_PATH}" \
    --training.data.data_path "${DATA_DIR}" \
    --training.checkpoint.output_dir "${OUTPUT_DIR}" \
    "$@"
