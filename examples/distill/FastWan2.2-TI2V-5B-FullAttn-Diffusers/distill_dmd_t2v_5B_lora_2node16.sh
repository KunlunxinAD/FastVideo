#!/usr/bin/env bash
# Two-node, eight-process-per-node XPU DMD2 smoke/stability test.
# Device selection belongs to the caller: this script never changes visibility.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

: "${MASTER_ADDR:?Set the rendezvous address reachable from both nodes}"
: "${NODE_RANK:?Set NODE_RANK=0 or NODE_RANK=1}"
: "${CUDA_VISIBLE_DEVICES:?Explicitly select exactly eight permitted devices}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo "Invalid NODE_RANK" >&2; exit 2; }
IFS=, read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
[[ ${#VISIBLE_DEVICES[@]} == 8 ]] || { echo "Exactly eight visible devices are required" >&2; exit 2; }
[[ -n "${VISIBLE_DEVICES[7]}" ]] || { echo "Empty visible device ID" >&2; exit 2; }
[[ "${MAX_STEPS:-5}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be a positive integer" >&2; exit 2; }

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/FastWan2.2-TI2V-5B-FullAttn-Diffusers}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Wan2.2-Syn-121x704x1280_subset800}"
RUN_ID="${RUN_ID:-wan22_dmd_2node16}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-5}"
[[ -f "${MODEL_PATH}/model_index.json" ]] || { echo "Missing local model: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${DATA_DIR}/map_style_cache/file_info.pkl" ]] || { echo "Run prepare_2node_data.py on each node before launch" >&2; exit 2; }

export NUM_GPUS=8 NNODES=2 NODE_RANK MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29760}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/node${NODE_RANK}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
HEALTH_ARGS=()
if [[ "${HEALTH_CHECK:-0}" == 1 ]]; then
    HEALTH_ARGS+=(--callbacks.health._target_ dmd_health_check.DMDHealthCheck)
fi

# Interface names may be overridden independently on the two hosts.
export BKCL_SOCKET_IFNAME="${BKCL_SOCKET_IFNAME:-ens21np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${BKCL_SOCKET_IFNAME}}"
export BKCL_RDMA_NICS="${BKCL_RDMA_NICS:-ens21np0,ens22np0,ens23np0,ens24np0,ens25np0,ens26np0,ens27np0,ens28np0}"
export BKCL_USE_RDMA="${BKCL_USE_RDMA:-1}"
export BKCL_TREE_THRESHOLD="${BKCL_TREE_THRESHOLD:-262144}"
export BKCL_RING_BUFFER_SIZE="${BKCL_RING_BUFFER_SIZE:-8388608}"
export BKCL_FORCE_L3_RDMA="${BKCL_FORCE_L3_RDMA:-0}"
export CUDA_ENABLE_P2P_NO_UVA="${CUDA_ENABLE_P2P_NO_UVA:-1}"
export BKCL_USE_PEERMEM_XDR="${BKCL_USE_PEERMEM_XDR:-1}"
export BKCL_FLAT_RING="${BKCL_FLAT_RING:-1}"
export BKCL_DEBUG="${BKCL_DEBUG:-0}"
export BKCL_TIMEOUT="${BKCL_TIMEOUT:-180}"

printf 'Run=%s node=%s CUDA_VISIBLE_DEVICES=%s HSDP=(2,8) steps=%s\n' \
    "${RUN_ID}" "${NODE_RANK}" "${CUDA_VISIBLE_DEVICES}" "${MAX_STEPS}"
exec bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/dmd2_t2v_lora_hsdp8.yaml" \
    --models.student.init_from "${MODEL_PATH}" \
    --models.teacher.init_from "${MODEL_PATH}" \
    --models.critic.init_from "${MODEL_PATH}" \
    --training.distributed.num_gpus 16 \
    --training.distributed.sp_size 1 \
    --training.distributed.tp_size 1 \
    --training.distributed.hsdp_replicate_dim 2 \
    --training.distributed.hsdp_shard_dim 8 \
    --training.data.data_path "${DATA_DIR}" \
    --training.data.require_existing_parquet_cache true \
    --training.loop.max_train_steps "${MAX_STEPS}" \
    --training.checkpoint.output_dir "${OUTPUT_DIR}" \
    --training.checkpoint.training_state_checkpointing_steps 0 \
    --training.tracker.trackers '[none]' \
    --training.tracker.run_name "${RUN_ID}" \
    "${HEALTH_ARGS[@]}" \
    "$@"
