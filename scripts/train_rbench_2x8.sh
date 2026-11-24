#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <node-rank: 0|1> <node-0-private-address> <run-id> [--dry-run|--smoke|--verify|--resume=CHECKPOINT]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "$#" -lt 3 || "$#" -gt 4 || ! "$1" =~ ^[01]$ ]]; then
    usage >&2
    exit 2
fi

node_rank=$1
master_addr=$2
run_id=$3
mode=${4:-}
if [[ ! "$master_addr" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ || ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Use a private IPv4 address or hostname and a run ID containing letters, digits, dots, underscores or hyphens." >&2
    exit 2
fi

smoke_args=()
restore_from=""
case "$mode" in
    ""|--dry-run) ;;
    --resume=*)
        restore_from=${mode#--resume=}
        [[ -s "$restore_from/COMPLETE.json" && -s "$restore_from/training_state.pt" ]] || exit 2
        smoke_args=(--training_checkpoint="$restore_from")
        ;;
    --verify)
        run_id="$run_id-verify"
        smoke_args=(--max_steps=2 --save_freq=2 --num_steps_before_decay=2 --wandb_log_freq=1)
        ;;
    --verify-resume)
        run_id="$run_id-verify"
        restore_from="${OFT_RUN_ROOT:-/public_fast_storage/wph/oft_runs}/$run_id--2_chkpt"
        smoke_args=(--training_checkpoint="$restore_from" --max_steps=3 --save_freq=3 --num_steps_before_decay=2 --wandb_log_freq=1)
        ;;
    --smoke)
        run_id="$run_id-smoke"
        smoke_args=(--max_steps=200 --save_freq=200)
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_dir"
project_dir=$(cd "$repo_dir/../.." && pwd -P)
python_bin=${OFT_PYTHON:-python}
master_port=${OFT_MASTER_PORT:-29500}
if [[ ! "$master_port" =~ ^[1-9][0-9]{0,4}$ ]] || ((master_port > 65535)); then
    echo "OFT_MASTER_PORT must be a port number from 1 to 65535." >&2
    exit 2
fi
data_root=${OFT_DATA_ROOT:-/public_fast_storage/wph/rlds_dataset}
base_model=${OFT_BASE_MODEL:-"$project_dir/models/openvla-7b"}
run_root=${OFT_RUN_ROOT:-/public_fast_storage/wph/oft_runs}
# Keep disk-backed dataset caches and temporary files local to each node.
local_cache="/var/tmp/rbench-oft-$UID"
export HF_DATASETS_CACHE="$local_cache/datasets"
export TMPDIR="$local_cache/tmp"
config_path="$repo_dir/scripts/rbench_libero_long.yaml"
staged_model="$run_root/.base-models/$run_id"

for path in "$data_root" "$base_model" "$run_root"; do
    if [[ "$path" != /* ]]; then
        echo "OFT_DATA_ROOT, OFT_BASE_MODEL and OFT_RUN_ROOT must be absolute paths shared by both nodes." >&2
        exit 2
    fi
done

export OPENVLA_ROBOT_PLATFORM=RBENCH
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

command=(
    "$python_bin" -m torch.distributed.run
    --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank"
    --master-addr="$master_addr" --master-port="$master_port" --max-restarts=0
    "$repo_dir/vla-scripts/finetune.py"
    --config_path="$config_path"
    --data_root_dir="$data_root"
    --vla_path="$staged_model"
    --run_root_dir="$run_root"
    --run_id_override="$run_id"
    ${smoke_args[@]+"${smoke_args[@]}"}
)

if [[ "$mode" == "--dry-run" ]]; then
    printf 'Node %s, 8 GPUs, shared model source %s\n' "$node_rank" "$base_model"
    printf 'Node-local datasets cache %s, temporary files %s\n' "$HF_DATASETS_CACHE" "$TMPDIR"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p -- "$HF_DATASETS_CACHE" "$TMPDIR"

if [[ ! -s "$data_root/manifest.json" || ! -s "$data_root/norm_stats.json" || ! -s "$base_model/config.json" ]]; then
    echo "Missing shared RBench manifest, normalization statistics or OpenVLA model configuration." >&2
    exit 1
fi

"$python_bin" - "$base_model" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

required = (
    "accelerate", "dlimp", "draccus", "peft", "tensorflow",
    "tensorflow_datasets", "timm", "torch", "transformers", "wandb",
)
missing = [module for module in required if importlib.util.find_spec(module) is None]
if missing:
    sys.exit("Use the OpenVLA-OFT training environment. Missing: " + ", ".join(missing))

model = Path(sys.argv[1])
index = json.loads((model / "model.safetensors.index.json").read_text())
for shard in set(index["weight_map"].values()):
    if not (model / shard).is_file():
        sys.exit(f"Missing OpenVLA weight shard: {model / shard}")

import torch

visible = torch.cuda.device_count()
if visible != 8 or not torch.distributed.is_nccl_available():
    sys.exit(f"This launcher needs 8 visible CUDA GPUs and NCCL; found {visible} GPUs.")
PY

if [[ "$node_rank" == 0 && -z "$restore_from" ]]; then
    mkdir -p "$run_root/.base-models"
    if ! mkdir "$staged_model"; then
        echo "Run ID '$run_id' already has a staging directory. Use a new run ID." >&2
        exit 1
    fi
    # OFT updates the model configuration and Python files in place. Keep run-owned copies.
    for file in "$base_model"/*; do
        [[ -f "$file" ]] || continue
        case "$file" in
            *.safetensors) ln -s "$file" "$staged_model/" ;;
            *.json|*.py|*.model) cp -pL "$file" "$staged_model/" ;;
        esac
    done
fi

exec "${command[@]}"
