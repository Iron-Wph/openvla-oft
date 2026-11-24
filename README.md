# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

**Project website: https://openvla-oft.github.io/**

**Paper: https://arxiv.org/abs/2502.19645**

**Summary video: https://youtu.be/T3Zkkr_NTSA**


## RBench single-arm training

This repository contains the training code used for the RBench OpenVLA-OFT run on
two nodes with eight GPUs each. It is based on upstream OpenVLA-OFT commit
`e4287e9`.

| File | Purpose |
| --- | --- |
| [`scripts/rbench_libero_long.yaml`](scripts/rbench_libero_long.yaml) | Training configuration used for the 16-GPU run |
| [`scripts/train_rbench_2x8.sh`](scripts/train_rbench_2x8.sh) | Two-node launcher, dry run and checkpoint resume |
| [`vla-scripts/finetune.py`](vla-scripts/finetune.py) | Python training entry point |
| [`prismatic/vla/datasets/rlds/rbench.py`](prismatic/vla/datasets/rlds/rbench.py) | RBench dataset mixture and shared normalization |
| [`prismatic/vla/rbench_checkpoint.py`](prismatic/vla/rbench_checkpoint.py) | Training state save and restore |

The run uses OpenVLA-7B with LoRA rank 32, FiLM, image augmentation and a
continuous L1 action head. Global batch size is 64 (16 GPUs × 4 samples, one
optimizer update per batch). The budget is 150,000 optimizer steps. Learning
rate starts at `5e-4` and drops to `5e-5` after 100,000 updates. Checkpoints
are saved every 10,000 updates. This is the configured budget, not a claim that
the recorded run reached 150,000 steps.

The `rbench_all` mixture uses two RGB views, eight state values and eight-step
action chunks. Each action has seven absolute Panda joint targets in radians and
one gripper opening target in meters. The loaders use the same `BOUNDS`
normalization statistics from the training split.

### Prepare training data

Use the OpenVLA-OFT environment with `pyarrow`, `h5py`, `google-crc32c`
and Pillow installed. The input is the LeRobot v2.1 task tree and the original
HDF5 tree. The converter copies the two camera streams and action-bearing
steps from LeRobot, and reads the terminal observation from the matching HDF5.
For TRACK-CLEAR-01, use an HDF5 root that combines the original and archived
episodes under their original relative paths.

1. **Convert by task.** The two archived converters are
   [standard (RLDS 1.1.0)](scripts/rbench_data/standard/convert_rbench_rlds.py)
   and [segmented (RLDS 1.2.0)](scripts/rbench_data/segmented/convert_rbench_rlds.py).
   Both write one TFRecord shard per episode. The segmented version uses
   multiple records within a shard to keep large episodes below the single-record
   size limit. In the 33-task run, the segmented tasks were BOOK-EXTRACT-01,
   CARTON-SHIP-01, L-DISTILL-01, LID-PROP-01, R-ELECTRORECOVERY-01, R-FOAM-01,
   R-IRON-OXYGEN-01, TRACK-CLEAR-01, W-U02 and W-U15. The other 23 used standard
   format. The full split reserves the last two complete episodes per scene for
   validation. Assign each task to just one worker machine. The batch entry
   points are the `run_rbench_rlds_batch.py` files next to the two converters.

   ```bash
   # Set LEROBOT_ROOT, HDF5_ROOT, RLDS_ROOT and LOG_ROOT to your dataset paths.
   mkdir -p "$RLDS_ROOT" "$LOG_ROOT"
   python scripts/rbench_data/standard/run_rbench_rlds_batch.py \
     "$LEROBOT_ROOT" "$HDF5_ROOT" "$RLDS_ROOT" "$LOG_ROOT/C-TR-01" --tasks C-TR-01
   python scripts/rbench_data/segmented/run_rbench_rlds_batch.py \
     "$LEROBOT_ROOT" "$HDF5_ROOT" "$RLDS_ROOT" "$LOG_ROOT/TRACK-CLEAR-01" --tasks TRACK-CLEAR-01
   ```

2. **Compute the shared statistics after all tasks are published.**
   [`compute_norm_stats.py`](scripts/rbench_data/compute_norm_stats.py) uses
   the conversion job's `statistics()` function. It reads only the LeRobot
   `actions` and `state` columns for training episodes, then computes one
   frame-weighted set of statistics for the entire mixture. The manifest and
   statistics are linked by a digest. The formal split contained 3,762 training
   episodes and 198 validation episodes.

   ```bash
   python scripts/rbench_data/compute_norm_stats.py "$LEROBOT_ROOT" "$RLDS_ROOT"
   ```

   To reproduce statistics from a frozen split without replacing existing
   files, pass `--manifest /path/to/manifest.json --output /path/to/review-dir`.

3. **Compress the training images.** The archived
   [JPEG encoder](scripts/rbench_data/reencode_rbench_jpeg.py) resizes both
   RGB views to 256×256 with bicubic interpolation and writes JPEG quality 95.
   The [in-place driver](scripts/rbench_data/inplace.py) verifies each shard
   before replacement, updates TFDS metadata and keeps receipts. Use a separate
   work directory and keep training stopped until finalization. This operation
   replaces the original PNG shards in the specified RLDS directory. The
   LeRobot input, action/state fields, split and existing `norm_stats.json`
   are preserved.

   ```bash
   # Prepare once. Then run the middle command once for each task.
   python scripts/rbench_data/inplace.py "$RLDS_ROOT" "$JPEG_WORK" prepare
   python scripts/rbench_data/inplace.py "$RLDS_ROOT" "$JPEG_WORK" C-TR-01 --workers 8
   # Run once after every task has finished.
   python scripts/rbench_data/inplace.py "$RLDS_ROOT" "$JPEG_WORK" finalize
   ```

   If you have a previously verified JPEG staging tree with matching receipts,
   pass `--reuse /path/to/staging` to `prepare`. The staging files are moved
   into the dataset and must be on the same filesystem.

4. **Train.** Point `OFT_DATA_ROOT` at the finalized JPEG RLDS directory and
   use the launcher below.

### Launch on two nodes

Set up [the OpenVLA-OFT environment](SETUP.md) on both nodes. Both nodes need
access to the same prepared RBench RLDS directory (including `manifest.json`
and `norm_stats.json`), the same local OpenVLA-7B files, and a shared output
directory. Each node must see eight CUDA GPUs. From the repository root, set
these variables on both nodes to match your installation:

```bash
export OFT_DATA_ROOT=/shared/path/to/rbench-rlds
export OFT_BASE_MODEL=/shared/path/to/openvla-7b
export OFT_RUN_ROOT=/shared/path/to/oft-runs
export MASTER_ADDR=node-0-private-hostname
export RUN_ID=rbench-oft-run
# Set OFT_PYTHON if the OFT environment is not already activated.
```

Run the matching command on each node:

```bash
# Node 0
bash scripts/train_rbench_2x8.sh 0 "$MASTER_ADDR" "$RUN_ID"

# Node 1
bash scripts/train_rbench_2x8.sh 1 "$MASTER_ADDR" "$RUN_ID"
```

Append `--dry-run` to print the command without starting training.
`OFT_MASTER_PORT` optionally overrides port 29500 and must match on both
nodes. W&B defaults to offline mode. The training curve is recorded in
`$OFT_RUN_ROOT/$RUN_ID/training_curve.jsonl`.

To resume, keep the same run ID, shared data and run directories, and the
same two-node layout. Append
`--resume="$OFT_RUN_ROOT/$RUN_ID--100000_chkpt"` to both node commands,
replacing 100000 with the completed checkpoint step. The checkpoint needs
`COMPLETE.json` and `training_state.pt`, and the run's staged base model must
remain available under `$OFT_RUN_ROOT/.base-models/$RUN_ID`.


## System Requirements

Inference:
* 1 GPU with ~16 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with ~18 GB VRAM for ALOHA robot tasks

Training:
* Between 1-8 GPUs with 27-80 GB, depending on the desired training setup (with default bfloat16 data type). See [this FAQ on our project website](https://openvla-oft.github.io/#train-compute) for details.

## Quick Start

First, set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Then, run the Python script below to download a pretrained OpenVLA-OFT checkpoint and run inference to generate an action chunk:

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)
```

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the conda environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Support

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2502.19645):

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
