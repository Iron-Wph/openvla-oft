"""Shared configuration for RBench's single-arm OpenVLA-OFT mix."""

import hashlib
import json
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import example_parser

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ROBOT_PLATFORM


def keep_action_steps(trajectory):
    """Exclude the final RLDS observation from imitation targets and action windows."""
    last = trajectory["is_last"]
    tf.debugging.assert_equal(last, tf.range(tf.shape(last)[0]) == tf.shape(last)[0] - 1)
    return tf.nest.map_structure(lambda value: value[:-1], trajectory)


class SegmentedEpisodeReader:
    """Present each multi-record shard as one logical trajectory to dlimp."""

    def __init__(self, root, name, shards):
        self.data_dir = str(Path(root) / name / "1.2.0")
        self.shards = shards
        self.info = tfds.builder_from_directory(self.data_dir).info

    def as_dataset(self, *, split, shuffle_files, decoders, read_config):
        paths = self.shards["train"] + self.shards["val"] if split == "all" else self.shards[split]
        parser = example_parser.ExampleParser(self.info.features.get_serialized_info())
        files = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle_files:
            files = files.shuffle(len(paths), seed=read_config.shuffle_seed)

        def load_episode(path):
            parts = tf.data.TFRecordDataset(path).map(parser.parse_example, num_parallel_calls=1)
            return parts.ragged_batch(64).map(join_segments, num_parallel_calls=1)

        # Limit whole-episode buffering. A single logical episode can exceed 2 GiB.
        return files.interleave(load_episode, cycle_length=1, num_parallel_calls=1)


def join_segments(parts):
    ids = parts["episode_id"]
    tf.debugging.assert_equal(ids, tf.fill(tf.shape(ids), ids[0]))
    steps = tf.nest.map_structure(lambda item: item.merge_dims(0, 1), parts["steps"])
    order = steps["frame_index"]
    tf.debugging.assert_equal(order, tf.range(tf.size(order, out_type=order.dtype), dtype=order.dtype))
    return {"episode_id": ids[0], "steps": steps}


def dataset_kwargs(data_root_dir, train):
    if ROBOT_PLATFORM != "RBENCH":
        raise ValueError("Select the RBench platform with --dataset_name rbench_all")
    root = Path(data_root_dir)
    stats = json.loads((root / "norm_stats.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if any(row.get("rlds_version") not in {"1.1.0", "1.2.0"} for row in manifest):
        raise ValueError("Re-export RBench data with terminal observations (RLDS 1.1.0)")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != stats["manifest_sha256"]:
        raise ValueError("RBench manifest and normalization statistics do not match")
    split = "train" if train else "val"
    datasets = []
    for name in stats["datasets"]:
        task_rows = [row for row in manifest if row["dataset"] == name]
        rows = [row for row in task_rows if row["split"] == split]
        segmented = {row["rlds_version"] for row in task_rows} == {"1.2.0"}
        read_options = {}
        if segmented:
            read_options["episode_shards"] = {key: [] for key in ("train", "val")}
            for row in task_rows:
                label = row["split"]
                file = f"{name}-{label}.tfrecord-{row['shard']:05d}-of-{row['shard_count']:05d}"
                read_options["episode_shards"][label].append(str(root / name / "1.2.0" / file))
        counts = {"num_transitions": sum(row["frames"] for row in rows), "num_trajectories": len(rows)}
        datasets.append(
            dict(
                name=name,
                standardize_fn=keep_action_steps,
                data_dir=str(root),
                image_obs_keys={"primary": "image", "wrist": "wrist_image"},
                state_obs_keys=["state"],
                language_key="language_instruction",
                absolute_action_mask=[True] * 8,
                action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
                dataset_statistics={**stats["stats"], **counts},
                **read_options,
            )
        )
    return datasets, [1.0] * len(datasets)
