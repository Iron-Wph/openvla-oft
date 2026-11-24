"""Build the two-task RBench TFDS pilot and calculate shared training statistics."""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
from PIL import Image

TASKS = {"C-TR-01": "rbench_ctr01", "W-U07": "rbench_wu07"}
CAMERAS = ["image", "wrist_image"]
VERSION = "1.2.0"
MAX_RECORD_BYTES = 1900 * 1024**2


def features():
    import tensorflow_datasets as tfds

    f = tfds.features

    def image():
        return f.Image(shape=(480, 640, 3), encoding_format="png")

    def vec():
        return f.Tensor(shape=(8,), dtype=np.float32)

    return f.FeaturesDict(
        {
            "episode_id": f.Text(),
            "steps": f.Dataset(
                {
                    "observation": {
                        "image": image(),
                        "wrist_image": image(),
                        "state": vec(),
                    },
                    "action": vec(),
                    "timestamp": np.float32,
                    "frame_index": np.int64,
                    "is_first": np.bool_,
                    "is_last": np.bool_,
                    "is_terminal": np.bool_,
                    "language_instruction": f.Text(),
                }
            ),
        }
    )


def worker(job):
    import struct

    import google_crc32c
    from tensorflow_datasets.proto.tf_example_generated_pb2 import Example

    row, source, hdf5, target, instruction = job
    if not instruction.strip() or instruction.startswith("__PENDING__"):
        raise ValueError(f"Missing instruction for {row['source']}")
    tmp = target.with_name(target.name + ".partial")

    def checksum(data):
        crc = google_crc32c.value(data)
        return struct.pack(
            "<I", (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF
        )

    def new_record():
        record = Example()
        record.features.feature["episode_id"].bytes_list.value.append(
            row["source"].encode()
        )
        return record

    def write_record(record, stream):
        if record.ByteSize() >= MAX_RECORD_BYTES:
            raise ValueError(f"Segment exceeds record limit: {row['source']}")
        payload = record.SerializeToString()
        length = struct.pack("<Q", len(payload))
        stream.writelines((length, checksum(length), payload, checksum(payload)))

    with tmp.open("wb") as stream:
        records = write_episode(
            row, source, hdf5, instruction, stream, new_record, write_record
        )
    tmp.rename(target)
    return records


def write_episode(row, source, hdf5, instruction, stream, new_record, write_record):
    example = new_record()
    fields = example.features.feature
    records = 0
    image_bytes = 0
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=12, use_threads=False):
        cols = batch.to_pydict()
        batch_bytes = sum(len(item["bytes"]) for cam in CAMERAS for item in cols[cam])
        if image_bytes and image_bytes + batch_bytes > 384 * 1024**2:
            write_record(example, stream)
            records += 1
            example = new_record()
            fields = example.features.feature
            image_bytes = 0
        image_bytes += batch_bytes
        for cam in CAMERAS:
            imgs = [im["bytes"] for im in cols[cam]]
            if not all(im.startswith(b"\x89PNG\r\n\x1a\n") for im in imgs):
                raise ValueError("Expected embedded PNG")
            fields["steps/observation/" + cam].bytes_list.value.extend(imgs)
        for col, key in [
            ("actions", "action"),
            ("state", "observation/state"),
            ("timestamp", "timestamp"),
        ]:
            a = np.asarray(cols[col], dtype=np.float32)
            if col != "timestamp" and a.shape != (batch.num_rows, 8):
                raise ValueError(f"Incorrect {col} shape {a.shape}")
            fields["steps/" + key].float_list.value.extend(a.ravel())
        ids = list(range(offset, offset + batch.num_rows))
        if cols["frame_index"] != ids or set(cols["episode_index"]) != {
            row["episode_index"]
        }:
            raise ValueError("Frame or episode index mismatch")
        fields["steps/frame_index"].int64_list.value.extend(ids)
        for k, idx in [("is_first", 0), ("is_last", -1), ("is_terminal", -1)]:
            fields["steps/" + k].int64_list.value.extend(int(i == idx) for i in ids)
        fields["steps/language_instruction"].bytes_list.value.extend(
            [instruction.encode()] * len(ids)
        )
        offset += len(ids)
    if offset != row["frames"]:
        raise ValueError("Frame count mismatch")
    with h5py.File(hdf5, "r") as recording:
        done = recording["done"][:]
        if len(done) != offset or done[:-1].any() or not recording.attrs["complete"]:
            raise ValueError(f"Invalid terminal flags or incomplete recording: {hdf5}")
        final = recording["final_observation"]
        timestamp = float(final["timestamp"][()])
        if not np.isclose(
            timestamp, recording["next_timestamp"][-1], rtol=0, atol=1e-6
        ):
            raise ValueError(
                f"Final observation timestamp differs from final transition: {hdf5}"
            )
        relative_time = timestamp - float(recording["timestamp"][0])
        previous = fields["steps/timestamp"].float_list.value[-1]
        if (
            not np.isclose(
                previous,
                recording["timestamp"][-1] - recording["timestamp"][0],
                rtol=0,
                atol=1e-4,
            )
            or relative_time <= previous
        ):
            raise ValueError(
                f"Final observation does not follow the LeRobot action frames: {hdf5}"
            )
        state = final["observations/qpos"][:]
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid final joint state: {hdf5}")
        fields["steps/observation/state"].float_list.value.extend(state)
        for target_camera, source_camera in zip(CAMERAS, ("agentview", "wrist")):
            pixels = final[f"observations/images/{source_camera}"][:]
            if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
                raise ValueError(f"Invalid final image: {hdf5}/{source_camera}")
            with io.BytesIO() as encoded:
                Image.fromarray(pixels).save(encoded, format="PNG")
                fields["steps/observation/" + target_camera].bytes_list.value.append(
                    encoded.getvalue()
                )
        terminal = int(done[-1])
    fields["steps/action"].float_list.value.extend([0.0] * 8)
    fields["steps/timestamp"].float_list.value.append(relative_time)
    fields["steps/frame_index"].int64_list.value.append(offset)
    for flag, value in (("is_first", 0), ("is_last", 1), ("is_terminal", terminal)):
        fields["steps/" + flag].int64_list.value.append(value)
    fields["steps/language_instruction"].bytes_list.value.append(instruction.encode())

    write_record(example, stream)
    return records + 1


def count_records(path):
    import struct

    count = 0
    size = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < size:
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f"Incomplete TFRecord header: {path}")
            length = struct.unpack("<Q", header[:8])[0]
            if not 0 < length < MAX_RECORD_BYTES or stream.tell() + length + 4 > size:
                raise ValueError(f"Incomplete or oversized TFRecord: {path}")
            stream.seek(length + 4, 1)
            count += 1
    if not count:
        raise ValueError(f"Empty shard: {path}")
    return count


def select_episodes(root, tasks=TASKS, full=False):
    selected = []
    for task, name in tasks.items():
        manifest = root / task / "episode_sources.jsonl"
        rows = sorted(
            map(json.loads, manifest.read_text().splitlines()),
            key=lambda row: row["episode_index"],
        )
        task_metadata = root / task / "meta" / "tasks.jsonl"
        labels = (
            {
                entry["task_index"]: entry["task"]
                for entry in map(json.loads, task_metadata.read_text().splitlines())
            }
            if task_metadata.is_file()
            else {}
        )
        for scene in ("laboratory", "living_room", "warehouse"):
            eligible = [
                row for row in rows if row["scene"] == scene and not row["truncated"]
            ]
            if len(eligible) < 2:
                raise ValueError(f"{task}/{scene}: need two complete episodes")
            split_rows = (
                [
                    (row, "val" if index >= len(eligible) - 2 else "train")
                    for index, row in enumerate(eligible)
                ]
                if full
                else list(zip(eligible[:2], ("train", "val")))
            )
            if full and len(eligible) < 3:
                raise ValueError(
                    f"{task}/{scene}: need training data and two held-out episodes"
                )
            for row, split in split_rows:
                episode = row["episode_index"]
                training_instruction = row.get("training_instruction")
                metadata_instruction = labels.get(row.get("task_index"))
                if (
                    training_instruction is not None
                    and metadata_instruction != training_instruction
                ):
                    raise ValueError(
                        f"LeRobot language metadata disagrees with the source manifest: {row['source']}"
                    )
                instruction = training_instruction or row.get("recorded_instruction")
                if (
                    not isinstance(instruction, str)
                    or not instruction.strip()
                    or instruction.startswith("__PENDING__")
                ):
                    raise ValueError(f"Missing instruction for {row['source']}")
                row["instruction"] = instruction
                row.update(dataset=name, split=split)
                row["parquet"] = str(
                    Path(task)
                    / "data"
                    / f"chunk-{episode // 1000:03d}"
                    / f"episode_{episode:06d}.parquet"
                )
                selected.append(row)
    if len({row["source"] for row in selected}) != len(selected):
        raise ValueError("Duplicate episode ID")
    for name in tasks.values():
        for split in ("train", "val"):
            group = [
                row
                for row in selected
                if (row["dataset"], row["split"]) == (name, split)
            ]
            for shard, row in enumerate(group):
                row.update(shard=shard, shard_count=len(group))
    return selected


def shard_path(root, row):
    name, split, shard = (row[key] for key in ("dataset", "split", "shard"))
    return (
        root
        / name
        / VERSION
        / f"{name}-{split}.tfrecord-{shard:05d}-of-{row.get('shard_count', 3):05d}"
    )


def statistics(root, rows):
    arrays = defaultdict(list)
    counts = {
        name: {"num_trajectories": 0, "num_transitions": 0}
        for name in {row["dataset"] for row in rows}
    }
    for row in rows:
        if row["split"] != "train":
            continue
        table = pq.read_table(root / row["parquet"], columns=["actions", "state"])
        if len(table) != row["frames"]:
            raise ValueError(f"Wrong frame count: {row['source']}")
        for col, key in (("actions", "action"), ("state", "proprio")):
            values = np.asarray(table[col].to_pylist(), dtype=np.float32)
            if values.shape != (row["frames"], 8) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {col}: {row['source']}")
            arrays[key].append(values)
        counts[row["dataset"]]["num_trajectories"] += 1
        counts[row["dataset"]]["num_transitions"] += len(table)
    result = {}
    for key, parts in arrays.items():
        values = np.concatenate(parts).astype(np.float64)
        result[key] = {
            stat: getattr(values, stat)(axis=0).tolist()
            for stat in ("mean", "std", "min", "max")
        }
        result[key].update(
            q01=np.quantile(values, 0.01, axis=0).tolist(),
            q99=np.quantile(values, 0.99, axis=0).tolist(),
        )
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"stats": result, "datasets": counts, "manifest_sha256": digest}


def source_hdf5(root, row):
    path = root / row["source"]
    if path.is_file():
        return path
    if row["task_id"] == "TRACK-CLEAR-01":
        archived = root.parent / ".hf-download/track-clear-update-2b8dbba42fd1/staging/dataset" / row["source"]
        if archived.is_file():
            return archived
    raise FileNotFoundError(f"Missing original HDF5 for {row['source']}: {path}")


def build(
    source,
    output,
    overrides,
    reuse,
    hdf5_root,
    tasks=TASKS,
    full=False,
    workers=4,
    resume=False,
):
    rows = select_episodes(source, tasks, full)
    for row in rows:
        row["rlds_version"] = VERSION
    if overrides.keys() - {row["source"] for row in rows}:
        raise ValueError("Unknown override episode")
    if reuse:
        if not overrides or json.loads((reuse / "manifest.json").read_text()) != rows:
            raise ValueError(
                "Language update requires overrides and a matching source manifest"
            )
        if (reuse / "language_overrides.json").read_text() != "{}":
            raise ValueError("Use the original export as the language-update baseline")
    else:
        stats = statistics(source, rows)
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=resume)
    jobs = []
    for row in rows:
        target = shard_path(staging, row)
        target.parent.mkdir(parents=True, exist_ok=True)
        if resume and target.is_file():
            continue
        if reuse and row["source"] not in overrides:
            os.link(shard_path(reuse, row), target)
        else:
            jobs.append(
                (
                    row,
                    source / row["parquet"],
                    source_hdf5(hdf5_root, row),
                    target,
                    overrides.get(row["source"], row["instruction"]),
                )
            )
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        list(pool.map(worker, jobs))
    for name in tasks.values():
        directory = staging / name / VERSION
        splits = [
            tfds.core.SplitInfo(
                name=split,
                shard_lengths=[
                    count_records(shard_path(staging, row))
                    for row in rows
                    if row["dataset"] == name and row["split"] == split
                ],
                num_bytes=sum(
                    p.stat().st_size for p in directory.glob(f"*-{split}.tfrecord-*")
                ),
            )
            for split in ("train", "val")
        ]
        tfds.folder_dataset.write_metadata(
            data_dir=directory,
            features=features(),
            split_infos=splits,
            version=VERSION,
            check_data=False,
        )
    (staging / "manifest.json").write_text(json.dumps(rows, indent=2))
    (staging / "language_overrides.json").write_text(
        json.dumps(overrides, ensure_ascii=False)
    )
    stats_file = staging / "norm_stats.json"
    if reuse:
        os.link(reuse / stats_file.name, stats_file)
    else:
        stats_file.write_text(json.dumps(stats, indent=2))
    staging.rename(output)
    print(f"Published {len(rows)} episodes, wrote {len(jobs)} shards: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language-overrides", type=Path)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    edits = (
        json.loads(args.language_overrides.read_text())
        if args.language_overrides
        else {}
    )
    tasks = (
        TASKS
        if not args.tasks
        else {task: "rbench_" + task.lower().replace("-", "") for task in args.tasks}
    )
    build(
        args.source,
        args.output,
        edits,
        args.reuse,
        args.hdf5_root,
        tasks,
        args.full,
        args.workers,
        args.resume,
    )
