"""Queue complete tasks for a bucket-backed RBench export."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from convert_rbench_rlds import MAX_RECORD_BYTES, VERSION, select_episodes, statistics


def estimated_record_size(source, row):
    meta = pq.ParquetFile(source / row["parquet"]).metadata
    cameras = [
        column.total_uncompressed_size
        for group in range(meta.num_row_groups)
        for idx in range(meta.num_columns)
        if (column := meta.row_group(group).column(idx)).path_in_schema.split(".")[0]
        in ("image", "wrist_image")
    ]
    if not cameras:
        raise ValueError(f"No image column sizes in {row['source']}")
    return (
        sum(cameras)
        + row["frames"] * (len(row["instruction"].encode()) + 256)
        + 4 * 1024**2
    )


def run(args):
    task_dirs = sorted(
        p for p in args.source.iterdir() if (p / "episode_sources.jsonl").is_file()
    )
    names = {p.name: "rbench_" + p.name.lower().replace("-", "") for p in task_dirs}
    assert len(names) == 33 and "TRACK-CLEAR-01" in names
    rows = select_episodes(args.source, names, full=True)
    assert len(rows) == 3960
    args.logs.mkdir(parents=True, exist_ok=True)
    queue = [task for task in names if task in args.tasks]
    queue = list(dict.fromkeys(queue))
    queue = [task for task in queue if task in args.tasks]
    print(f"Queued {len(queue)} tasks with lossless segmented storage", flush=True)
    staging = args.bucket / ".staging"
    staging.mkdir(exist_ok=True)

    def convert(task):
        name = names[task]
        work = staging / task
        destination = args.bucket / name
        if destination.exists():
            return task, "already_present"
        command = [
            sys.executable,
            str(Path(__file__).with_name("convert_rbench_rlds.py")),
            str(args.source),
            str(work),
            "--hdf5-root",
            str(args.hdf5_root),
            "--tasks",
            task,
            "--full",
            "--resume",
            "--workers",
            "4",
        ]
        with (args.logs / (task + ".log")).open("w") as log:
            print(f"START {task}", flush=True)
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode:
            print(f"FAILED {task}: {result.returncode}", flush=True)
            return task, "failed"
        manifest = json.loads((work / "manifest.json").read_text())
        expected = {row["source"] for row in rows if row["task_id"] == task}
        assert {row["source"] for row in manifest} == expected
        config = work / name / VERSION / "dataset_info.json"
        assert config.is_file()
        (args.logs / f"{task}.manifest.json").write_text(json.dumps(manifest, indent=2))
        (work / name).rename(destination)
        print(f"PUBLISHED {task}: {len(manifest)} episodes", flush=True)
        return task, "published"

    with ThreadPoolExecutor(max_workers=1) as executor:
        status = dict(executor.map(convert, queue))
    (args.logs / "status.json").write_text(json.dumps(status, indent=2))
    if any(state == "failed" for state in status.values()):
        raise SystemExit(1)
    if len(queue) == len(names) and all(
        state == "published" for state in status.values()
    ):
        manifests = [
            row
            for task in queue
            for row in json.loads((args.logs / f"{task}.manifest.json").read_text())
        ]
        (args.bucket / "manifest.json").write_text(json.dumps(manifests, indent=2))
        (args.bucket / "norm_stats.json").write_text(
            json.dumps(statistics(args.source, manifests), indent=2)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "hdf5_root", "bucket", "logs"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    run(parser.parse_args())
