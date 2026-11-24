"""Publish the mixed RBench manifest and training-only normalization statistics."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from standard.convert_rbench_rlds import select_episodes, statistics


def mix_manifest(source, root):
    tasks = {
        path.name: "rbench_" + path.name.lower().replace("-", "")
        for path in sorted(source.iterdir())
        if (path / "episode_sources.jsonl").is_file()
    }
    if not tasks:
        raise ValueError(f"No LeRobot tasks found in {source}")
    versions = {}
    for task, name in tasks.items():
        available = [
            version for version in ("1.1.0", "1.2.0")
            if all((root / name / version / file).is_file()
                   for file in ("dataset_info.json", "features.json"))
        ]
        if len(available) != 1:
            raise ValueError(f"{task}: expected exactly one published RLDS version, got {available}")
        versions[task] = available[0]
    rows = select_episodes(source, tasks, full=True)
    for row in rows:
        row["rlds_version"] = versions[row["task_id"]]
        name, version, split = (row[key] for key in ("dataset", "rlds_version", "split"))
        shard = f"{name}-{split}.tfrecord-{row['shard']:05d}-of-{row['shard_count']:05d}"
        if not (root / name / version / shard).is_file():
            raise FileNotFoundError(root / name / version / shard)
    return rows


def publish(path, value):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as staged:
        json.dump(value, staged, ensure_ascii=False, indent=2)
        staged.flush()
        # Atomic creation also prevents overwriting an existing manifest or statistics.
        path.hardlink_to(staged.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LeRobot root")
    parser.add_argument("rlds_root", type=Path, help="root of the exported RLDS task directories")
    parser.add_argument("--manifest", type=Path, help="reuse an existing frozen manifest")
    parser.add_argument("--output", type=Path, help="directory for the manifest and statistics")
    args = parser.parse_args()
    root = args.rlds_root
    out = args.output or root
    out.mkdir(parents=True, exist_ok=True)
    outputs = (out / "manifest.json", out / "norm_stats.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Use a directory without an existing manifest or statistics")
    rows = json.loads(args.manifest.read_bytes()) if args.manifest else mix_manifest(args.source, root)
    result = statistics(args.source, rows)
    publish(outputs[1], result)
    publish(outputs[0], rows)
    print(f"Published {len(rows)} episodes and shared training statistics to {out}")


if __name__ == "__main__":
    main()
