"""Re-encode one verified RBench RLDS shard at a time; keep source split and episode boundaries."""
import hashlib
import io
import json
import os
import struct
from pathlib import Path
from time import monotonic

import google_crc32c
import numpy as np
from PIL import Image
from tensorflow_datasets.proto.tf_example_generated_pb2 import Example

CAMERAS = ("steps/observation/image", "steps/observation/wrist_image")


def crc(data):
    value = google_crc32c.value(data)
    return struct.pack("<I", (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF)


def records(stream):
    while header := stream.read(8):
        if len(header) != 8 or stream.read(4) != crc(header):
            raise ValueError("Invalid TFRecord header")
        size = struct.unpack("<Q", header)[0]
        raw = stream.read(size)
        if len(raw) != size or stream.read(4) != crc(raw):
            raise ValueError("Invalid TFRecord payload")
        yield Example.FromString(raw)
        del raw


def image_data(png):
    with Image.open(io.BytesIO(png)) as image:
        if image.size != (640, 480) or image.format != "PNG":
            raise ValueError("Unexpected source camera format")
        return image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)


def fingerprint(fields):
    return {k: hashlib.sha256(v.SerializeToString()).hexdigest() for k, v in fields.items() if k not in CAMERAS}


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def transcode(job):
    src, dst, row = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    receipt = dst.with_name(dst.name + ".json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["source"] != row["source"] or saved["source_stat"] != [src.stat().st_size, src.stat().st_mtime_ns]:
            raise ValueError(f"Source changed for {dst}")
        if dst.stat().st_size != saved["out"] or hashlib.file_digest(dst.open("rb"), "sha256").hexdigest() != saved["sha256"]:
            raise ValueError(f"Published output changed: {dst}")
        return saved
    tmp = dst.with_name(dst.name + ".partial")
    start = monotonic()
    read = encode = write = 0.
    planned = row["frames"] + 1
    count = 0
    checks, samples = [], {}
    with src.open("rb") as reader, tmp.open("wb") as writer:
        iterator = records(reader)
        while True:
            tic = monotonic()
            example = next(iterator, None)
            read += monotonic() - tic
            if example is None:
                break
            tic = monotonic()
            fields = example.features.feature
            assert fields["episode_id"].bytes_list.value[0].decode() == row["source"]
            n = len(fields["steps/frame_index"].int64_list.value)
            assert list(fields["steps/frame_index"].int64_list.value) == list(range(count, count+n))
            checks.append((fingerprint(fields), n))
            for camera in CAMERAS:
                frames = fields[camera].bytes_list.value
                assert len(frames) == n
                for idx in range(n):
                    im = image_data(frames[idx])
                    buffer = io.BytesIO()
                    im.save(buffer, format="JPEG", quality=95)
                    frames[idx] = buffer.getvalue()
                    if count+idx in (0, planned//2, planned-1):
                        samples[camera, count+idx] = np.asarray(im, dtype=np.int16)
            encode += monotonic() - tic
            tic = monotonic()
            raw = example.SerializeToString()
            header = struct.pack("<Q", len(raw))
            writer.writelines((header, crc(header), raw, crc(raw)))
            del fields, frames, example, raw
            count += n
            write += monotonic() - tic
        os.fsync(writer.fileno())
    assert count == planned
    tic = monotonic()
    count = 0
    errors = []
    with tmp.open("rb") as reader:
        for example, (expected, n) in zip(records(reader), checks, strict=True):
            fields = example.features.feature
            assert fingerprint(fields) == expected
            for camera in CAMERAS:
                assert len(fields[camera].bytes_list.value) == n
                for idx, data in enumerate(fields[camera].bytes_list.value):
                    with Image.open(io.BytesIO(data)) as im:
                        assert im.format == "JPEG" and im.size == (256, 256) and im.mode == "RGB"
                        im.load()
                        key = (camera, count+idx)
                        if key in samples:
                            errors.append(float(np.abs(samples[key] - np.asarray(im)).mean()))
                            assert errors[-1] < 12, (key, errors[-1])
            count += n
    assert count == planned and len(errors) == 6
    with tmp.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    tmp.replace(dst)
    summary = dict(source=row["source"], source_stat=[src.stat().st_size, src.stat().st_mtime_ns],
                   dataset=row["dataset"], version=row["rlds_version"], images=count*2, records=len(checks),
                   inp=src.stat().st_size, out=dst.stat().st_size, sha256=digest, read=read, encode=encode,
                   write=write, verify=monotonic()-tic, seconds=monotonic()-start, max_sample_mae=max(errors))
    save(receipt, summary)
    return summary
