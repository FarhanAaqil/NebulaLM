"""Pack token ids into 256-token windows.

Eos is appended once per document and carries zero text bytes.
Pad positions are not prediction targets. Segment ids block cross-document attention.
"""

from __future__ import annotations

import struct
from pathlib import Path

CONTEXT = 256


def _distribute_bytes(token_count: int, byte_count: int) -> list[int]:
    if token_count <= 0:
        return []
    base, extra = divmod(byte_count, token_count)
    return [base + (1 if index < extra else 0) for index in range(token_count)]


def pack_documents(
    documents: list[tuple[str, list[int], int]],
    *,
    eos_id: int,
    pad_id: int,
    context: int = CONTEXT,
) -> list[dict]:
    """Pack (document_id, content_ids, utf8_bytes) into context windows."""
    stream_ids: list[int] = []
    stream_docs: list[str] = []
    stream_bytes: list[int] = []
    stream_segments: list[int] = []
    segment = 0
    for document_id, content_ids, byte_count in documents:
        if eos_id in content_ids or pad_id in content_ids:
            raise ValueError(f"{document_id} already contains a control id")
        pieces = list(content_ids) + [eos_id]
        byte_parts = _distribute_bytes(len(content_ids), byte_count) + [0]
        if len(pieces) != len(byte_parts):
            raise RuntimeError("byte distribution length mismatch")
        stream_ids.extend(pieces)
        stream_docs.extend([document_id] * len(pieces))
        stream_bytes.extend(byte_parts)
        stream_segments.extend([segment] * len(pieces))
        segment += 1
    windows = []
    for start in range(0, len(stream_ids), context):
        chunk_ids = stream_ids[start : start + context]
        chunk_docs = stream_docs[start : start + context]
        chunk_bytes = stream_bytes[start : start + context]
        chunk_segments = stream_segments[start : start + context]
        pad_count = context - len(chunk_ids)
        next_pad_segment = -1
        pad_segments = []
        for _ in range(pad_count):
            pad_segments.append(next_pad_segment)
            next_pad_segment -= 1
        input_ids = chunk_ids + [pad_id] * pad_count
        segment_ids = chunk_segments + pad_segments
        loss_mask = [0] * context
        for index in range(1, context):
            if input_ids[index] != pad_id:
                loss_mask[index] = 1
        windows.append(
            {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "segment_ids": segment_ids,
                "byte_count": sum(chunk_bytes),
                "document_ids": sorted(set(chunk_docs)),
            }
        )
    return windows


def write_shard(path: Path, windows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"NBLM")
        handle.write(struct.pack("<I", 1))
        handle.write(struct.pack("<I", len(windows)))
        for window in windows:
            if len(window["input_ids"]) != CONTEXT:
                raise ValueError("shard windows must be fixed length")
            handle.write(struct.pack(f"<{CONTEXT}i", *window["input_ids"]))
            handle.write(struct.pack(f"<{CONTEXT}B", *window["loss_mask"]))
            handle.write(struct.pack(f"<{CONTEXT}i", *window["segment_ids"]))
            handle.write(struct.pack("<I", window["byte_count"]))


def read_shard(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"NBLM":
            raise ValueError("not a NebulaLM shard")
        (version,) = struct.unpack("<I", handle.read(4))
        if version != 1:
            raise ValueError(f"unsupported shard version {version}")
        (count,) = struct.unpack("<I", handle.read(4))
        windows = []
        for _ in range(count):
            input_ids = list(struct.unpack(f"<{CONTEXT}i", handle.read(CONTEXT * 4)))
            loss_mask = list(struct.unpack(f"<{CONTEXT}B", handle.read(CONTEXT)))
            segment_ids = list(struct.unpack(f"<{CONTEXT}i", handle.read(CONTEXT * 4)))
            (byte_count,) = struct.unpack("<I", handle.read(4))
            windows.append(
                {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                    "segment_ids": segment_ids,
                    "byte_count": byte_count,
                }
            )
        return windows
