from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

_TYPE_SIZES = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def read_gguf_architecture(path: Path) -> str | None:
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            return None

        handle.seek(8)
        _tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)

        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            if key == "general.architecture" and value_type == 8:
                return _read_string(handle)
            _skip_value(handle, value_type)

    return None


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", handle.read(4))[0]


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    return handle.read(length).decode("utf-8", errors="replace")


def _skip_value(handle: BinaryIO, value_type: int) -> None:
    if value_type in _TYPE_SIZES:
        handle.seek(_TYPE_SIZES[value_type], 1)
        return

    if value_type == 8:
        length = _read_u64(handle)
        handle.seek(length, 1)
        return

    if value_type == 9:
        item_type = _read_u32(handle)
        item_count = _read_u64(handle)
        if item_type in _TYPE_SIZES:
            handle.seek(_TYPE_SIZES[item_type] * item_count, 1)
            return
        if item_type == 8:
            for _ in range(item_count):
                length = _read_u64(handle)
                handle.seek(length, 1)
            return

    raise ValueError(f"Unsupported GGUF metadata type: {value_type}")
