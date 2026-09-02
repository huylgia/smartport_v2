"""Đếm khung từ bảng mẫu MP4 — phép đo nhịp nguồn chính xác nhất có được."""

from __future__ import annotations

import struct
from pathlib import Path

from internal.pkg.mp4probe import read_mp4_info


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body) + 8) + kind + body


def mp4(*, timescale: int = 1000, duration: int = 30_000, frames: int = 900) -> bytes:
    """MP4 tối thiểu chỉ có hai box mà phép đo cần."""
    mvhd = box(b"mvhd", bytes(4) + bytes(8) + struct.pack(">II", timescale, duration))
    stsz = box(b"stsz", bytes(4) + struct.pack(">II", 0, frames))
    stbl = box(b"stbl", stsz)
    minf = box(b"minf", stbl)
    mdia = box(b"mdia", minf)
    trak = box(b"trak", mdia)
    return box(b"ftyp", b"isom" + bytes(8)) + box(b"moov", mvhd + trak)


def test_it_reads_frames_and_duration(tmp_path: Path) -> None:
    f = tmp_path / "a.mp4"
    f.write_bytes(mp4(frames=900, timescale=1000, duration=30_000))

    info = read_mp4_info(f)

    assert info is not None
    assert (info.frames, info.duration_sec) == (900, 30.0)
    assert info.fps == 30.0


def test_it_finds_the_camera_that_runs_slower(tmp_path: Path) -> None:
    """⚠️ Đo thật: 3/10 camera GC03 không chạy 30 fps (18, 27, 24), trong khi config khai
    30 cho tất cả. Chính phép đếm này bắt được — đoạn ghi là passthrough nên nó chứa đúng
    bitstream đã tới."""
    f = tmp_path / "b.mp4"
    f.write_bytes(mp4(frames=500, timescale=1000, duration=27_780))

    info = read_mp4_info(f)

    assert info is not None
    assert round(info.fps, 2) == 18.0


def test_a_file_that_is_not_mp4_gives_none(tmp_path: Path) -> None:
    """Trả ``None`` chứ không ném: nơi gọi duyệt cả thư mục, và một file dở dang không
    đáng làm hỏng phép đo của chín camera còn lại."""
    f = tmp_path / "c.mp4"
    f.write_bytes(b"day khong phai mp4")

    assert read_mp4_info(f) is None


def test_a_missing_file_gives_none(tmp_path: Path) -> None:
    assert read_mp4_info(tmp_path / "khong-ton-tai.mp4") is None


def test_a_zero_duration_does_not_divide_by_zero(tmp_path: Path) -> None:
    f = tmp_path / "d.mp4"
    f.write_bytes(mp4(frames=10, duration=0))

    info = read_mp4_info(f)

    assert info is not None
    assert info.fps == 0.0
