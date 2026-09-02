"""Đọc số khung và thời lượng từ chính bảng mẫu của file MP4.

Vì sao tự đọc thay vì gọi ``ffprobe``/``gst-discoverer``: cái ta cần chỉ là hai số nguyên
nằm ở vị trí cố định trong hai box, và cả hai công cụ kia đều **không có trong image
ds_app**. Thêm ffmpeg vào image chỉ để đếm khung là thêm vài trăm MB và một bộ giải mã nữa
phải vá khi có CVE.

Đây là phép đo nhịp nguồn **chính xác nhất** có được, vì đoạn ghi là passthrough: nó chứa
đúng bitstream đã tới, không qua decode, không qua decimate. Chính nó đã bắt được camera
``..._1517`` chạy 18 fps trong khi config khai 30 (HARDWARE_BUDGET §6.3).

Chỉ đọc đoạn **đã đóng**: file đang ghi có ``moov`` cập nhật mỗi giây nhưng bảng mẫu chưa
đầy đủ, và đếm nó cho ra nhịp thấp giả.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Mp4Info", "read_mp4_info"]


@dataclass(frozen=True, slots=True)
class Mp4Info:
    """Số khung và thời lượng của một đoạn."""

    frames: int
    duration_sec: float

    @property
    def fps(self) -> float:
        """``0.0`` nếu thời lượng bằng 0 — nơi gọi phải loại trước khi dùng làm mẫu."""
        return self.frames / self.duration_sec if self.duration_sec > 0 else 0.0


def _boxes(data: bytes, start: int, end: int):  # type: ignore[no-untyped-def]
    """Duyệt các box ở MỘT cấp. Trả ``(loại, đầu_nội_dung, cuối_box)``."""
    pos = start
    while pos + 8 <= end:
        (size,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        if size == 0:  # box cuối, kéo tới hết
            size = end - pos
        if size < 8 or pos + size > end:
            return
        yield kind, pos + 8, pos + size
        pos += size


def _find(data: bytes, start: int, end: int, path: tuple[bytes, ...]) -> tuple[int, int] | None:
    """Tìm box theo đường dẫn lồng nhau, ví dụ ``(b"moov", b"mvhd")``."""
    for kind, body, stop in _boxes(data, start, end):
        if kind != path[0]:
            continue
        if len(path) == 1:
            return body, stop
        found = _find(data, body, stop, path[1:])
        if found is not None:
            return found
    return None


def read_mp4_info(path: Path) -> Mp4Info | None:
    """Số khung + thời lượng, hoặc ``None`` nếu file không đọc được như MP4.

    Trả ``None`` chứ không ném: nơi gọi duyệt cả thư mục, và một file dở dang không đáng
    làm hỏng phép đo của chín camera còn lại.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    head = _find(raw, 0, len(raw), (b"moov", b"mvhd"))
    table = _find(raw, 0, len(raw), (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsz"))
    if head is None or table is None:
        return None

    # mvhd phiên bản 0: version(1) flags(3) ctime(4) mtime(4) timescale(4) duration(4)
    if raw[head[0]] != 0:
        # Phiên bản 1 dùng 64-bit cho ctime/mtime/duration; chưa gặp với mp4mux nên
        # không đoán — trả None để nơi gọi biết là chưa đọc được.
        return None
    timescale, duration = struct.unpack(">II", raw[head[0] + 12 : head[0] + 20])
    # stsz: version(1) flags(3) sample_size(4) sample_count(4)
    (frames,) = struct.unpack(">I", raw[table[0] + 8 : table[0] + 12])

    if timescale <= 0:
        return None
    return Mp4Info(frames=frames, duration_sec=duration / timescale)
