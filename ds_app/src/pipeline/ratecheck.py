"""Đo nhịp THẬT của nguồn lúc chạy, và đối chiếu với con số đã khai.

Vì sao phải đo thay vì đọc: **không đường nào khác nói cho ta biết.**

* ``nvv4l2decoder.drop-frame-interval`` khai ``changeable only in NULL or READY state``,
  nên nhịp phải quyết TRƯỚC khi khung đầu tiên về — không thể chờ rồi mới tính.
* Caps của nguồn khai ``framerate=0/1`` trên **cả 10 camera GC03** (đo 2026-09-02), tức
  "biến thiên, tự đo lấy". Đối chiếu với caps là đối chiếu với một chỗ trống.

Nên ``source_fps`` vẫn phải khai trong config. Cái làm được ở đây là **bắt quả tang khi
con số đó sai**: camera ``..._1517`` chạy 18 fps trong khi config khai 30, và nó chỉ lộ ra
sau 30 phút chạy khi có người để ý bảng tổng kết. Với lớp này nó lộ trong vài giây.

Phép đo::

    fps = drop_frame_interval / trung_vị(Δ frame_ts giữa hai khung liên tiếp)

``frame_ts`` lấy từ PTS nên nó là nhịp nguồn thật, không phải nhịp sau khi hàng đợi bỏ bớt.
Dùng **trung vị**, không dùng trung bình: một khoảng trống do mất mạng sẽ kéo trung bình đi
rất xa trong khi trung vị không nhúc nhích.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass

__all__ = ["RateCheck", "SourceRate"]

MIN_SAMPLES = 8
"""Số khoảng tối thiểu trước khi dám kết luận. Dưới ngưỡng này, vài khung đầu của nguồn
RTSP còn đang ổn định nhịp và số đo chưa nói lên gì."""

WINDOW = 64
"""Số khoảng gần nhất giữ lại. Đủ để trung vị ổn định, đủ ngắn để một camera đổi nhịp giữa
chừng vẫn lộ ra thay vì bị pha loãng bởi lịch sử."""

TOLERANCE = 0.5
"""fps chênh bao nhiêu thì coi là khai sai. Nhịp nguồn dao động vài phần trăm là bình
thường; 0,5 fps đủ rộng để không báo nhiễu, đủ hẹp để bắt 18-vs-30."""


@dataclass
class SourceRate:
    """Nhịp đo được của một camera."""

    declared: float
    """``source_fps`` trong config — con số đang được dùng để tính decimate."""

    measured: float | None
    """Nhịp thật, hoặc ``None`` khi chưa đủ mẫu."""

    samples: int

    @property
    def mismatched(self) -> bool:
        return self.measured is not None and abs(self.measured - self.declared) > TOLERANCE


class RateCheck:
    """Gom khoảng thời gian giữa các khung, theo từng camera.

    An toàn luồng: probe của nhiều role chạy trên nhiều thread streaming khác nhau.
    """

    def __init__(self) -> None:
        self._gaps: dict[str, list[float]] = {}
        self._last: dict[str, tuple[int, float]] = {}
        self._declared: dict[str, float] = {}
        self._interval: dict[str, int] = {}
        self._lock = threading.Lock()
        self._warned: set[str] = set()

    def observe(
        self,
        camera_code: str,
        frame_id: int,
        frame_ts: float,
        *,
        declared_fps: float,
        drop_frame_interval: int,
    ) -> SourceRate | None:
        """Ghi nhận một khung. Trả về :class:`SourceRate` **lần đầu** phát hiện lệch.

        Chỉ trả một lần cho mỗi camera: nơi gọi in ra cảnh báo, và một cảnh báo lặp mỗi
        khung sẽ nhấn chìm mọi thứ khác trong log.
        """
        with self._lock:
            self._declared[camera_code] = declared_fps
            self._interval[camera_code] = max(1, drop_frame_interval)
            prev = self._last.get(camera_code)
            self._last[camera_code] = (frame_id, frame_ts)
            if prev is None:
                return None

            d_id, d_ts = frame_id - prev[0], frame_ts - prev[1]
            # Bỏ khoảng không hợp lệ: khung trùng, hoặc thời gian lùi sau khi neo lại.
            if d_id <= 0 or d_ts <= 0:
                return None
            # Chuẩn hoá về "giây cho mỗi khung NGUỒN" — chịu được cả khi một khung bị bỏ
            # giữa chừng làm `d_id` lớn hơn một bước decimate.
            gaps = self._gaps.setdefault(camera_code, [])
            gaps.append(d_ts / d_id)
            if len(gaps) > WINDOW:
                del gaps[0]

            rate = self._rate(camera_code)
            if rate is None or not rate.mismatched or camera_code in self._warned:
                return None
            self._warned.add(camera_code)
            return rate

    def rate(self, camera_code: str) -> SourceRate | None:
        with self._lock:
            return self._rate(camera_code)

    def report(self) -> list[tuple[str, SourceRate]]:
        with self._lock:
            out = []
            for code in sorted(self._gaps):
                r = self._rate(code)
                if r is not None:
                    out.append((code, r))
            return out

    def _rate(self, camera_code: str) -> SourceRate | None:
        """Gọi trong khoá."""
        gaps = self._gaps.get(camera_code, [])
        declared = self._declared.get(camera_code, 0.0)
        if len(gaps) < MIN_SAMPLES:
            return SourceRate(declared=declared, measured=None, samples=len(gaps))
        per_source_frame = statistics.median(gaps)
        if per_source_frame <= 0:
            return None
        return SourceRate(declared=declared, measured=1.0 / per_source_frame, samples=len(gaps))
