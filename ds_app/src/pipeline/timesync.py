"""Một trục thời gian dùng chung cho nhánh ghi và nhánh model, và sổ tra cứu đoạn.

Vấn đề nó giải: hai nhánh đóng dấu thời gian bằng **hai đồng hồ khác nhau** thì chúng trôi
khỏi nhau, và không có gì báo lỗi — chỉ là cửa sổ cắt clip lệch dần.

* Nhánh ghi đóng dấu lúc mở đoạn. Nếu dùng đồng hồ tường, dấu đó gộp cả độ trễ hàng đợi.
* Nhánh model đóng dấu khung theo PTS (xem ``internal/pkg/timebase.py``), một trục đều.

``evidenced`` hỏi "khoảnh khắc T nằm trong đoạn nào" bằng cách so hai dấu đó. Chúng phải
nằm trên **cùng một trục**, nếu không phép so là vô nghĩa.

Cách làm: neo ``PTS → unix`` **một lần cho mỗi camera** (:class:`TimeSync`), rồi cả hai
nhánh quy đổi qua cùng cái neo đó. Chênh lệch tuyệt đối so với giờ thật không quan trọng;
điều quan trọng là hai bên **nhất quán với nhau**.

⚠️ **Độ dài đoạn phải HỌC, không được giả định.** ``splitmuxsink`` chỉ cắt tại keyframe nên
độ dài thật là bội số của GOP — và GOP đo được **dao động** trên chính camera cảng (1,00 s
và 1,67 s ở hai lần chạy khác nhau, 2026-08-30). Ai chờ hết một đoạn phải lấy số từ
:meth:`FragmentIndex.observed_duration`, không lấy từ config.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

__all__ = ["Fragment", "FragmentIndex", "TimeBase", "TimeSync"]


@dataclass(frozen=True, slots=True)
class TimeBase:
    """Neo ``PTS → unix`` của một camera, bất biến.

    Bất biến là cố ý: neo đổi giữa chừng nghĩa là mọi dấu thời gian trước và sau đó nằm
    trên hai trục khác nhau, và không cách nào biết dấu nào thuộc trục nào.
    """

    base_unix: float
    """Thời điểm unix ứng với :attr:`first_pts_sec`."""

    first_pts_sec: float
    """PTS của khung đầu tiên thấy được, tính bằng giây."""

    def to_unix(self, pts_sec: float) -> float:
        """Quy đổi một PTS bất kỳ sang unix trên trục đã neo."""
        return self.base_unix + (pts_sec - self.first_pts_sec)


class TimeSync:
    """Neo thời gian theo từng camera. **Ghi lần đầu thắng.**

    Ai chạm vào khung/buffer trước sẽ đặt neo — có thể là nhánh ghi, có thể là probe. Cái
    nào không quan trọng, miễn là **chỉ một** neo tồn tại cho mỗi camera.

    Đọc không cần khoá: dict chứa dataclass đông cứng và ``dict.get`` là nguyên tử.
    """

    def __init__(self) -> None:
        self._bases: dict[str, TimeBase] = {}
        self._lock = threading.Lock()

    def get(self, camera_code: str) -> TimeBase | None:
        return self._bases.get(camera_code)

    def anchor(self, camera_code: str, pts_sec: float, now_unix: float) -> TimeBase | None:
        """Lấy neo của camera, đặt nó từ khung này nếu chưa có.

        Trả ``None`` khi chưa neo được — PTS không hợp lệ. Nơi gọi phải chịu được điều đó:
        vài buffer đầu của nguồn RTSP có thể không có PTS.
        """
        existing = self._bases.get(camera_code)
        if existing is not None:
            return existing
        if pts_sec <= 0 or now_unix <= 0:
            return None
        with self._lock:
            # Kiểm lại trong khoá: hai luồng có thể cùng tới đây.
            found = self._bases.get(camera_code)
            if found is not None:
                return found
            base = TimeBase(base_unix=now_unix, first_pts_sec=pts_sec)
            self._bases[camera_code] = base
            return base


@dataclass(frozen=True, slots=True)
class Fragment:
    """Một đoạn ghi hình: đường dẫn và cửa sổ ``[start, end)`` trên trục đã neo.

    ``end_unix`` là **tạm tính** cho tới khi đoạn kế tiếp mở ra — lúc đó nó được chốt lại
    bằng mốc mở của đoạn sau, tức độ dài THẬT.
    """

    path: str
    start_unix: float
    end_unix: float


_GUARD_SEC = 1.0
"""Vùng đệm trước điểm cuối (có thể ngắn hơn dự kiến) của một đoạn chưa chốt."""


class FragmentIndex:
    """Sổ đoạn ghi hình theo từng camera, để trả lời "khoảnh khắc này nằm ở đoạn nào".

    ⚠️ Phải tra theo **cửa sổ**, không phải "lấy đoạn mới nhất": nhánh ghi có thể chậm hơn
    nhánh model cả một đoạn hoặc hơn, nên đoạn mới nhất thường **không** phải đoạn chứa
    khung đang xét.
    """

    def __init__(self, max_history: int = 128) -> None:
        self._frags: dict[str, list[Fragment]] = {}
        self._shortest: dict[str, float] = {}
        self._nominal: float = 0.0
        self._max_history = max_history
        self._lock = threading.Lock()

    def open_fragment(
        self, camera_code: str, path: str, start_unix: float, duration_sec: float
    ) -> None:
        """Ghi nhận một đoạn vừa mở. Gọi từ callback ``format-location-full``."""
        with self._lock:
            if self._nominal <= 0 < duration_sec:
                self._nominal = max(duration_sec - _GUARD_SEC, _GUARD_SEC)

            frags = self._frags.setdefault(camera_code, [])
            if frags and start_unix > frags[-1].start_unix:
                # Đoạn trước kết thúc đúng lúc đoạn này bắt đầu — đây là độ dài THẬT của nó.
                prev = frags[-1]
                frags[-1] = Fragment(prev.path, prev.start_unix, start_unix)
                real = start_unix - prev.start_unix
                if real > 0:
                    self._shortest[camera_code] = min(self._shortest.get(camera_code, real), real)

            frags.append(Fragment(path, start_unix, start_unix + duration_sec))
            if len(frags) > self._max_history:
                del frags[: len(frags) - self._max_history]

    def resolve(
        self, camera_code: str, frame_unix: float, *, require_closed: bool = False
    ) -> tuple[Fragment | None, bool]:
        """Đoạn chứa ``frame_unix``, kèm mức tin cậy.

        Args:
            require_closed: Chỉ coi là chắc chắn khi đoạn đã **đóng** (có đoạn sau mở ra),
                tức file mp4 đã có ``moov`` đầy đủ và đọc được từ đầu tới cuối.

        Returns:
            ``(đoạn, chắc_chắn)``. ``chắc_chắn=False`` nghĩa là câu trả lời có thể đúng
            nhưng chưa kiểm được — nơi gọi nên chờ thay vì cắt clip từ nó.
        """
        with self._lock:
            frags = self._frags.get(camera_code)
            if not frags:
                return None, False

            chosen: Fragment | None = None
            closed = False
            for frag in frags:
                if frag.start_unix <= frame_unix:
                    chosen = frag
                else:
                    closed = True  # có đoạn mở SAU khoảnh khắc này ⇒ đoạn chứa nó đã đóng
                    break

            if chosen is None:
                return None, False
            if require_closed:
                return chosen, closed

            window = self._shortest.get(camera_code, self._nominal)
            return chosen, closed or (window > 0 and frame_unix < chosen.start_unix + window)

    def observed_duration(self, camera_code: str) -> float:
        """Độ dài đoạn **thật**, học từ hai mốc mở liên tiếp. ``0`` = chưa học được.

        ⚠️ Dùng số này, không dùng ``max-size-time`` trong config: ``splitmuxsink`` chỉ cắt
        tại keyframe nên độ dài thật là bội số của GOP, và GOP của camera cảng **không cố
        định** (đo được 1,00 s và 1,67 s ở hai lần chạy). Ai chờ hết một đoạn mà lấy số từ
        config sẽ chờ thiếu.
        """
        with self._lock:
            return self._shortest.get(camera_code, self._nominal)

    def latest(self, camera_code: str) -> Fragment | None:
        with self._lock:
            frags = self._frags.get(camera_code)
            return frags[-1] if frags else None
