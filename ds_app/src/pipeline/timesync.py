"""Một trục thời gian dùng chung cho nhánh ghi và nhánh model.

Vấn đề nó giải: hai nhánh đóng dấu thời gian bằng **hai đồng hồ khác nhau** thì chúng trôi
khỏi nhau, và không có gì báo lỗi — chỉ là cửa sổ cắt clip lệch dần.

* Nhánh ghi đóng dấu lúc mở đoạn. Nếu dùng đồng hồ tường, dấu đó gộp cả độ trễ hàng đợi.
* Nhánh model đóng dấu khung theo PTS (xem ``internal/pkg/timebase.py``), một trục đều.

Cách làm: neo ``PTS → unix`` **một lần cho mỗi camera** (:class:`TimeSync`), rồi cả hai
nhánh quy đổi qua cùng cái neo đó. Chênh lệch tuyệt đối so với giờ thật không quan trọng;
điều quan trọng là hai bên **nhất quán với nhau**.

Sổ tra cứu đoạn ở ``internal/pkg/fragments.py`` — nó dùng chung với ``evidenced``, còn neo
PTS thì chỉ pipeline mới cần.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

__all__ = ["TimeBase", "TimeSync"]


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
