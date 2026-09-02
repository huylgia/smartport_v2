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
    """Neo thời gian theo từng camera. **Ghi lần đầu thắng — trừ khi PTS đứt.**

    Ai chạm vào khung/buffer trước sẽ đặt neo — có thể là nhánh ghi, có thể là probe. Cái
    nào không quan trọng, miễn là **một** neo phục vụ cả hai nhánh; đó là điều giữ cho cửa
    sổ cắt clip không lệch giữa chúng.

    ⚠️ Một ngoại lệ, và nó bắt buộc: **PTS lùi**. Nguồn RTSP nối lại có thể phát PTS từ
    đầu, và một neo cũ áp lên PTS mới cho ra dấu thời gian ở **quá khứ** — clip evidence sẽ
    được cắt ở chỗ chưa từng xảy ra chuyện gì, và không có gì báo. Gặp lùi thì neo lại, và
    ĐẾM (:attr:`resets`) để nó không im lặng.

    PTS nhảy **tiến** thì không neo lại: đó là camera mất mạng, chuyện thường, và đã đo —
    30 s mất mạng cho PTS tiến đúng 30 s (HARDWARE_BUDGET §6.1). Neo lại ở đó sẽ xoá mất
    thông tin thật.

    Đọc không cần khoá: dict chứa dataclass đông cứng và ``dict.get`` là nguyên tử.
    """

    def __init__(self) -> None:
        self._bases: dict[str, TimeBase] = {}
        self._last_pts: dict[str, float] = {}
        self._lock = threading.Lock()
        self.resets: dict[str, int] = {}
        """Số lần phải neo lại vì PTS lùi, theo từng camera. Khác 0 nghĩa là nguồn có đứt
        quãng — mọi dấu thời gian trước và sau nằm trên hai trục khác nhau."""

    def get(self, camera_code: str) -> TimeBase | None:
        return self._bases.get(camera_code)

    def anchor(self, camera_code: str, pts_sec: float, now_unix: float) -> TimeBase | None:
        """Lấy neo của camera, đặt nó từ khung này nếu chưa có hoặc nếu PTS đã lùi.

        Trả ``None`` khi chưa neo được — PTS không hợp lệ. Nơi gọi phải chịu được điều đó:
        vài buffer đầu của nguồn RTSP có thể không có PTS.
        """
        if pts_sec <= 0 or now_unix <= 0:
            return None

        existing = self._bases.get(camera_code)
        last = self._last_pts.get(camera_code)
        # So với PTS gần nhất, KHÔNG so với `first_pts_sec`: sau một đợt mất mạng dài thì
        # PTS vẫn lớn hơn mốc đầu rất nhiều, nên so với mốc đầu sẽ không bao giờ thấy đứt.
        went_backwards = last is not None and pts_sec < last

        if existing is not None and not went_backwards:
            self._last_pts[camera_code] = pts_sec
            return existing

        with self._lock:
            # Kiểm lại trong khoá: hai luồng có thể cùng tới đây.
            found = self._bases.get(camera_code)
            if found is not None and not went_backwards:
                self._last_pts[camera_code] = pts_sec
                return found
            if found is not None:
                self.resets[camera_code] = self.resets.get(camera_code, 0) + 1
            base = TimeBase(base_unix=now_unix, first_pts_sec=pts_sec)
            self._bases[camera_code] = base
            self._last_pts[camera_code] = pts_sec
            return base
