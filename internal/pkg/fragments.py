"""Sổ tra cứu đoạn ghi hình, và kế hoạch cắt clip từ nhiều đoạn.

Ở ``internal/pkg/`` chứ không ở ``ds_app/``: **hai process khác nhau cùng cần nó**. ds_app
dựng sổ khi ghi hình; ``evidenced`` dựng lại sổ **từ đĩa** khi cắt clip — nó là service
Python thuần (PyInstaller + systemd), không đọc được bộ nhớ của container DeepStream, và
không được import ngược vào cây ``ds_app/``.

Thuần: không I/O ngoài liệt kê thư mục, không GStreamer, không phụ thuộc gì của pipeline.

⚠️ **Tên file CHÍNH LÀ chỉ mục.** Tên là epoch nguyên giây lúc mở đoạn, nên `evidenced`
tra được chỉ bằng ``camera_code`` (tên thư mục) và một khoảng thời gian tuyệt đối. Đó là
lý do định dạng tên được chốt là con số chứ không phải chuỗi giờ: chuỗi giờ buộc phải phân
tích ngược, mà phân tích ngược thì phải đoán múi giờ — đoán sai một lần là lệch cả giờ.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ClipPiece", "Fragment", "FragmentIndex"]


@dataclass(frozen=True, slots=True)
class Fragment:
    """Một đoạn ghi hình: đường dẫn và cửa sổ ``[start, end)`` trên trục đã neo.

    ``end_unix`` là **tạm tính** cho tới khi đoạn kế tiếp mở ra — lúc đó nó được chốt lại
    bằng mốc mở của đoạn sau, tức độ dài THẬT.
    """

    path: str
    """Đường dẫn **cuối cùng**. Có thể chưa tồn tại nếu đoạn còn đang ghi — xem
    :attr:`live_path`."""

    start_unix: float

    end_unix: float
    """Tới lúc đoạn SAU mở ra — dùng để **tra cứu** ("khoảnh khắc này thuộc đoạn nào").

    ⚠️ KHÔNG phải chỗ nội dung kết thúc. Nếu đoạn kế bị mất (camera rớt, đoạn bị dọn) thì
    trường này kéo dài qua cả khoảng trống — dùng nó để cắt clip sẽ xin ffmpeg một khoảng
    thời gian không có trong file. Dùng :attr:`content_end` cho việc đó."""

    nominal_sec: float = 0.0
    """Độ dài đoạn theo cấu hình. ``0`` = không biết (đoạn dựng tay trong test)."""

    _SLACK_RATIO = 1.2
    _SLACK_SEC = 2.0
    """Đoạn thật dài hơn cấu hình vì ``splitmuxsink`` chỉ cắt tại keyframe: độ dài thật là
    ``ceil(giới hạn / GOP) lần GOP``, tức dôi **tối đa một GOP**.

    Lấy cái CHẶT hơn trong hai biên: 20 % đúng cho đoạn ngắn, ``+2 s`` đúng cho đoạn dài
    (giới hạn 30 s, GOP 1,67 s ⇒ dôi 1,67 s, không phải 6 s).

    ⚠️ Đây là **xấp xỉ**, nên mốc bắt đầu của lỗ hổng là một *cận dưới*: lỗ hổng thật có
    thể rộng hơn. Muốn chính xác thì phải lấy thời điểm đóng thật từ message
    ``splitmuxsink-fragment-closed`` (nó mang ``running-time``) — việc của Phase 7 khi
    ``evidenced`` cần độ chính xác đó."""

    @property
    def content_end(self) -> float:
        """Chỗ nội dung thật sự kết thúc — dùng khi CẮT clip.

        Khác :attr:`end_unix` đúng ở chỗ có lỗ hổng. Không biết ``nominal_sec`` thì đành
        tin ``end_unix``.
        """
        if self.nominal_sec <= 0:
            return self.end_unix
        slack = min(self.nominal_sec * self._SLACK_RATIO, self.nominal_sec + self._SLACK_SEC)
        return min(self.end_unix, self.start_unix + slack)

    def frame_unix(self, pts_offset_sec: float) -> float:
        """Thời điểm CHỤP của một khung, từ độ lệch PTS trong đoạn.

        Đây là nguồn giờ duy nhất đúng cho đồng hồ vẽ lên clip bằng chứng (DN-015).

        ⚠️ Đừng dùng ``birthtime``/``mtime`` của file: đo được chúng lệch **+2 s** và
        **+32 s** so với thời điểm chụp — birthtime gộp độ trễ jitterbuffer và hàng đợi,
        mtime là lúc đóng file. Và tuyệt đối đừng dùng ``datetime.now()`` lúc vẽ: job
        evidence chạy sau sự kiện 20-40 giây.
        """
        return self.start_unix + pts_offset_sec

    @property
    def live_path(self) -> str:
        """Nơi đọc khi đoạn **chưa chốt**: cùng tên nhưng có đuôi ``.part``.

        Đoạn đang ghi vẫn đọc được (``reserved-moov-update-period`` làm mới ``moov`` mỗi
        giây) và ``evidenced`` thường cần chính nó — cửa sổ bằng chứng hay chạm vào đoạn
        hiện tại. Nên đừng chờ đoạn đóng: thử :attr:`path`, không có thì đọc cái này."""
        return self.path + ".part"


@dataclass(frozen=True, slots=True)
class ClipPiece:
    """Một lát cắt từ **một** đoạn, để ghép thành clip bằng chứng.

    Cửa sổ bằng chứng rộng hơn một đoạn (``[-35 s, +10 s]`` = 45 s, đoạn 30 s), nên clip
    **luôn** ghép từ 2-3 đoạn. Mỗi lát mang theo mốc tuyệt đối của chính nó.

    ⚠️ Đây là lý do đồng hồ phải vẽ **trước** khi ghép, theo từng lát. Tính giờ trên clip
    đã ghép (``clip_start + n/fps``) là sai hai lần: đoạn không dài đều nhau (đo được
    30,00 s và 28,47 s), và ``ffmpeg concat`` đặt lại PTS về 0. Xem DN-015.
    """

    fragment: Fragment
    start_offset: float
    """Giây tính từ đầu đoạn — đưa thẳng vào ``ffmpeg -ss``."""

    end_offset: float
    """Giây tính từ đầu đoạn — đưa thẳng vào ``ffmpeg -to``."""

    @property
    def start_unix(self) -> float:
        """Mốc tuyệt đối của khung đầu lát này. Gốc cho đồng hồ vẽ lên nó."""
        return self.fragment.start_unix + self.start_offset

    @property
    def duration(self) -> float:
        return self.end_offset - self.start_offset


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

    @classmethod
    def from_directory(
        cls, root: Path | str, camera_code: str, *, nominal_sec: float
    ) -> FragmentIndex:
        """Dựng lại sổ **từ đĩa**. Đây là cách ``evidenced`` tra đoạn.

        Args:
            root: Thư mục ghi hình; cấu trúc ``<root>/<camera_code>/<epoch>.mp4``.
            nominal_sec: Độ dài đoạn theo cấu hình — cần để biết chỗ nội dung kết thúc khi
                đoạn kế bị thiếu (xem :attr:`Fragment.content_end`).

        ``evidenced`` là process riêng, đóng gói riêng (PyInstaller + systemd), nên nó
        **không đọc được** sổ trong bộ nhớ của ds_app. Tên file là epoch nguyên giây lúc mở
        đoạn, nên chỉ cần liệt kê thư mục là dựng lại đủ.

        Bỏ qua ``.part``: đoạn chưa chốt vẫn đọc được, nhưng nó chưa có mốc kết thúc nên
        đưa vào sổ sẽ cho ra một ``content_end`` đoán mò. Cần nó thì dùng
        :attr:`Fragment.live_path` của đoạn cuối.

        Bỏ qua file tên không phải số: thư mục ghi hình có thể lẫn thứ khác, và một
        ``ValueError`` giữa lúc dựng bằng chứng thì tệ hơn là bỏ qua một file lạ.
        """
        index = cls()
        folder = Path(root) / camera_code
        if not folder.is_dir():
            return index

        starts: list[tuple[float, str]] = []
        for f in folder.glob("*.mp4"):
            try:
                starts.append((float(int(f.stem)), str(f)))
            except ValueError:
                continue
        for start, path in sorted(starts):
            index.open_fragment(camera_code, path, start, nominal_sec)
        return index

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
                frags[-1] = Fragment(prev.path, prev.start_unix, start_unix, prev.nominal_sec)
                real = start_unix - prev.start_unix
                if real > 0:
                    self._shortest[camera_code] = min(self._shortest.get(camera_code, real), real)

            frags.append(Fragment(path, start_unix, start_unix + duration_sec, duration_sec))
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

    def plan(
        self, camera_code: str, from_unix: float, to_unix: float
    ) -> tuple[list[ClipPiece], list[tuple[float, float]]]:
        """Các lát cần cắt để dựng clip cho cửa sổ ``[from_unix, to_unix)``, và các **lỗ hổng**.

        Returns:
            ``(lát, lỗ hổng)``. Lỗ hổng là những khoảng không đoạn nào phủ — đoạn đã bị
            dọn, hoặc camera mất kết nối lúc đó.

        ⚠️ Trả lỗ hổng ra chứ không lặng lẽ cắt ngắn clip. Một clip thiếu 8 giây ở giữa
        trông y hệt một clip bình thường, và người xem lại sự kiện sẽ tin nó đầy đủ. Nơi
        gọi phải quyết định: báo, hay dựng clip kèm ghi chú.
        """
        if to_unix <= from_unix:
            return [], []

        with self._lock:
            frags = list(self._frags.get(camera_code, ()))

        pieces: list[ClipPiece] = []
        gaps: list[tuple[float, float]] = []
        cursor = from_unix

        for frag in frags:
            # `content_end`, KHÔNG phải `end_unix`: cái sau kéo dài qua cả lỗ hổng.
            covers_to = frag.content_end
            if covers_to <= cursor or frag.start_unix >= to_unix:
                continue
            if frag.start_unix > cursor:
                gaps.append((cursor, frag.start_unix))
                cursor = frag.start_unix

            stop = min(covers_to, to_unix)
            pieces.append(
                ClipPiece(
                    fragment=frag,
                    start_offset=cursor - frag.start_unix,
                    end_offset=stop - frag.start_unix,
                )
            )
            cursor = stop
            if cursor >= to_unix:
                break

        if cursor < to_unix:
            gaps.append((cursor, to_unix))
        return pieces, gaps

    def latest(self, camera_code: str) -> Fragment | None:
        with self._lock:
            frags = self._frags.get(camera_code)
            return frags[-1] if frags else None
