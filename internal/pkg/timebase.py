"""Trục thời gian suy từ frame.

Vì sao không dùng wall-clock (``time.time()`` tại điểm xử lý):

Đóng dấu thời gian bằng đồng hồ hệ thống *tại điểm xử lý* thì con số thu được gộp cả độ
trễ hàng đợi lẫn độ trễ inference, nên nó **dao động theo tải GPU**. Mọi thứ tính từ nó —
cửa sổ cắt clip, khoảng cách giữa hai container trong twin-lift, cửa sổ "cẩu đang thao
tác" — đều bị nhiễu theo, và nhiễu mạnh nhất đúng lúc hệ thống bận nhất.

Thay bằng::

    startTime + frame_id / fps

Đây là một trục *đều*, suy ra từ chỉ số frame, hoàn toàn không phụ thuộc thời điểm xử lý.
Hai lần chạy lại cùng một đoạn video cho ra cùng một dấu thời gian — điều kiện cần để
golden test so sánh được.

Lưu ý về decimation: DeepStream bỏ frame ngay tại decoder bằng
``nvv4l2decoder.drop-frame-interval``, nên ``frame_meta.frame_num`` đếm theo frame *đã
qua lọc*. Phải khôi phục chỉ số thật bằng :func:`restore_frame_id` trước khi tính thời
gian, nếu không trục sẽ co lại theo đúng tỉ lệ decimate.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FrameClock",
    "frame_timestamp",
    "restore_frame_id",
]


def restore_frame_id(frame_num: int, drop_frame_interval: int) -> int:
    """Khôi phục chỉ số frame gốc từ chỉ số sau khi decimate.

    Ngữ nghĩa của DeepStream, trích nguyên văn ``gst-inspect-1.0 nvv4l2decoder``::

        drop-frame-interval : Interval to drop the frames, ex: value of 5 means
                              every 5th frame will be given by decoder, rest all dropped
                              Unsigned Integer. Range: 0 - 30  Default: 0

    Tức là **N là chu kỳ giữ**, không phải số khung bị bỏ giữa hai khung được giữ:
    ``fps_ra = fps_nguồn / N``, và chỉ số gốc ``= frame_num * N``. ``0`` và ``1`` đều
    nghĩa là giữ mọi khung.

    Quy ước này khớp với chính ``nvv4l2decoder``: giá trị đặt cho thuộc tính
    ``drop-frame-interval`` cũng chính là số nhân để khôi phục chỉ số gốc. Nhờ vậy chỉ có
    **một** con số đi từ config xuống decoder rồi ngược lên đây — không có phép ``±1`` nào
    ở giữa để làm sai.

    Args:
        frame_num: Chỉ số frame do DeepStream cung cấp (đã qua decimate).
        drop_frame_interval: Giá trị đặt cho decoder, ``0``-``30``. ``0``/``1`` = không bỏ.

    Raises:
        ValueError: nếu ``frame_num`` âm, hoặc ``drop_frame_interval`` ngoài ``0``-``30``.
    """
    if frame_num < 0:
        raise ValueError(f"frame_num phải >= 0, nhận {frame_num}")
    if not 0 <= drop_frame_interval <= 30:
        raise ValueError(f"drop_frame_interval phải trong 0..30, nhận {drop_frame_interval}")
    return frame_num * max(1, drop_frame_interval)


def frame_timestamp(start_ts: float, frame_id: int, fps: float) -> float:
    """Thời điểm của một frame trên trục đều.

    Args:
        start_ts: Mốc bắt đầu luồng, epoch giây.
        frame_id: Chỉ số frame **gốc** (đã qua :func:`restore_frame_id` nếu có decimate).
        fps: FPS của **nguồn**, không phải fps sau decimate.

    Raises:
        ValueError: nếu ``fps`` không dương hoặc ``frame_id`` âm.
    """
    if fps <= 0:
        raise ValueError(f"fps phải > 0, nhận {fps}")
    if frame_id < 0:
        raise ValueError(f"frame_id phải >= 0, nhận {frame_id}")
    return start_ts + frame_id / fps


@dataclass(frozen=True, slots=True)
class FrameClock:
    """Trục thời gian của một luồng camera.

    Gói ``start_ts`` + ``fps`` + ``drop_frame_interval`` lại để nơi gọi không phải mang
    ba tham số đi khắp nơi và không thể ghép nhầm fps của camera này với start_ts của
    camera khác.
    """

    start_ts: float
    fps: float
    drop_frame_interval: int = 0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps phải > 0, nhận {self.fps}")
        if not 0 <= self.drop_frame_interval <= 30:
            raise ValueError(
                f"drop_frame_interval phải trong 0..30, nhận {self.drop_frame_interval}"
            )

    @property
    def effective_fps(self) -> float:
        """FPS thực tế sau decimate — số frame mỗi giây mà rule thực sự nhìn thấy.

        ⚠️ Đây là fps *hạ nguồn*. Nó KHÔNG phải fps mà NVDEC phải giải mã: nguồn của
        smartport là HEVC cấu trúc IPPP, GOP 50, **không có khung B** (đo 2026-08-29),
        nên không có khung non-reference nào để bỏ qua — decoder vẫn phải giải mã đủ
        ``fps`` gốc rồi mới vứt output. Ngân sách NVDEC phải tính theo :attr:`fps`.
        Chi tiết: ``docs/HARDWARE_BUDGET.md`` §2.2.
        """
        return self.fps / max(1, self.drop_frame_interval)

    def timestamp(self, frame_num: int, *, decimated: bool = True) -> float:
        """Thời điểm của một frame.

        Args:
            frame_num: Chỉ số frame.
            decimated: ``True`` (mặc định) nếu ``frame_num`` đến từ DeepStream và đã qua
                decimate — sẽ khôi phục trước khi tính. Đặt ``False`` khi chỉ số đã là
                chỉ số gốc.
        """
        frame_id = restore_frame_id(frame_num, self.drop_frame_interval) if decimated else frame_num
        return frame_timestamp(self.start_ts, frame_id, self.fps)

    def frame_at(self, ts: float) -> int:
        """Chỉ số frame **gốc** gần nhất với thời điểm ``ts``. Nghịch đảo của :meth:`timestamp`."""
        return max(0, round((ts - self.start_ts) * self.fps))
