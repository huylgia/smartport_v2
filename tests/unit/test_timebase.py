from __future__ import annotations

import pytest

from internal.pkg.timebase import (
    FrameClock,
    TimeWindow,
    frame_timestamp,
    restore_frame_id,
)

# ---------------------------------------------------------------- restore_frame_id


@pytest.mark.parametrize("interval", [0, 1])
def test_restore_frame_id_no_decimation(interval: int) -> None:
    """DeepStream: 0 và 1 đều nghĩa là giữ mọi khung."""
    assert restore_frame_id(100, interval) == 100


@pytest.mark.parametrize(
    ("frame_num", "interval", "expected"),
    [
        (10, 6, 60),  # ccode:  30/6  = 5 fps
        (10, 9, 90),  # crane:  30/9  = 3.3 fps
        (10, 15, 150),  # tcode:  30/15 = 2 fps
        (0, 6, 0),
    ],
)
def test_restore_frame_id_scales_by_interval(frame_num: int, interval: int, expected: int) -> None:
    """Ngữ nghĩa DeepStream: N = chu kỳ GIỮ, nên chỉ số gốc = frame_num * N.

    Nhân (N+1) là sai — đó là cách hiểu "N khung bị bỏ giữa hai khung được giữ".
    """
    assert restore_frame_id(frame_num, interval) == expected


@pytest.mark.parametrize(("frame_num", "interval"), [(-1, 0), (0, -1), (0, 31)])
def test_restore_frame_id_rejects_out_of_range(frame_num: int, interval: int) -> None:
    with pytest.raises(ValueError):
        restore_frame_id(frame_num, interval)


# ---------------------------------------------------------------- frame_timestamp


def test_frame_timestamp_is_linear_in_frame_id() -> None:
    start = 1_756_300_000.0
    assert frame_timestamp(start, 0, 10.0) == start
    assert frame_timestamp(start, 10, 10.0) == start + 1.0
    assert frame_timestamp(start, 25, 10.0) == start + 2.5


def test_frame_timestamp_is_deterministic() -> None:
    """Cùng đầu vào phải cho cùng đầu ra — điều kiện để golden test so sánh được."""
    args = (1_756_300_000.0, 12_345, 10.0)
    assert frame_timestamp(*args) == frame_timestamp(*args)


@pytest.mark.parametrize(("fps", "frame_id"), [(0.0, 1), (-1.0, 1), (10.0, -1)])
def test_frame_timestamp_rejects_invalid(fps: float, frame_id: int) -> None:
    with pytest.raises(ValueError):
        frame_timestamp(1.0, frame_id, fps)


# ---------------------------------------------------------------- FrameClock


def test_frame_clock_effective_fps() -> None:
    """Nguồn thật của smartport là 30 fps (đo 2026-08-29, cả 10 camera)."""
    assert FrameClock(0.0, 30.0, 0).effective_fps == 30.0
    assert FrameClock(0.0, 30.0, 1).effective_fps == 30.0
    assert FrameClock(0.0, 30.0, 6).effective_fps == 5.0  # ccode
    assert FrameClock(0.0, 30.0, 15).effective_fps == 2.0  # tcode
    assert round(FrameClock(0.0, 30.0, 9).effective_fps, 2) == 3.33  # crane


def test_frame_clock_restores_decimated_frame_numbers() -> None:
    """Đây là cái bẫy: nếu quên khôi phục, trục thời gian co lại theo tỉ lệ decimate."""
    clock = FrameClock(start_ts=1000.0, fps=30.0, drop_frame_interval=6)

    # DeepStream báo frame_num=50; frame gốc là 300 ⇒ 10 s kể từ start.
    assert clock.timestamp(50) == 1010.0

    # Nếu nói rõ chỉ số đã là gốc thì không nhân nữa.
    assert clock.timestamp(50, decimated=False) == pytest.approx(1001.6667)


def test_frame_clock_frame_at_inverts_timestamp() -> None:
    clock = FrameClock(start_ts=1000.0, fps=30.0)
    for frame_id in (0, 1, 37, 12_345):
        ts = clock.timestamp(frame_id, decimated=False)
        assert clock.frame_at(ts) == frame_id


def test_frame_clock_frame_at_never_negative() -> None:
    clock = FrameClock(start_ts=1000.0, fps=30.0)
    assert clock.frame_at(900.0) == 0


@pytest.mark.parametrize(("fps", "interval"), [(0.0, 0), (-1.0, 0), (10.0, -1), (10.0, 31)])
def test_frame_clock_rejects_invalid(fps: float, interval: int) -> None:
    with pytest.raises(ValueError):
        FrameClock(start_ts=0.0, fps=fps, drop_frame_interval=interval)


# ---------------------------------------------------------------- TimeWindow


def test_time_window_around_anchor() -> None:
    """Offset thực địa: -20/+15 cho camera thường, -35/+10 cho camera đáy."""
    anchor = 1_756_312_837.0

    normal = TimeWindow.around(anchor, (-20.0, 15.0))
    assert normal.start == anchor - 20.0
    assert normal.end == anchor + 15.0
    assert normal.duration == 35.0

    bottom = TimeWindow.around(anchor, (-35.0, 10.0))
    assert bottom.duration == 45.0


def test_time_window_contains() -> None:
    w = TimeWindow(100.0, 200.0)
    assert w.contains(100.0)
    assert w.contains(150.0)
    assert w.contains(200.0)
    assert not w.contains(99.9)
    assert not w.contains(200.1)


def test_time_window_overlaps_is_symmetric() -> None:
    a = TimeWindow(100.0, 200.0)
    for other, expected in [
        (TimeWindow(150.0, 250.0), True),
        (TimeWindow(200.0, 300.0), True),  # chạm biên vẫn tính là giao
        (TimeWindow(201.0, 300.0), False),
        (TimeWindow(0.0, 99.0), False),
        (TimeWindow(120.0, 130.0), True),  # nằm gọn bên trong
    ]:
        assert a.overlaps(other) is expected
        assert other.overlaps(a) is expected


def test_time_window_shifted() -> None:
    assert TimeWindow(100.0, 200.0).shifted(50.0) == TimeWindow(150.0, 250.0)


def test_time_window_clamped_to_available_segments() -> None:
    w = TimeWindow(100.0, 200.0)
    assert w.clamped(120.0, 180.0) == TimeWindow(120.0, 180.0)
    assert w.clamped(0.0, 1000.0) == w


def test_time_window_clamped_raises_when_disjoint() -> None:
    w = TimeWindow(100.0, 200.0)
    with pytest.raises(ValueError, match="không giao"):
        w.clamped(300.0, 400.0)


def test_time_window_rejects_inverted() -> None:
    with pytest.raises(ValueError):
        TimeWindow(200.0, 100.0)


def test_time_window_is_hashable() -> None:
    """frozen+slots ⇒ dùng được làm key khi gom job evidence theo cửa sổ."""
    assert len({TimeWindow(1.0, 2.0), TimeWindow(1.0, 2.0)}) == 1
