"""Trục thời gian chung: neo ``PTS → unix`` một lần cho mỗi camera.

Cái mà module này chặn là một lỗi **không có triệu chứng**: hai nhánh đóng dấu bằng hai
đồng hồ khác nhau, mọi thứ trông vẫn chạy, chỉ có cửa sổ cắt clip lệch dần.
"""

from __future__ import annotations

import pytest

from ds_app.src.pipeline.timesync import TimeBase, TimeSync

# ---------------------------------------------------------------- neo PTS→unix


def test_anchor_maps_pts_to_unix() -> None:
    sync = TimeSync()
    base = sync.anchor("1", pts_sec=100.0, now_unix=1_700_000_000.0)

    assert base is not None
    assert base.to_unix(100.0) == 1_700_000_000.0
    assert base.to_unix(110.0) == 1_700_000_010.0, "10 s PTS ⇒ 10 s unix"


def test_first_anchor_wins() -> None:
    """Neo đổi giữa chừng ⇒ dấu trước và sau nằm trên hai trục, không cách nào phân biệt."""
    sync = TimeSync()
    first = sync.anchor("1", pts_sec=100.0, now_unix=1_700_000_000.0)
    second = sync.anchor("1", pts_sec=500.0, now_unix=1_700_009_999.0)

    assert second is first


def test_cameras_anchor_independently() -> None:
    """Mỗi camera có đồng hồ PTS riêng — chung neo là sai lệch bằng đúng khoảng cách hai PTS."""
    sync = TimeSync()
    sync.anchor("1", pts_sec=100.0, now_unix=1_700_000_000.0)
    sync.anchor("4", pts_sec=7.0, now_unix=1_700_000_000.0)

    assert sync.get("1") != sync.get("4")
    assert sync.get("4").to_unix(17.0) == 1_700_000_010.0  # type: ignore[union-attr]


def test_invalid_pts_does_not_anchor() -> None:
    """Vài buffer đầu của RTSP có thể không có PTS — nơi gọi phải chịu được ``None``."""
    sync = TimeSync()
    assert sync.anchor("1", pts_sec=0.0, now_unix=1_700_000_000.0) is None
    assert sync.get("1") is None


def test_anchor_survives_a_later_valid_frame() -> None:
    sync = TimeSync()
    sync.anchor("1", pts_sec=-1.0, now_unix=1_700_000_000.0)
    base = sync.anchor("1", pts_sec=5.0, now_unix=1_700_000_050.0)

    assert base is not None
    assert base.to_unix(5.0) == 1_700_000_050.0


def test_time_base_is_frozen() -> None:
    base = TimeBase(base_unix=1.0, first_pts_sec=2.0)
    with pytest.raises(Exception):  # noqa: B017 — dataclass đông cứng
        base.base_unix = 9.0  # type: ignore[misc]
