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


# ---------------------------------------------------------------- PTS đứt quãng


def test_a_backwards_pts_re_anchors_and_is_counted() -> None:
    """⚠️ Nguồn RTSP nối lại có thể phát PTS từ đầu. Neo cũ áp lên PTS mới cho ra dấu thời
    gian ở **quá khứ** — clip evidence bị cắt ở chỗ chưa xảy ra chuyện gì, và không có gì
    báo. Neo lại, và đếm.
    """
    sync = TimeSync()
    sync.anchor("cam", pts_sec=10.0, now_unix=1000.0)
    sync.anchor("cam", pts_sec=11.0, now_unix=1001.0)

    base = sync.anchor("cam", pts_sec=0.5, now_unix=1030.0)

    assert base is not None
    assert base.to_unix(0.5) == 1030.0, "khung này phải là BÂY GIỜ, không phải quá khứ"
    assert sync.resets["cam"] == 1


def test_a_long_forward_gap_keeps_the_anchor() -> None:
    """Camera mất mạng rồi có lại: PTS tiến đúng bằng thời gian mất. Đo được 30 s mất mạng
    cho PTS tiến 30 s. Neo lại ở đây sẽ xoá mất chính thông tin đó."""
    sync = TimeSync()
    first = sync.anchor("cam", pts_sec=10.0, now_unix=1000.0)

    after = sync.anchor("cam", pts_sec=40.3, now_unix=1030.3)

    assert after is first
    assert after is not None
    assert after.to_unix(40.3) == pytest.approx(1030.3)
    assert "cam" not in sync.resets


def test_the_comparison_is_against_the_previous_frame_not_the_first() -> None:
    """Nếu so với ``first_pts_sec`` thì sau một đợt mất mạng dài, PTS vẫn lớn hơn mốc đầu
    rất nhiều — và một cú lùi thật sẽ không bao giờ bị phát hiện."""
    sync = TimeSync()
    sync.anchor("cam", pts_sec=10.0, now_unix=1000.0)
    sync.anchor("cam", pts_sec=500.0, now_unix=1490.0)

    sync.anchor("cam", pts_sec=100.0, now_unix=1500.0)

    assert sync.resets["cam"] == 1


def test_cameras_do_not_share_a_reset() -> None:
    sync = TimeSync()
    sync.anchor("a", pts_sec=10.0, now_unix=1000.0)
    sync.anchor("b", pts_sec=10.0, now_unix=1000.0)

    sync.anchor("a", pts_sec=1.0, now_unix=1030.0)

    assert sync.resets == {"a": 1}
