"""Trục thời gian chung và sổ tra đoạn.

Cái mà module này chặn là một lỗi **không có triệu chứng**: hai nhánh đóng dấu bằng hai
đồng hồ khác nhau, mọi thứ trông vẫn chạy, chỉ có cửa sổ cắt clip lệch dần.
"""

from __future__ import annotations

import pytest

from ds_app.src.pipeline.timesync import Fragment, FragmentIndex, TimeBase, TimeSync

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


# ---------------------------------------------------------------- tra đoạn


def _index(
    cam: str = "1", *, starts: tuple[float, ...] = (), duration: float = 10.0
) -> FragmentIndex:
    idx = FragmentIndex()
    for i, s in enumerate(starts):
        idx.open_fragment(cam, f"/rec/{cam}/seg{i}.mp4", s, duration)
    return idx


def test_resolve_finds_the_fragment_containing_the_moment() -> None:
    idx = _index(starts=(1000.0, 1010.0, 1020.0))
    frag, _ = idx.resolve("1", 1015.0)

    assert frag is not None
    assert frag.path.endswith("seg1.mp4")


def test_resolve_does_not_just_return_the_newest() -> None:
    """⚠️ Nhánh ghi có thể chậm hơn nhánh model cả một đoạn.

    Lấy "đoạn mới nhất" là câu trả lời sai trong đúng tình huống hay xảy ra nhất.
    """
    idx = _index(starts=(1000.0, 1010.0, 1020.0, 1030.0))
    frag, _ = idx.resolve("1", 1005.0)

    assert frag is not None
    assert frag.path.endswith("seg0.mp4"), "phải là đoạn CHỨA khoảnh khắc, không phải mới nhất"


def test_moment_before_any_fragment_is_unresolved() -> None:
    idx = _index(starts=(1000.0,))
    frag, confident = idx.resolve("1", 999.0)

    assert frag is None
    assert not confident


def test_unknown_camera_is_unresolved() -> None:
    frag, confident = _index(starts=(1000.0,)).resolve("khong-co", 1005.0)
    assert frag is None and not confident


# ---------------------------------------------------------------- mức tin cậy


def test_a_bracketed_moment_is_confident() -> None:
    """Có đoạn mở SAU khoảnh khắc ⇒ đoạn chứa nó đã đóng, mp4 có moov đầy đủ."""
    idx = _index(starts=(1000.0, 1010.0))
    _, confident = idx.resolve("1", 1005.0)
    assert confident


def test_the_open_fragment_is_not_confident_when_closure_required() -> None:
    """``evidenced`` cần file đọc được từ đầu tới cuối, nên nó đòi đoạn đã đóng."""
    idx = _index(starts=(1000.0,))
    frag, confident = idx.resolve("1", 1005.0, require_closed=True)

    assert frag is not None, "vẫn biết là đoạn nào"
    assert not confident, "nhưng chưa chắc chắn vì đoạn còn mở"


def test_within_the_learned_window_is_confident_enough() -> None:
    idx = _index(starts=(1000.0,), duration=10.0)
    _, confident = idx.resolve("1", 1002.0)
    assert confident, "còn xa điểm cuối dự kiến"


def test_near_the_end_of_an_open_fragment_is_not_confident() -> None:
    """Sát điểm cuối của một đoạn CHƯA đóng: nó có thể đã kết thúc sớm hơn dự kiến."""
    idx = _index(starts=(1000.0,), duration=10.0)
    _, confident = idx.resolve("1", 1009.5)
    assert not confident


# ---------------------------------------------------------------- độ dài THẬT


def test_real_duration_is_learned_from_consecutive_starts() -> None:
    """⚠️ splitmuxsink cắt tại keyframe, nên độ dài thật KHÁC giá trị cấu hình."""
    idx = FragmentIndex()
    idx.open_fragment("1", "a.mp4", 1000.0, 10.0)
    idx.open_fragment("1", "b.mp4", 1011.7, 10.0)  # thật ra 11,7 s
    idx.open_fragment("1", "c.mp4", 1023.4, 10.0)

    assert idx.observed_duration("1") == pytest.approx(11.7)


def test_shortest_observed_duration_wins() -> None:
    """Lấy đoạn NGẮN NHẤT: ai chờ hết một đoạn mà lấy số dài hơn sẽ chờ thiếu."""
    idx = FragmentIndex()
    for start in (1000.0, 1012.0, 1020.0, 1032.0):
        idx.open_fragment("1", "x.mp4", start, 10.0)

    assert idx.observed_duration("1") == pytest.approx(8.0)


def test_duration_falls_back_to_nominal_before_learning() -> None:
    """Đoạn đầu tiên chưa có gì để học — dùng giá trị cấu hình trừ vùng đệm."""
    idx = FragmentIndex()
    idx.open_fragment("1", "a.mp4", 1000.0, 10.0)

    assert 0 < idx.observed_duration("1") < 10.0


def test_previous_fragment_end_is_finalised_by_the_next_start() -> None:
    idx = FragmentIndex()
    idx.open_fragment("1", "a.mp4", 1000.0, 10.0)
    idx.open_fragment("1", "b.mp4", 1011.7, 10.0)

    frag, _ = idx.resolve("1", 1005.0)
    assert frag is not None
    assert frag.end_unix == pytest.approx(1011.7), "điểm cuối chốt bằng mốc mở của đoạn sau"


# ---------------------------------------------------------------- bộ nhớ


def test_history_is_bounded() -> None:
    """Chạy 24/7: giữ hết đoạn thì bộ nhớ phình vô hạn."""
    idx = FragmentIndex(max_history=5)
    for i in range(50):
        idx.open_fragment("1", f"{i}.mp4", 1000.0 + i * 10, 10.0)

    frag, _ = idx.resolve("1", 1495.0)
    assert frag is not None
    assert idx.latest("1").path == "49.mp4"  # type: ignore[union-attr]


def test_fragment_is_frozen() -> None:
    frag = Fragment(path="a", start_unix=1.0, end_unix=2.0)
    with pytest.raises(Exception):  # noqa: B017
        frag.path = "b"  # type: ignore[misc]


def test_frame_unix_is_the_capture_time_not_the_file_time() -> None:
    """Đồng hồ trên clip bằng chứng phải là thời điểm CHỤP (DN-015).

    Đo trên 6 đoạn thật: ``birthtime`` lệch **+2 s** (độ trễ jitterbuffer + hàng đợi),
    ``mtime`` lệch **+32 s** (lúc đóng file). Chỉ mốc mở đoạn nằm đúng trục PTS đã neo.
    """
    frag = Fragment(path="/rec/CAM/1788279527.mp4", start_unix=1788279527.0, end_unix=1788279557.0)

    assert frag.frame_unix(0.0) == 1788279527.0, "khung đầu = mốc mở đoạn"
    assert frag.frame_unix(12.5) == 1788279539.5
    # Tên file là epoch NGUYÊN giây; mốc chính xác giữ phần thập phân ở đây.
    assert frag.frame_unix(0.25) == 1788279527.25
