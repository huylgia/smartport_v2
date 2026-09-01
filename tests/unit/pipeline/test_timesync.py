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


# ---------------------------------------------------------------- ghép nhiều đoạn


def _even_index(n: int = 6, *, start: float = 1000.0, seg: float = 30.0) -> FragmentIndex:
    ix = FragmentIndex()
    for i in range(n):
        ix.open_fragment("CAM", f"/rec/{int(start + i * seg)}.mp4", start + i * seg, seg)
    return ix


def test_a_window_inside_one_segment_needs_one_piece() -> None:
    pieces, gaps = _even_index().plan("CAM", 1005.0, 1015.0)
    assert len(pieces) == 1 and gaps == []
    assert (pieces[0].start_offset, pieces[0].end_offset) == (5.0, 15.0)


def test_an_evidence_window_spans_several_segments() -> None:
    """Cửa sổ bằng chứng (45 s) rộng hơn một đoạn (30 s) ⇒ clip LUÔN ghép từ nhiều đoạn.

    Đây là lý do đồng hồ phải vẽ theo TỪNG lát: mỗi lát có mốc tuyệt đối riêng.
    """
    pieces, gaps = _even_index().plan("CAM", 1045.0, 1090.0)

    assert gaps == []
    assert len(pieces) == 2, "45 s cửa sổ, đoạn 30 s"
    assert [p.fragment.path for p in pieces] == ["/rec/1030.mp4", "/rec/1060.mp4"]
    assert pieces[0].start_unix == 1045.0, "mốc tuyệt đối của lát đầu"
    assert pieces[1].start_unix == 1060.0, "lát sau bắt đầu ở mốc mở đoạn của NÓ"
    assert sum(p.duration for p in pieces) == 45.0


def test_each_piece_carries_its_own_absolute_start() -> None:
    """⚠️ Tính giờ trên clip đã ghép là sai: `ffmpeg concat` đặt lại PTS về 0, và đoạn
    không dài đều nhau. Mốc phải lấy từ đoạn NGUỒN."""
    pieces, _ = _even_index().plan("CAM", 1025.0, 1095.0)
    for piece in pieces:
        assert piece.start_unix == piece.fragment.start_unix + piece.start_offset


def test_uneven_segments_do_not_shift_the_clock() -> None:
    """Đoạn thật dài không đều (đo được 30,00 s và 28,47 s) — mốc vẫn phải đúng."""
    ix = FragmentIndex()
    ix.open_fragment("CAM", "/rec/a.mp4", 1000.0, 30.0)
    ix.open_fragment("CAM", "/rec/b.mp4", 1028.47, 30.0)  # đoạn trước chỉ dài 28,47 s

    pieces, gaps = ix.plan("CAM", 1020.0, 1035.0)
    assert gaps == []
    assert pieces[0].start_unix == 1020.0
    assert pieces[1].start_unix == 1028.47, "không phải 1030 — đoạn trước ngắn hơn danh nghĩa"


def test_a_swept_segment_shows_up_as_a_gap() -> None:
    """⚠️ Lỗ hổng phải báo ra, không được lặng lẽ cắt ngắn clip.

    Clip thiếu 30 giây ở giữa trông y hệt clip bình thường, và người xem lại sự kiện sẽ
    tin nó đầy đủ.
    """
    ix = FragmentIndex()
    ix.open_fragment("CAM", "/rec/a.mp4", 1000.0, 30.0)
    ix.open_fragment("CAM", "/rec/c.mp4", 1060.0, 30.0)  # đoạn 1030 đã bị dọn

    pieces, gaps = ix.plan("CAM", 1010.0, 1070.0)
    assert [p.fragment.path for p in pieces] == ["/rec/a.mp4", "/rec/c.mp4"]

    # Mốc bắt đầu lỗ hổng là CẬN DƯỚI: ta chỉ biết đoạn a không dài quá cấu hình cộng một
    # GOP, không biết chính xác nó dừng ở đâu. Điều phải đúng là lỗ hổng **được báo** và
    # có độ rộng gần đúng.
    assert len(gaps) == 1
    start, end = gaps[0]
    assert end == 1060.0
    assert 1030.0 <= start <= 1032.0, f"lỗ hổng ~28-30 s, nhận {end - start:.1f} s"


def test_a_window_reaching_past_what_was_recorded_is_a_gap() -> None:
    """Xin đoạn cũ hơn thứ còn giữ (retention 3 phút) ⇒ lỗ hổng ở đầu."""
    ix = _even_index(n=2, start=1000.0)
    pieces, gaps = ix.plan("CAM", 940.0, 1030.0)
    assert gaps and gaps[0] == (940.0, 1000.0)
    assert pieces and pieces[0].start_unix == 1000.0


def test_an_empty_or_inverted_window_plans_nothing() -> None:
    ix = _even_index()
    assert ix.plan("CAM", 1050.0, 1050.0) == ([], [])
    assert ix.plan("CAM", 1050.0, 1040.0) == ([], [])
    assert ix.plan("KHONG_CO", 1000.0, 1100.0) == ([], [(1000.0, 1100.0)])
