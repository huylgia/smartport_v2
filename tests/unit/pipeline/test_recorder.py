"""Nhánh ghi hình: cắm trước decoder, và các bẫy đã trả giá để tìm ra.

Mỗi test ở đây khoá lại một lỗi đã gặp thật. Chúng trông vụn vặt cho tới khi ai đó "dọn"
một dòng và mất cả bản ghi bằng chứng.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ds_app.src.pipeline.elements import MUXER, RECORD_QUEUE, SPLITMUX
from ds_app.src.pipeline.recorder import RecordingBranch


def _ready_source_bin(gst: Any) -> Any:
    """Source bin đã dựng xong phần bên trong, như ``nvurisrcbin`` sau khi khởi động."""
    bin_ = gst.Bin.new("source-bin-00")
    for name, factory in (
        ("tee_rtsp_pre_decode", "tee"),
        ("parser", "h265parse"),
        ("src_cap_filter_nvvidconv", "capsfilter"),
    ):
        bin_.add(gst.ElementFactory.make(factory, name))
    return bin_


def _attach(gst: Any, tmp_path: Path, **kw: Any) -> tuple[RecordingBranch, Any]:
    branch = RecordingBranch(tmp_path, **kw)
    bin_ = _ready_source_bin(gst)
    branch.attach(gst, bin_, "1")
    return branch, bin_


# ---------------------------------------------------------------- ghép vào


def test_taps_the_internal_tee_not_a_new_one(gst: Any, tmp_path: Path) -> None:
    """⚠️ Không tự dựng tee.

    Tự ghép ``rtspsrc ! depay ! h265parse ! tee`` chết ở negotiate: nvv4l2decoder đòi
    ``byte-stream`` còn mp4mux đòi ``hvc1`` — một h265parse chỉ thương lượng được một, và
    lỗi hiện ra là ``not-negotiated (-4)`` không nhắc gì tới caps.
    """
    _, bin_ = _attach(gst, tmp_path)

    tee = bin_.get_by_name("tee_rtsp_pre_decode")
    requested = [p for p in tee._pads.values() if p.peer is not None]
    assert requested, "phải cắm vào tee CÓ SẴN"
    assert not [c for c in bin_.children if c.factory == "tee" and c.name != "tee_rtsp_pre_decode"]


def test_waits_for_internals_when_not_ready_yet(gst: Any, tmp_path: Path) -> None:
    """nvurisrcbin dựng phần bên trong BẤT ĐỒNG BỘ — ghép ngay là không thấy gì."""
    branch = RecordingBranch(tmp_path)
    bin_ = gst.Bin.new("source-bin-00")

    branch.attach(gst, bin_, "1")
    assert not [c for c in bin_.children if c.factory == "splitmuxsink"], "chưa được ghép vội"

    for name, factory in (("tee_rtsp_pre_decode", "tee"), ("parser", "h265parse")):
        bin_.add(gst.ElementFactory.make(factory, name))
    marker = gst.ElementFactory.make("capsfilter", "src_cap_filter_nvvidconv")
    bin_.add(marker)
    bin_.emit("deep-element-added", None, marker)

    assert bin_.get_by_name("splitmuxsink_1") is not None


def test_missing_internals_is_a_clear_error(gst: Any, tmp_path: Path) -> None:
    """Đổi tên element bên trong giữa các bản DeepStream phải nổ rõ ràng, không im lặng."""
    branch = RecordingBranch(tmp_path)
    bin_ = gst.Bin.new("source-bin-00")
    bin_.add(gst.ElementFactory.make("capsfilter", "src_cap_filter_nvvidconv"))

    with pytest.raises(RuntimeError, match="tee_rtsp_pre_decode"):
        branch.attach(gst, bin_, "1")


def test_record_parser_matches_the_source_codec(gst: Any, tmp_path: Path) -> None:
    """Nguồn có thể là H.264 hoặc H.265 — lấy đúng họ parser mà nhánh decode đang dùng."""
    branch = RecordingBranch(tmp_path)
    bin_ = gst.Bin.new("source-bin-00")
    bin_.add(gst.ElementFactory.make("tee", "tee_rtsp_pre_decode"))
    bin_.add(gst.ElementFactory.make("h264parse", "parser"))
    bin_.add(gst.ElementFactory.make("capsfilter", "src_cap_filter_nvvidconv"))

    branch.attach(gst, bin_, "1")
    assert bin_.get_by_name("parse_record_1").factory == "h264parse"


# ---------------------------------------------------------------- bẫy đã trả giá


def test_config_interval_reinserts_parameter_sets(gst: Any, tmp_path: Path) -> None:
    """⚠️ ``config-interval=-1`` bắt buộc: chèn lại VPS/SPS/PPS trước mỗi keyframe.

    Không có nó thì từng đoạn không tự đứng được — file ghi ra nhưng ``evidenced`` không
    mở nổi.
    """
    _, bin_ = _attach(gst, tmp_path)
    assert bin_.get_by_name("parse_record_1").props["config-interval"] == -1


def test_muxer_is_configured_by_factory_not_element(gst: Any, tmp_path: Path) -> None:
    """⚠️ Đặt thuộc tính ``muxer`` bằng element cho ra fragment 0 byte, không báo lỗi."""
    _, bin_ = _attach(gst, tmp_path)
    sink = bin_.get_by_name("splitmuxsink_1")

    assert sink.props["muxer-factory"] == MUXER["factory"]
    assert "muxer" not in sink.props, "không được đặt thuộc tính `muxer`"


def test_robust_muxing_sets_the_update_period_explicitly(gst: Any, tmp_path: Path) -> None:
    """``use-robust-muxing`` MỘT MÌNH không làm gì — phải đặt reserved-moov-update-period."""
    _, bin_ = _attach(gst, tmp_path)
    sink = bin_.get_by_name("splitmuxsink_1")

    assert sink.props["use-robust-muxing"] is True
    assert "reserved-moov-update-period" in sink.props["muxer-properties"]
    assert "reserved-max-duration" in sink.props["muxer-properties"]


def test_record_queue_leaks_rather_than_blocks(gst: Any, tmp_path: Path) -> None:
    """Tee đẩy TUẦN TỰ: nhánh ghi chặn là nhánh model đứng theo."""
    _, bin_ = _attach(gst, tmp_path)
    q = bin_.get_by_name("queue_record_1")

    assert q.props["leaky"] == RECORD_QUEUE["leaky"] == 2
    assert q.props["max-size-time"] == RECORD_QUEUE["max-size-time"]
    assert q.props["max-size-buffers"] == 0, "chặn theo thời gian, không theo số buffer"


def test_segment_length_is_configured(gst: Any, tmp_path: Path) -> None:
    _, bin_ = _attach(gst, tmp_path)
    assert bin_.get_by_name("splitmuxsink_1").props["max-size-time"] == SPLITMUX["max-size-time"]


# ---------------------------------------------------------------- buffer thiếu PTS


def _probe_of(bin_: Any) -> Any:
    return bin_.get_by_name("parse_record_1").get_static_pad("src").probes[0][1]


class _Buffer:
    def __init__(self, pts: int, keyframe: bool = False) -> None:
        self.pts = pts
        self.mini_object = type("M", (), {"flags": 0 if keyframe else 1})()


def test_pts_less_buffers_are_dropped(gst: Any, tmp_path: Path) -> None:
    """⚠️ Nguồn RTSP thỉnh thoảng đẩy buffer không có PTS. Tới mp4mux là nó **abort** —
    mất cả đoạn đang ghi, không chỉ một khung."""
    _, bin_ = _attach(gst, tmp_path)

    info = type("I", (), {"get_buffer": staticmethod(lambda: _Buffer(gst.CLOCK_TIME_NONE))})()
    assert _probe_of(bin_)(None, info, "1") == gst.PadProbeReturn.DROP


def test_buffers_with_pts_pass(gst: Any, tmp_path: Path) -> None:
    _, bin_ = _attach(gst, tmp_path)

    info = type("I", (), {"get_buffer": staticmethod(lambda: _Buffer(1_000_000_000))})()
    assert _probe_of(bin_)(None, info, "1") == gst.PadProbeReturn.OK


def test_pts_warning_is_logged_once_per_camera(gst: Any, tmp_path: Path, capsys: Any) -> None:
    """Nguồn hỏng thì nó xảy ra liên tục — log mỗi buffer sẽ nhấn chìm mọi thứ khác."""
    _, bin_ = _attach(gst, tmp_path)
    probe = _probe_of(bin_)
    info = type("I", (), {"get_buffer": staticmethod(lambda: _Buffer(gst.CLOCK_TIME_NONE))})()

    for _ in range(5):
        probe(None, info, "1")

    assert capsys.readouterr().out.count("không có PTS") == 1


# ---------------------------------------------------------------- tên đoạn


def test_fragment_path_is_reported_to_the_caller(gst: Any, tmp_path: Path) -> None:
    """``evidenced`` biết đoạn nào chứa khoảnh khắc nào nhờ callback này.

    Thay cho việc quét thư mục rồi đoán theo tên file — cách đó đoán sai ở ranh giới hai
    đoạn, đúng chỗ hay cần cắt nhất.
    """
    seen: list[tuple[str, str, float]] = []
    _branch, bin_ = _attach(gst, tmp_path, on_fragment=lambda *a: seen.append(a))

    sink = bin_.get_by_name("splitmuxsink_1")
    callback = sink.signals["format-location-full"][0][0]
    path = callback(sink, 0, None, "1")

    assert path.endswith(".mp4")
    assert str(tmp_path / "1") in path, "mỗi camera một thư mục con"
    assert len(seen) == 1 and seen[0][1] == path


def test_output_directory_is_created_per_camera(gst: Any, tmp_path: Path) -> None:
    _attach(gst, tmp_path)
    assert (tmp_path / "1").is_dir()
