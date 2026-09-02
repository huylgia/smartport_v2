"""Nguồn camera: ai vào muxer, ai chỉ ghi hình, và các chốt chặn dễ mất khi sửa."""

from __future__ import annotations

from typing import Any

import pytest

from common.config import CameraConfig, load_crane
from common.enum import CameraRole
from ds_app.src.pipeline.elements import DEC_QUEUE, NVURISRCBIN, SOURCE_QUEUE
from ds_app.src.pipeline.sources import (
    build_sources,
    make_source_bin,
    replace_source,
    source_queue_name,
)
from tests.conftest import GC03

# Không cần env: định danh luồng nằm trong chính config, nên `load_crane` cho đúng mã
# production mà không phải dựng lại URL nào.
ENV: dict[str, str] = {}


def _cam(role: CameraRole, index: int = 1, **kw: Any) -> CameraConfig:
    return CameraConfig(role=role, index=index, desc="x", stream="rtsp://h:1/s", **kw)


# ---------------------------------------------------------------- tập nguồn


def all_muxers(gst: Any) -> dict[Any, Any]:
    """Một muxer cho mỗi role chạy model — hình dạng mà `run.py` dựng ra."""
    return {
        r: gst.ElementFactory.make("nvstreammux", f"mux_{r.value}")
        for r in CameraRole
        if r.runs_model
    }


def flat(attached: dict[Any, dict[int, Any]]) -> dict[tuple[Any, int], str]:
    """Gộp `{role: {pad: camera}}` thành `{(role, pad): code}` cho các phép đếm."""
    return {(r, i): c.code for r, pads in attached.items() for i, c in pads.items()}


def test_every_camera_gets_a_source_bin(gst: Any) -> None:
    """Cả camera chỉ-ghi cũng cần nvurisrcbin — nó cấp tee cho nhánh ghi."""
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    build_sources(gst, pipeline, crane, all_muxers(gst))

    bins = [c for c in pipeline.children if c.name.startswith("source-bin-")]
    assert len(bins) == 10, "phải có nguồn cho MỌI camera, kể cả camera không decode"


def test_only_decoding_cameras_reach_the_muxer(gst: Any) -> None:
    """8/10 vào muxer. Đây là ngân sách NVDEC, không phải tối ưu — xem HARDWARE_BUDGET §2.2."""
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    pad_map = flat(build_sources(gst, pipeline, crane, all_muxers(gst)))

    assert len(pad_map) == 8
    assert len(set(pad_map.values())) == 8
    codes = {c.code for c in crane.record_cameras if not c.decodes}
    assert not (codes & set(pad_map.values())), "camera chỉ-ghi không được decode"


def test_pad_index_is_contiguous_from_zero(gst: Any) -> None:
    """Camera bị bỏ qua KHÔNG được để lại lỗ hổng trong chỉ số pad.

    Probe ánh xạ khung → camera bằng ``pad_index``; một lỗ hổng làm lệch toàn bộ ánh xạ và
    mọi phát hiện bị gán nhầm camera — sai im lặng, không có lỗi nào.
    """
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    pad_map = flat(build_sources(gst, pipeline, crane, all_muxers(gst)))

    for role in {r for r, _ in pad_map}:
        pads = sorted(i for r, i in pad_map if r == role)
        assert pads == list(range(len(pads))), f"{role}: chỉ số pad có lỗ hổng"


def test_pad_map_follows_declaration_order(gst: Any) -> None:
    """Thứ tự khai báo trong YAML = thứ tự pad. Đổi YAML là đổi ánh xạ khung→camera."""
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    pad_map = flat(build_sources(gst, pipeline, crane, all_muxers(gst)))

    # Trong MỖI role, thứ tự pad theo thứ tự khai báo trong YAML.
    for role in {r for r, _ in pad_map}:
        got = [pad_map[(role, i)] for i in sorted(i for r, i in pad_map if r == role)]
        assert got == [c.code for c in crane.model_cameras if c.role is role]


# ---------------------------------------------------------------- không decimate


def test_no_drop_frame_interval_is_set(gst: Any) -> None:
    """⚠️ CỐ Ý không đặt. Nó KHÔNG giảm tải NVDEC.

    Nguồn là IPPP, GOP 50, không một khung B nào — mọi khung đều phải giải mã, và
    ``drop-frame-interval`` chỉ vứt output SAU khi đã giải mã xong. Đặt nó là trả giá
    (mất khung cho suy luận) mà không được gì ở chỗ nghẽn thật. HARDWARE_BUDGET §2.2.
    """
    src = make_source_bin(gst, 0, _cam(CameraRole.CCODE)).get_by_name("uridecode_0")
    assert "drop-frame-interval" not in src.props


# ---------------------------------------------------------------- chốt chặn


def test_jitterbuffer_does_not_drop_late_packets(gst: Any) -> None:
    """DeepStream mặc định TRUE, và nó vứt gói trễ ở phía TRƯỚC tee.

    Trước tee nghĩa là mất cả bản ghi hình lẫn dữ liệu suy luận — bằng chứng biến mất vì
    mạng chậm.
    """
    src = make_source_bin(gst, 0, _cam(CameraRole.CCODE)).get_by_name("uridecode_0")
    assert src.props["drop-on-latency"] is False


def test_dec_queue_leaks_downstream_so_the_tee_never_stalls(gst: Any) -> None:
    """DeepStream mặc định 0 (chặn). Chặn ở dec_que làm đứng tee dùng chung ⇒ đứng nhánh ghi."""
    src = make_source_bin(gst, 0, _cam(CameraRole.CCODE)).get_by_name("uridecode_0")
    assert src.props["leaky"] == 2
    assert NVURISRCBIN["leaky"] == 2


def test_rtsp_uses_tcp(gst: Any) -> None:
    """UDP mất gói thành vệt nhiễu kéo tới keyframe kế tiếp, cách nhau ~1,7 s."""
    src = make_source_bin(gst, 0, _cam(CameraRole.CCODE)).get_by_name("uridecode_0")
    assert src.props["select-rtp-protocol"] == 4


def test_reconnect_never_gives_up(gst: Any) -> None:
    """Đếm thất bại LIÊN TIẾP, nên giới hạn hữu hạn = camera chết vĩnh viễn sau N lần interval."""
    src = make_source_bin(gst, 0, _cam(CameraRole.CCODE)).get_by_name("uridecode_0")
    assert src.props["rtsp-reconnect-attempts"] == -1


def test_dec_queue_bounds_are_grafted_when_it_appears(gst: Any) -> None:
    """nvurisrcbin để max-size-time ở mặc định gst là MỘT GIÂY — đủ chặn tee dùng chung."""
    bin_ = make_source_bin(gst, 0, _cam(CameraRole.CCODE))

    dec_que = gst.ElementFactory.make("queue", "dec_que")
    bin_.emit("deep-element-added", None, dec_que)

    assert dec_que.props["max-size-time"] == DEC_QUEUE["max-size-time"] == 4_000_000_000
    assert dec_que.props["max-size-bytes"] == DEC_QUEUE["max-size-bytes"]


def test_source_queue_decouples_from_the_shared_muxer(gst: Any) -> None:
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    build_sources(gst, pipeline, crane, all_muxers(gst))

    q = pipeline.get_by_name(source_queue_name(0))
    assert q.props["max-size-buffers"] == SOURCE_QUEUE["max-size-buffers"]
    assert q.props["leaky"] == 2


def test_eos_is_dropped_at_the_muxer_pad(gst: Any) -> None:
    """Một camera rớt mạng KHÔNG được kéo sập muxer dùng chung.

    Nguồn live không "hết" hợp lệ bao giờ: EOS nghĩa là camera chết. Để nó đi tiếp thì
    nvstreammux coi cả batch là xong và mọi camera còn lại chết theo.
    """
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    muxers = all_muxers(gst)
    build_sources(gst, pipeline, crane, muxers)

    # Camera đầu của GC03 là `ccode`, nên pad 0 của muxer ccode là pad của nó.
    pad = muxers[CameraRole.CCODE].request_pad_simple("sink_0")
    assert pad.probes, "phải có probe chặn EOS"
    kind, callback = pad.probes[0]
    assert kind == gst.PadProbeType.EVENT_DOWNSTREAM

    class _Info:
        @staticmethod
        def get_event() -> Any:
            return type("E", (), {"type": gst.EventType.EOS})()

    assert callback(pad, _Info()) == gst.PadProbeReturn.DROP


def test_non_eos_events_pass_through(gst: Any) -> None:
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    muxers = all_muxers(gst)
    build_sources(gst, pipeline, crane, muxers)

    _, callback = muxers[CameraRole.CCODE].request_pad_simple("sink_0").probes[0]

    class _Info:
        @staticmethod
        def get_event() -> Any:
            return type("E", (), {"type": "CAPS"})()

    assert callback(None, _Info()) == gst.PadProbeReturn.OK


# ---------------------------------------------------------------- thay nguồn


def test_replace_source_keeps_the_muxer_pad(gst: Any) -> None:
    """⚠️ Chỉ thay cái bin. Trả request pad của streammux lúc chạy là thao tác làm treo pipeline."""
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    build_sources(gst, pipeline, crane, all_muxers(gst))

    queue_before = pipeline.get_by_name(source_queue_name(0))
    pad_before = queue_before.get_static_pad("sink")

    fresh = replace_source(gst, pipeline, 0, crane.record_cameras[0])

    assert fresh is not pipeline.get_by_name("khong-co")
    assert pipeline.get_by_name(source_queue_name(0)) is queue_before, "queue phải được giữ nguyên"
    assert queue_before.get_static_pad("sink") is pad_before, "pad của muxer không được trả lại"


def test_replace_source_reattaches_recording(gst: Any) -> None:
    """Bin cũ mang theo nhánh ghi của nó khi biến mất — phải ghép lại, nếu không camera đó
    im lặng ngừng được ghi hình trong khi mọi thứ khác trông vẫn bình thường."""
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")

    attached: list[str] = []
    recorder = type("R", (), {"attach": lambda _s, _g, _b, cam: attached.append(cam)})()

    build_sources(gst, pipeline, crane, all_muxers(gst), recorder=recorder)
    attached.clear()
    replace_source(gst, pipeline, 0, crane.record_cameras[0], recorder=recorder)

    assert attached == [crane.record_cameras[0].code]


def test_replace_missing_source_is_an_error(gst: Any) -> None:
    pipeline = gst.Bin.new("p")
    with pytest.raises(RuntimeError, match="không có gì để thay"):
        replace_source(gst, pipeline, 3, _cam(CameraRole.CCODE))


def test_missing_plugin_names_the_element(gst: Any) -> None:
    """Element hỏng trả None — im lặng. Lỗi phải nói rõ thiếu plugin nào."""
    from ds_app.src.pipeline.elements import make

    with pytest.raises(RuntimeError, match="khong-ton-tai"):
        make(gst, "khong-ton-tai", "x")


def test_replace_source_refuses_when_the_queue_name_does_not_match(gst: Any) -> None:
    """⚠️ Hồi quy đo được: ``detect.py`` đặt tên hàng đợi kiểu khác, nên ``replace_source``
    không tìm thấy gì và **lặng lẽ bỏ bước nối**. Nguồn mới chạy, pipeline không báo lỗi,
    và camera đó im hẳn — đo được 65 khung trước khi dựng lại, 0 sau đó.

    Nơi gọi là watchdog, nên "không làm gì" là kết quả tệ nhất có thể: nó biến một camera
    rớt tạm thời thành một camera chết tới lần khởi động lại process.
    """
    crane = load_crane(GC03, env=ENV)
    camera = crane.record_cameras[0]
    pipeline = gst.Bin.new("p")
    pipeline.add(make_source_bin(gst, 0, camera))

    with pytest.raises(RuntimeError, match="source_queue_name"):
        replace_source(gst, pipeline, 0, camera)


def test_the_queue_is_named_by_bin_index_not_pad_index(gst: Any) -> None:
    """⚠️ Hai chỉ số này lệch nhau ngay khi có camera KHÔNG decode nằm TRƯỚC một camera có
    decode, và ``replace_source(index)`` tra hàng đợi theo chỉ số **bin**.

    GC03 hiện đặt hai camera chỉ-ghi ở cuối nên chưa lệch — đó là may, không phải thiết kế.
    Test dựng đúng tình huống sẽ lệch: bỏ camera đầu ra khỏi mọi muxer.
    """
    crane = load_crane(GC03, env=ENV)
    pipeline = gst.Bin.new("p")
    # Không cấp muxer cho `ccode` ⇒ 5 camera đầu không vào muxer, nên camera bin thứ 5
    # (tcode1) nhận pad 0. Tên hàng đợi phải theo 5, không theo 0.
    muxers = {
        r: gst.ElementFactory.make("nvstreammux", f"mux_{r.value}") for r in (CameraRole.TCODE,)
    }

    build_sources(gst, pipeline, crane, muxers)

    assert pipeline.get_by_name(source_queue_name(5)) is not None, (
        "hàng đợi phải mang chỉ số BIN (5) — replace_source(5) tra theo số này"
    )
    assert pipeline.get_by_name(source_queue_name(0)) is None, (
        "chỉ số PAD (0) không được dùng làm tên: camera bin 0 đâu có vào muxer"
    )
