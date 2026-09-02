"""Nguồn camera: một ``nvurisrcbin`` cho mỗi camera, ghép vào ``nvstreammux`` dùng chung.

Hai tập camera **khác nhau**, và đó là điểm cốt lõi của bài toán cảng:

* **Ghi hình: cả 10 camera.** Ảnh bằng chứng 6 mặt cần cả camera không chạy model.
* **Vào nhánh model: chỉ 8.** ``bottom`` và ``evidence_only`` không có model nào để chạy,
  và decode cả 10 luồng 2688x1520@30 vượt trần một NVDEC của GA106. Xem
  ``docs/HARDWARE_BUDGET.md`` §2.2.

Camera chỉ-ghi vẫn cần một ``nvurisrcbin`` (nó cấp ``tee_rtsp_pre_decode`` cho nhánh ghi),
nhưng **không** nối vào muxer — nên nhánh decode của nó không bao giờ có ai kéo dữ liệu.

``gi``/``Gst`` import muộn để module nạp được trên máy không có DeepStream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from common.config import CameraConfig, CraneConfig
from common.enum import CameraRole
from ds_app.src.pipeline.elements import (
    DEC_QUEUE,
    NVURISRCBIN,
    SOURCE_QUEUE,
    apply_props,
    link_pads,
    make,
)

__all__ = ["build_sources", "make_source_bin", "replace_source", "source_queue_name"]


def source_bin_name(index: int) -> str:
    return f"source-bin-{index:02d}"


def source_queue_name(index: int) -> str:
    """Tên hàng đợi giữa nguồn ``index`` và muxer.

    Một hàm chứ không phải một chuỗi f-string lặp lại: :func:`replace_source` **tìm hàng
    đợi theo tên** để nối lại nguồn mới. Hai chỗ tự đặt tên độc lập thì `replace_source`
    không tìm thấy gì — và bản đầu của nó lặng lẽ bỏ qua bước nối, nên camera dựng lại
    xong không bao giờ có khung nào nữa. Đo được: 65 khung trước restart, 0 sau đó.
    """
    return f"queue_source_{index:02d}"


def build_sources(
    Gst: Any,
    pipeline: Any,
    crane: CraneConfig,
    muxers: Mapping[CameraRole, Any],
    *,
    recorder: Any | None = None,
    watchdog: Any | None = None,
) -> dict[CameraRole, dict[int, CameraConfig]]:
    """Dựng nguồn cho **mọi** camera, một lần duy nhất cho cả hai nhánh.

    Đây là đường dựng nguồn **duy nhất**. Nhánh ghi và nhánh model dùng chung một
    ``nvurisrcbin`` cho mỗi camera: nhánh ghi cắm vào ``tee_rtsp_pre_decode`` bên trong nó
    (trước decode), nhánh model lấy src pad đã decode. Mở hai kết nối RTSP cho cùng một
    camera là gấp đôi số socket và gấp đôi tải mạng để lấy đúng một luồng.

    Args:
        muxers: ``{role: nvstreammux}``. Một muxer cho MỖI role — muxer cho ra một batch
            và không có cách nào bảo tầng sau chỉ xử lý vài nguồn trong đó. Role không có
            trong bảng thì camera của nó **không** nối vào đâu cả: đó là cách `bottom`,
            `evidence_only`, và các role chưa có model giữ NVDEC ở 0.

    Returns:
        ``{role: {pad_index: camera}}`` cho các camera có vào muxer. Probe cần bảng này:
        metadata của DeepStream chỉ mang ``pad_index``, không mang danh tính camera.
    """
    attached: dict[CameraRole, dict[int, CameraConfig]] = {}
    pad_index: dict[CameraRole, int] = {}

    # `record_cameras`, KHÔNG phải `crane.cameras`: cái sau là một ánh xạ, và duyệt nó
    # cho ra KHOÁ chứ không phải camera — im lặng, tới tận lúc chạm thuộc tính đầu tiên.
    for index, camera in enumerate(crane.record_cameras):
        bin_ = make_source_bin(Gst, index, camera)
        pipeline.add(bin_)

        if recorder is not None:
            recorder.attach(Gst, bin_, camera.code)

        muxer = muxers.get(camera.role) if camera.decodes else None
        if muxer is None:
            # Không nối vào muxer: camera này chỉ ghi hình. Nhánh decode bên trong
            # nvurisrcbin vẫn tồn tại nhưng không ai kéo, nên nó dừng ngay ở buffer đầu
            # (NOT_LINKED) và KHÔNG tốn NVDEC.
            #
            # Đã đo, không còn là suy đoán: 10 camera ghi hình với src pad không nối cho
            # NVDEC 0 %; nối một `fakesink` vào cho "gọn" đẩy lên 11,6 %. Cả hai đều ghi
            # hình đúng nên không có gì báo khi làm sai. Xem HARDWARE_BUDGET §6.3.
            continue

        pad = pad_index.get(camera.role, 0)
        sink_pad = muxer.request_pad_simple(f"sink_{pad}")
        if sink_pad is None:
            raise RuntimeError(f"nvstreammux {camera.role.value} không cấp được pad sink_{pad}")

        # ⚠️ Tên hàng đợi theo CHỈ SỐ BIN, không theo chỉ số pad. `replace_source(index)`
        # tra hàng đợi theo chỉ số bin, và hai số này lệch nhau ngay khi có một camera
        # không decode nằm TRƯỚC một camera có decode. Cấu hình GC03 hiện tại đặt chúng ở
        # cuối nên chưa lệch — đó là may, không phải thiết kế.
        qname = source_queue_name(index)
        queue = make(Gst, "queue", qname)
        apply_props(queue, SOURCE_QUEUE)
        pipeline.add(queue)

        name = source_bin_name(index)
        link_pads(Gst, bin_.get_static_pad("src"), queue.get_static_pad("sink"), name, qname)
        link_pads(Gst, queue.get_static_pad("src"), sink_pad, qname, f"mux_{camera.role.value}")

        # Một camera rớt mạng KHÔNG được kéo sập muxer dùng chung: chặn EOS tại pad của nó.
        _drop_eos(Gst, sink_pad)

        if watchdog is not None:
            watchdog.watch(index, camera, bin_.get_static_pad("src"))

        attached.setdefault(camera.role, {})[pad] = camera
        pad_index[camera.role] = pad + 1

    return attached


def make_source_bin(Gst: Any, index: int, camera: CameraConfig, uri: str | None = None) -> Any:
    """Một ``nvurisrcbin`` bọc trong bin có đúng một ghost src pad.

    Args:
        uri: ghi đè URL trong config. Chỉ dùng để **thí nghiệm** — trỏ vào một RTSP server
            cục bộ mà ta cắt được kết nối, thứ không làm được với camera thật ở cảng.
    """
    bin_ = Gst.Bin.new(source_bin_name(index))
    uri = uri or camera.rtsp_record
    src = make(Gst, "nvurisrcbin", f"uridecode_{index}")
    src.set_property("uri", uri)
    apply_props(src, NVURISRCBIN)

    # Giảm nhịp cho nhánh model. KHÔNG giảm tải NVDEC — nguồn là IPPP, GOP 50, không một
    # khung B nào (đo 2026-08-29), nên mọi khung vẫn phải giải mã và cái này chỉ vứt output
    # SAU đó. Nhưng NVDEC không phải nút thắt duy nhất: nó cắt 6 lần công việc ở
    # nvstreammux, copy buffer, nvinferserver, probe và Kafka. Xem HARDWARE_BUDGET §2.2.
    #
    # Đặt ở decoder chứ không ở probe vì đây là chỗ SỚM NHẤT bỏ được khung: probe chỉ cứu
    # được phần suy luận, khung đã bị gộp batch và copy rồi.
    if camera.drop_frame_interval:
        src.set_property("drop-frame-interval", camera.drop_frame_interval)

    ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    bin_.add_pad(ghost)
    linked = [False]

    def _on_pad_added(_element: Any, pad: Any) -> None:
        if linked[0]:
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps:
            return
        struct = caps.get_structure(0)
        if struct is None or "video" not in struct.get_name():
            return

        features = caps.get_features(0)
        if features is not None and features.contains("memory:NVMM"):
            ghost.set_target(pad)
            linked[0] = True
            return

        # Pad ở bộ nhớ hệ thống: nvstreammux cần NVMM, nên phải chuyển.
        conv = make(Gst, "nvvideoconvert", f"vidconv_{index}")
        bin_.add(conv)
        conv.sync_state_with_parent()
        pad.link(conv.get_static_pad("sink"))
        ghost.set_target(conv.get_static_pad("src"))
        linked[0] = True

    src.connect("pad-added", _on_pad_added)
    bin_.add(src)
    _bound_dec_queue(bin_)
    return bin_


def _bound_dec_queue(source_bin: Any) -> None:
    """Chặn hàng đợi nội bộ của ``nvurisrcbin`` theo thời gian và byte.

    ⚠️ ``nvurisrcbin`` chỉ đặt ``leaky`` và ``max-size-buffers``, để ``max-size-time`` ở
    mặc định gst là **một giây**. Một giây tồn đọng đủ chặn ``tee_rtsp_pre_decode``, và tee
    đẩy tuần tự nên nhánh ghi đứng theo — mất bằng chứng vì suy luận chậm.

    Hàng đợi chỉ tồn tại sau khi ``nvurisrcbin`` dựng xong phần bên trong; nguồn là file
    thì không bao giờ có, và đó không phải lỗi.
    """

    def _apply(queue: Any) -> None:
        apply_props(queue, DEC_QUEUE)

    existing = source_bin.get_by_name("dec_que")
    if existing is not None:
        _apply(existing)
        return

    def _on_deep_added(_bin: Any, _sub: Any, element: Any) -> None:
        if element.get_name() == "dec_que":
            _apply(element)

    source_bin.connect("deep-element-added", _on_deep_added)


def _drop_eos(Gst: Any, pad: Any) -> None:
    """Nuốt EOS tại pad vào muxer.

    Nguồn live không bao giờ "hết" một cách hợp lệ: EOS nghĩa là camera rớt. Để nó đi tiếp
    thì ``nvstreammux`` coi như cả batch kết thúc và kéo sập **mọi** camera còn lại.
    """

    def _probe(_pad: Any, info: Any) -> Any:
        event = info.get_event()
        if event is not None and event.type == Gst.EventType.EOS:
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, _probe)


def replace_source(
    Gst: Any,
    pipeline: Any,
    index: int,
    camera: CameraConfig,
    *,
    recorder: Any | None = None,
    watchdog: Any | None = None,
    uri: str | None = None,
) -> Any:
    """Thay một source bin đã chết bằng bin mới, các camera khác vẫn chạy.

    ⚠️ **Chỉ cái bin bị thay.** Hàng đợi giữ nguyên request pad của nó ở
    ``nvstreammux``, nên muxer không phải thương lượng lại. Trả request pad của streammux
    lúc đang chạy là thao tác đã biết là làm treo pipeline, và thiết kế này không có lý do
    nào phải làm vậy.
    """
    name = source_bin_name(index)
    old = pipeline.get_by_name(name)
    if old is None:
        raise RuntimeError(f"{name}: không có gì để thay")

    qname = source_queue_name(index)
    queue = pipeline.get_by_name(qname)
    if queue is None:
        # ⚠️ Ném, KHÔNG bỏ qua. Bản đầu để `queue_sink = None` rồi lặng lẽ không nối —
        # nguồn mới chạy, pipeline không báo gì, và camera đó im cho tới lần khởi động lại
        # process. Nơi gọi là watchdog, nên "không làm gì" là kết quả tệ nhất có thể.
        raise RuntimeError(
            f"{qname}: không tìm thấy hàng đợi của nguồn {index}. Nơi dựng pipeline phải "
            f"đặt tên bằng source_queue_name(), nếu không replace_source() không nối lại được."
        )
    queue_sink = queue.get_static_pad("sink")

    old_src = old.get_static_pad("src")
    if old_src is not None and old_src.get_peer() is queue_sink:
        old_src.unlink(queue_sink)
    old.set_state(Gst.State.NULL)
    old.get_state(5 * Gst.SECOND)
    pipeline.remove(old)

    fresh = make_source_bin(Gst, index, camera, uri)
    pipeline.add(fresh)
    if recorder is not None:
        recorder.attach(Gst, fresh, camera.code)
    link_pads(Gst, fresh.get_static_pad("src"), queue_sink, name, qname)
    if watchdog is not None:
        watchdog.watch(index, camera, fresh.get_static_pad("src"))

    if not fresh.sync_state_with_parent():
        fresh.set_state(Gst.State.PLAYING)
    return fresh
