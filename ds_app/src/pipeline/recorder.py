"""Nhánh ghi hình: ``splitmuxsink`` cắm vào luồng **CHƯA DECODE** bên trong ``nvurisrcbin``.

Đây là điều kiện sống còn trên RTX 3060. Card GeForce chỉ có **1 NVENC** và bị driver giới
hạn số phiên encode đồng thời, nên ghi 10 camera bằng cách decode rồi encode lại là bất
khả thi. Cắm vào trước decoder và mux thẳng bitstream gốc ⇒ **0 phiên encode, 0 phiên
decode thêm**. Đo thật: `enc=0` suốt lần chạy. Xem ``docs/DESIGN_NOTES.md`` DN-014.

⚠️ **Không tự dựng ``tee``.** ``nvurisrcbin`` đã có ``tee_rtsp_pre_decode`` bên trong, đúng
chỗ cần. Tự ghép ``rtspsrc ! depay ! h265parse ! tee`` thì chết ở negotiate: hai đích cần
``stream-format`` loại trừ nhau — ``nvv4l2decoder`` đòi ``byte-stream``, ``mp4mux`` đòi
``hvc1``/``hev1`` — và một ``h265parse`` chỉ thương lượng được một. Lỗi hiện ra là
``not-negotiated (-4)``, không nhắc gì tới caps.

``nvurisrcbin`` dựng phần bên trong **bất đồng bộ**, nên phải chờ ``deep-element-added``
rồi mới ghép vào.

Module này **không import ``gi``**: ``Gst`` được chuyền vào :meth:`RecordingBranch.attach`
và giữ lại cho các callback. Nhờ vậy nó nạp và test được trên máy không có DeepStream.
"""

from __future__ import annotations

import datetime
import statistics
from pathlib import Path
from typing import Any

from ds_app.src.pipeline.elements import MUXER, RECORD_QUEUE, SPLITMUX, apply_props, make
from ds_app.src.pipeline.timesync import FragmentIndex, TimeSync

__all__ = ["RecordingBranch"]

_GOP_SAMPLES = 3
"""Số khoảng keyframe lấy mẫu trước khi báo GOP. Khoảng đầu tiên sau khi kết nối là khoảng
cụt (ta vào giữa GOP), nên lấy trung vị vài mẫu thay vì tin mẫu đầu."""

_MARKER = "src_cap_filter_nvvidconv"
"""Element đánh dấu ``nvurisrcbin`` đã dựng xong phần bên trong. Chờ đúng nó thay vì chờ
``tee_rtsp_pre_decode``: tee xuất hiện sớm hơn, lúc ``parser`` chưa có."""


class RecordingBranch:
    """Ghép nhánh ghi vào từng source bin.

    Args:
        output_dir: Thư mục gốc; mỗi camera một thư mục con theo ``cam_id``.
        file_format: Mẫu ``strftime`` cho tên file, không kèm đuôi.
        time_sync: Neo ``PTS → unix`` dùng chung với nhánh model. ``None`` ⇒ dùng đồng hồ
            tường, và khi đó mốc đoạn sẽ **trôi khỏi** dấu thời gian khung — cửa sổ cắt
            clip lệch dần mà không có gì báo. Chỉ để ``None`` khi chạy độc lập.
        fragments: Sổ đoạn để tra "khoảnh khắc T nằm ở đoạn nào".
        on_fragment: Gọi ``(cam_id, path, mo_luc_unix)`` mỗi khi mở một đoạn mới.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        file_format: str = "%Y%m%d_%H%M%S",
        time_sync: TimeSync | None = None,
        fragments: FragmentIndex | None = None,
        on_fragment: Any | None = None,
    ) -> None:
        self._root = Path(output_dir)
        self._file_format = file_format
        self._time_sync = time_sync
        self._fragments = fragments
        self._on_fragment = on_fragment
        self._Gst: Any = None
        """``Gst`` nhận được ở :meth:`attach`.

        Giữ lại thay vì ``from gi.repository import Gst`` trong từng callback: import muộn
        làm module này KHÔNG nạp được trên máy không có DeepStream, đúng thứ nó tự nhận là
        làm được. Nơi gọi đã có ``Gst`` rồi — chuyền vào là đủ."""

        self._no_pts_warned: set[str] = set()
        self._last_keyframe: dict[str, float] = {}
        self._gop_gaps: dict[str, list[float]] = {}
        self._gop_reported: set[str] = set()

    # ------------------------------------------------------------------ gắn vào
    def attach(self, Gst: Any, source_bin: Any, cam_id: str) -> None:
        """Ghép nhánh ghi vào ``source_bin``. An toàn khi gọi trước lúc bin dựng xong."""
        self._Gst = Gst
        (self._root / cam_id).mkdir(parents=True, exist_ok=True)

        for child in _descendants(Gst, source_bin):
            if child.get_name() == _MARKER:
                self._graft(Gst, source_bin, cam_id)
                return
        source_bin.connect("deep-element-added", self._on_deep_added, cam_id)

    def _on_deep_added(self, top_bin: Any, _sub_bin: Any, element: Any, cam_id: str) -> None:
        if element.get_name() == _MARKER:
            self._graft(self._Gst, top_bin, cam_id)

    def _graft(self, Gst: Any, source_bin: Any, cam_id: str) -> None:
        tee = source_bin.get_by_name("tee_rtsp_pre_decode")
        parser = source_bin.get_by_name("parser")
        if tee is None or parser is None:
            raise RuntimeError(
                f"camera {cam_id}: không thấy phần bên trong của nvurisrcbin "
                f"(tee_rtsp_pre_decode={tee!r}, parser={parser!r}). "
                f"Phiên bản DeepStream đổi tên element bên trong?"
            )

        # Element mới phải vào ĐÚNG cái bin chứa tee — nó có thể lồng sâu hơn source_bin.
        host = tee.get_parent()

        # Cùng họ parser với nhánh decode: nguồn có thể là H.264 hoặc H.265.
        parse_factory = parser.get_factory().get_name()
        record_parse = make(Gst, parse_factory, f"parse_record_{cam_id}")
        # ⚠️ config-interval=-1 BẮT BUỘC: chèn lại VPS/SPS/PPS trước mỗi keyframe. Không có
        # nó thì từng đoạn không tự đứng được và `evidenced` không mở nổi file.
        record_parse.set_property("config-interval", -1)

        queue = make(Gst, "queue", f"queue_record_{cam_id}")
        apply_props(queue, RECORD_QUEUE)

        sink = make(Gst, "splitmuxsink", f"splitmuxsink_{cam_id}")
        apply_props(sink, SPLITMUX)
        _configure_muxer(Gst, sink)

        # Chặn buffer không có PTS TRƯỚC khi nó tới muxer — xem _drop_pts_less.
        record_parse.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._drop_pts_less, cam_id
        )

        for element in (queue, record_parse, sink):
            host.add(element)
            element.sync_state_with_parent()

        tee_pad = tee.get_request_pad("src_%u")
        if tee_pad is None:
            raise RuntimeError(f"camera {cam_id}: tee_rtsp_pre_decode không cấp được request pad")
        if tee_pad.link(queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"camera {cam_id}: không nối được tee -> queue ghi hình")
        if not queue.link(record_parse) or not record_parse.link(sink):
            raise RuntimeError(f"camera {cam_id}: không nối được chuỗi ghi hình")

        sink.connect("format-location-full", self._on_new_fragment, cam_id)

    # ------------------------------------------------------------------ probe
    def _drop_pts_less(self, _pad: Any, info: Any, cam_id: str) -> Any:
        """Bỏ buffer không có PTS, và nhân tiện đo GOP.

        ⚠️ Nguồn RTSP thỉnh thoảng đẩy ra buffer không có dấu thời gian. Cho nó tới
        ``mp4mux`` thì muxer **abort** — mất cả đoạn đang ghi, không chỉ một khung.
        """
        Gst = self._Gst
        buf = info.get_buffer()
        if buf is not None and buf.pts != Gst.CLOCK_TIME_NONE:
            self._sample_gop(Gst, buf, cam_id)
            return Gst.PadProbeReturn.OK

        if cam_id not in self._no_pts_warned:
            self._no_pts_warned.add(cam_id)
            # Log MỘT lần cho mỗi camera: nguồn lỗi thì nó xảy ra liên tục, và log mỗi
            # buffer sẽ nhấn chìm mọi thứ khác.
            print(  # noqa: T201 — ds_app chạy trong container, stdout là log
                f"[record] {cam_id}: bỏ buffer không có PTS (sẽ làm abort mp4mux); "
                f"các lần sau của camera này không log nữa",
                flush=True,
            )
        return Gst.PadProbeReturn.DROP

    def _sample_gop(self, Gst: Any, buf: Any, cam_id: str) -> None:
        """Báo GOP thật của nguồn, một lần cho mỗi camera.

        GOP quyết định độ dài đoạn THẬT: ``splitmuxsink`` chỉ cắt tại keyframe, nên
        ``độ dài thật = ceil(giới hạn / GOP) lần GOP``. ``evidenced`` tính cửa sổ theo độ
        dài thật, nên con số này phải đo chứ không giả định.
        """
        if cam_id in self._gop_reported:
            return
        if buf.mini_object.flags & Gst.BufferFlags.DELTA_UNIT:
            return  # không phải keyframe

        pts_sec = buf.pts / Gst.SECOND
        previous = self._last_keyframe.get(cam_id)
        self._last_keyframe[cam_id] = pts_sec
        if previous is None:
            return

        gaps = self._gop_gaps.setdefault(cam_id, [])
        gaps.append(pts_sec - previous)
        if len(gaps) < _GOP_SAMPLES:
            return

        gop = statistics.median(gaps)
        self._gop_reported.add(cam_id)
        limit = SPLITMUX["max-size-time"] / 1e9
        real = gop if gop >= limit else (int(limit / gop) + (limit % gop > 0)) * gop
        print(  # noqa: T201
            f"[record] {cam_id}: GOP nguồn {gop:.2f}s, giới hạn {limit:.0f}s "
            f"⇒ đoạn dài thật ~{real:.1f}s",
            flush=True,
        )

    # ------------------------------------------------------------------ tên file
    def _on_new_fragment(self, _sink: Any, _fragment_id: int, sample: Any, cam_id: str) -> str:
        opened = self._fragment_unix(sample, cam_id)
        stamp = datetime.datetime.fromtimestamp(opened, tz=datetime.timezone.utc).strftime(
            self._file_format
        )
        path = str(self._root / cam_id / f"{stamp}.mp4")
        if self._fragments is not None:
            self._fragments.open_fragment(cam_id, path, opened, SPLITMUX["max-size-time"] / 1e9)
        if self._on_fragment is not None:
            self._on_fragment(cam_id, path, opened)
        return path

    def _fragment_unix(self, sample: Any, cam_id: str) -> float:
        """Mốc mở đoạn, **trên cùng trục mà probe đóng dấu khung**.

        Đồng hồ tường chỉ là đường lui khi chưa neo được (vài buffer đầu của RTSP có thể
        không có PTS). Lui hoài nghĩa là mốc đoạn và ``detect_time`` nằm trên hai trục — và
        hậu quả là cửa sổ cắt clip lệch dần, im lặng.
        """
        now = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
        if self._time_sync is None or sample is None:
            return now
        buf = sample.get_buffer()
        pts = getattr(buf, "pts", None) if buf is not None else None
        if pts is None or pts == self._Gst.CLOCK_TIME_NONE:
            return now
        pts_sec = pts / self._Gst.SECOND
        base = self._time_sync.anchor(cam_id, pts_sec, now)
        return base.to_unix(pts_sec) if base is not None else now


# ---------------------------------------------------------------------- trợ giúp


def _descendants(Gst: Any, container: Any) -> Any:
    it = container.iterate_elements()
    while True:
        ok, child = it.next()
        if ok != Gst.IteratorResult.OK:
            break
        yield child
        if hasattr(child, "iterate_elements"):
            yield from _descendants(Gst, child)


def _configure_muxer(Gst: Any, sink: Any) -> None:
    """Bật robust muxing để đoạn ĐANG GHI vẫn đọc được.

    ⚠️ Phải qua ``muxer-factory`` + ``muxer-properties``. Đặt thuộc tính ``muxer`` bằng một
    element cho ra fragment **0 byte** ("moov atom not found") — không có lỗi nào báo.
    """
    if sink.find_property("use-robust-muxing"):
        sink.set_property("use-robust-muxing", True)

    props = (
        "properties"
        f",reserved-max-duration=(guint64){int(MUXER['reserved-max-duration-sec'] * Gst.SECOND)}"
        f",reserved-moov-update-period=(guint64){int(MUXER['reserved-moov-update-period-sec'] * Gst.SECOND)}"
    )
    structure = Gst.Structure.new_from_string(props)
    if structure is None:
        raise RuntimeError(f"muxer-properties không phân tích được: {props!r}")
    sink.set_property("muxer-factory", MUXER["factory"])
    sink.set_property("muxer-properties", structure)
