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

import dataclasses
import datetime
import statistics
from pathlib import Path
from typing import Any

from ds_app.src.pipeline.elements import MUXER, RECORD_QUEUE, SPLITMUX, apply_props, make
from ds_app.src.pipeline.timesync import TimeSync
from internal.pkg.fragments import FragmentIndex

__all__ = ["RecordingBranch"]


@dataclasses.dataclass
class RecordLoss:
    """Bằng chứng nhánh ghi đã mất dữ liệu, đếm theo từng camera.

    Tồn tại vì mất mát ở nhánh ghi **không tự lộ ra**: file vẫn được tạo, vẫn mở được,
    chỉ thiếu hình ở giữa. Ai đó đi tìm bằng chứng cho một sự kiện sẽ phát hiện điều đó
    vài ngày sau, và lúc ấy không còn gì để truy.

    Hai nguồn số liệu đo hai thứ khác nhau, cố ý:

    * :attr:`overruns` — hàng đợi ghi báo ĐẦY, tức nó vừa vứt buffer. Bằng chứng trực tiếp
      về **nguyên nhân**, nhưng chỉ bắt được đúng nguyên nhân đó.
    * :attr:`keyframe_gaps` — hai keyframe cách nhau xa hơn GOP. Đo **kết quả**, nên bắt
      được mọi nguyên nhân: mất gói, hàng đợi xả, muxer nghẽn, camera trục trặc.
    """

    overruns: dict[str, int] = dataclasses.field(default_factory=dict)
    keyframe_gaps: dict[str, int] = dataclasses.field(default_factory=dict)
    rename_failures: dict[str, str] = dataclasses.field(default_factory=dict)
    """Đoạn đã ghi xong nhưng không đổi tên được — dữ liệu còn, tên còn `.part`."""

    @property
    def clean(self) -> bool:
        return not self.overruns and not self.keyframe_gaps and not self.rename_failures

    def report(self) -> list[str]:
        """Dòng báo cho từng camera có vấn đề. Rỗng = không mất gì."""
        out = []
        for name, why in sorted(self.rename_failures.items()):
            out.append(f"{name}: không đổi tên được khỏi `.part` — {why}")
        for code in sorted(set(self.overruns) | set(self.keyframe_gaps)):
            parts = []
            if n := self.overruns.get(code):
                parts.append(f"{n} lần hàng đợi ghi đầy (đã vứt buffer)")
            if n := self.keyframe_gaps.get(code):
                parts.append(f"{n} lần nghi mất khung I")
            out.append(f"{code}: " + ", ".join(parts))
        return out


PART_SUFFIX = ".part"
"""Hậu tố của đoạn **đang ghi**.

Đổi tên khi đóng xong là thao tác nguyên tử trong cùng thư mục, nên ai duyệt ``*.mp4``
không bao giờ thấy một file dở dang — kể cả khi tiến trình chết giữa chừng, file còn lại
mang đuôi ``.part`` và tự khai nó chưa hoàn tất.

⚠️ ``.part`` **không** có nghĩa là không đọc được: ``reserved-moov-update-period`` làm mới
``moov`` mỗi giây nên đoạn đang ghi vẫn mở được, và ``evidenced`` thường cần chính nó (cửa
sổ bằng chứng hay chạm vào đoạn hiện tại). Đuôi này chỉ nói **"chưa chốt"**, không nói
"hỏng"."""

_GOP_SAMPLES = 3
"""Số khoảng keyframe lấy mẫu trước khi báo GOP. Khoảng đầu tiên sau khi kết nối là khoảng
cụt (ta vào giữa GOP), nên lấy trung vị vài mẫu thay vì tin mẫu đầu."""

_MARKER = "src_cap_filter_nvvidconv"
"""Element đánh dấu ``nvurisrcbin`` đã dựng xong phần bên trong. Chờ đúng nó thay vì chờ
``tee_rtsp_pre_decode``: tee xuất hiện sớm hơn, lúc ``parser`` chưa có."""


class RecordingBranch:
    """Ghép nhánh ghi vào từng source bin.

    Args:
        output_dir: Thư mục gốc; mỗi camera một thư mục con theo ``camera_code``.
        file_format: ``"epoch"`` (mặc định) ⇒ tên file là dấu thời gian epoch **nguyên
            giây**, chính là lúc đoạn được tạo. Máy đọc được ngay mà không phải phân tích
            chuỗi, không dính múi giờ, và sắp xếp theo tên trùng sắp xếp theo thời gian.
            Truyền một mẫu ``strftime`` nếu cần tên cho người đọc.
        segment_sec: Độ dài đoạn mong muốn, giây.

            ⚠️ Đây là **giới hạn**, không phải độ dài thật. ``splitmuxsink`` chỉ cắt tại
            keyframe nên độ dài thật là ``ceil(giới hạn / GOP) lần GOP``. Đoạn dài hơn thì
            ít file hơn nhưng đoạn **đang mở** kéo dài hơn — và ``evidenced`` phải chờ một
            đoạn đóng mới cắt được từ nó, nên độ dài đoạn chính là độ trễ tệ nhất của
            bằng chứng.

        time_sync: Neo ``PTS → unix`` dùng chung với nhánh model. ``None`` ⇒ dùng đồng hồ
            tường, và khi đó mốc đoạn sẽ **trôi khỏi** dấu thời gian khung — cửa sổ cắt
            clip lệch dần mà không có gì báo. Chỉ để ``None`` khi chạy độc lập.
        fragments: Sổ đoạn để tra "khoảnh khắc T nằm ở đoạn nào".
        on_fragment: Gọi ``(camera_code, path, mo_luc_unix)`` mỗi khi mở một đoạn mới.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        segment_sec: float = 10.0,
        file_format: str = "epoch",
        time_sync: TimeSync | None = None,
        fragments: FragmentIndex | None = None,
        on_fragment: Any | None = None,
    ) -> None:
        self._root = Path(output_dir)
        if segment_sec <= 0:
            raise ValueError(f"segment_sec phải dương, nhận {segment_sec}")
        self._segment_sec = float(segment_sec)
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
        self._gop: dict[str, float] = {}
        self._loss = RecordLoss()
        self._pending: dict[str, Path] = {}

    # ------------------------------------------------------------------ gắn vào
    def attach(self, Gst: Any, source_bin: Any, camera_code: str) -> None:
        """Ghép nhánh ghi vào ``source_bin``. An toàn khi gọi trước lúc bin dựng xong."""
        self._Gst = Gst
        (self._root / camera_code).mkdir(parents=True, exist_ok=True)

        for child in _descendants(Gst, source_bin):
            if child.get_name() == _MARKER:
                self._graft(Gst, source_bin, camera_code)
                return
        source_bin.connect("deep-element-added", self._on_deep_added, camera_code)

    def _on_deep_added(self, top_bin: Any, _sub_bin: Any, element: Any, camera_code: str) -> None:
        if element.get_name() == _MARKER:
            self._graft(self._Gst, top_bin, camera_code)

    def _graft(self, Gst: Any, source_bin: Any, camera_code: str) -> None:
        tee = source_bin.get_by_name("tee_rtsp_pre_decode")
        parser = source_bin.get_by_name("parser")
        if tee is None or parser is None:
            raise RuntimeError(
                f"camera {camera_code}: không thấy phần bên trong của nvurisrcbin "
                f"(tee_rtsp_pre_decode={tee!r}, parser={parser!r}). "
                f"Phiên bản DeepStream đổi tên element bên trong?"
            )

        # Element mới phải vào ĐÚNG cái bin chứa tee — nó có thể lồng sâu hơn source_bin.
        host = tee.get_parent()

        # Cùng họ parser với nhánh decode: nguồn có thể là H.264 hoặc H.265.
        parse_factory = parser.get_factory().get_name()
        record_parse = make(Gst, parse_factory, f"parse_record_{camera_code}")
        # ⚠️ config-interval=-1 BẮT BUỘC: chèn lại VPS/SPS/PPS trước mỗi keyframe. Không có
        # nó thì từng đoạn không tự đứng được và `evidenced` không mở nổi file.
        record_parse.set_property("config-interval", -1)

        queue = make(Gst, "queue", f"queue_record_{camera_code}")
        apply_props(queue, RECORD_QUEUE)
        # `overrun` báo hàng đợi ĐẦY. Với queue có `leaky`, đầy nghĩa là nó vừa vứt một
        # buffer — bằng chứng TRỰC TIẾP rằng nhánh ghi vừa mất dữ liệu. Không bắt tín hiệu
        # này thì mất mát hoàn toàn im lặng: file vẫn ra, vẫn mở được, chỉ thiếu hình.
        queue.connect("overrun", self._on_overrun, camera_code)

        sink = make(Gst, "splitmuxsink", f"splitmuxsink_{camera_code}")
        apply_props(sink, SPLITMUX)
        sink.set_property("max-size-time", int(self._segment_sec * 1e9))
        _configure_muxer(Gst, sink)

        # Chặn buffer không có PTS TRƯỚC khi nó tới muxer — xem _drop_pts_less.
        record_parse.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._drop_pts_less, camera_code
        )

        for element in (queue, record_parse, sink):
            host.add(element)
            element.sync_state_with_parent()

        tee_pad = tee.get_request_pad("src_%u")
        if tee_pad is None:
            raise RuntimeError(
                f"camera {camera_code}: tee_rtsp_pre_decode không cấp được request pad"
            )
        if tee_pad.link(queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"camera {camera_code}: không nối được tee -> queue ghi hình")
        if not queue.link(record_parse) or not record_parse.link(sink):
            raise RuntimeError(f"camera {camera_code}: không nối được chuỗi ghi hình")

        sink.connect("format-location-full", self._on_new_fragment, camera_code)

    @property
    def loss(self) -> RecordLoss:
        """Thống kê mất mát. Kiểm nó sau mỗi lần chạy — sạch không phải là mặc định."""
        return self._loss

    def handle_bus_message(self, message: Any) -> bool:
        """Đổi tên ``.part`` → ``.mp4`` khi ``splitmuxsink`` báo đã đóng xong đoạn.

        Nối vào ``message::element`` của bus pipeline. Trả ``True`` nếu message này là của
        nhánh ghi.

        ⚠️ Phải chờ message này, KHÔNG được đổi tên lúc đoạn kế mở ra: ``async-finalize``
        đóng file ở luồng khác, nên "đoạn sau đã mở" không có nghĩa "đoạn trước đã ghi
        xong". Đo được: ``fragment-closed`` của đoạn N tới **sau** ``fragment-opened`` của
        đoạn N+1.
        """
        st = message.get_structure()
        if st is None or st.get_name() != "splitmuxsink-fragment-closed":
            return False
        ok, location = st.get_string("location"), None
        location = ok if isinstance(ok, str) else None
        if location is None:
            return True

        final = self._pending.pop(location, None)
        if final is None:
            return True
        try:
            Path(location).rename(final)
        except OSError as exc:
            # Không ném: một đoạn không đổi tên được vẫn còn nguyên dữ liệu dưới `.part`,
            # và làm đứng pipeline vì chuyện này thì mất nhiều hơn được.
            self._loss.rename_failures[final.name] = str(exc)
            print(  # noqa: T201
                f"[record] ⚠️ không đổi tên được {location} → {final}: {exc}", flush=True
            )
        return True

    # ------------------------------------------------------------------ probe
    def _drop_pts_less(self, _pad: Any, info: Any, camera_code: str) -> Any:
        """Bỏ buffer không có PTS, và nhân tiện đo GOP.

        ⚠️ Nguồn RTSP thỉnh thoảng đẩy ra buffer không có dấu thời gian. Cho nó tới
        ``mp4mux`` thì muxer **abort** — mất cả đoạn đang ghi, không chỉ một khung.
        """
        Gst = self._Gst
        buf = info.get_buffer()
        if buf is not None and buf.pts != Gst.CLOCK_TIME_NONE:
            self._sample_gop(Gst, buf, camera_code)
            return Gst.PadProbeReturn.OK

        if camera_code not in self._no_pts_warned:
            self._no_pts_warned.add(camera_code)
            # Log MỘT lần cho mỗi camera: nguồn lỗi thì nó xảy ra liên tục, và log mỗi
            # buffer sẽ nhấn chìm mọi thứ khác.
            print(  # noqa: T201 — ds_app chạy trong container, stdout là log
                f"[record] {camera_code}: bỏ buffer không có PTS (sẽ làm abort mp4mux); "
                f"các lần sau của camera này không log nữa",
                flush=True,
            )
        return Gst.PadProbeReturn.DROP

    def _on_overrun(self, _queue: Any, camera_code: str) -> None:
        self._loss.overruns[camera_code] = self._loss.overruns.get(camera_code, 0) + 1

    def _check_keyframe_gap(self, camera_code: str, gap: float) -> None:
        """Khoảng cách giữa hai keyframe vượt xa GOP đã học ⇒ đã mất một keyframe.

        Đây là phép dò chịu được mọi nguyên nhân — mất gói, hàng đợi xả, muxer nghẽn — vì
        nó đo **kết quả** chứ không đo một cơ chế cụ thể. Mất một khung I là mất cả GOP
        theo sau nó, tức tới 1,7 s hình không dựng lại được.
        """
        gop = self._gop.get(camera_code)
        if gop is None or gop <= 0:
            return
        # Ngưỡng 1,5 lần: GOP dao động thật (đo được 1,00 và 1,67 s ở hai lần chạy), nên
        # ngưỡng sát quá sẽ báo động giả. 1,5 lần thì chỉ khớp khi đã nhảy hẳn một chu kỳ.
        if gap > gop * 1.5:
            self._loss.keyframe_gaps[camera_code] = self._loss.keyframe_gaps.get(camera_code, 0) + 1
            print(  # noqa: T201
                f"[record] ⚠️ {camera_code}: hai keyframe cách {gap:.2f}s, GOP là "
                f"{gop:.2f}s — nhiều khả năng MẤT một khung I (mất tới {gap:.1f}s hình)",
                flush=True,
            )

    def _sample_gop(self, Gst: Any, buf: Any, camera_code: str) -> None:
        """Báo GOP thật của nguồn, một lần cho mỗi camera.

        GOP quyết định độ dài đoạn THẬT: ``splitmuxsink`` chỉ cắt tại keyframe, nên
        ``độ dài thật = ceil(giới hạn / GOP) lần GOP``. ``evidenced`` tính cửa sổ theo độ
        dài thật, nên con số này phải đo chứ không giả định.
        """
        if buf.mini_object.flags & Gst.BufferFlags.DELTA_UNIT:
            return  # không phải keyframe

        pts_sec = buf.pts / Gst.SECOND
        previous = self._last_keyframe.get(camera_code)
        self._last_keyframe[camera_code] = pts_sec
        if previous is None:
            return
        gap = pts_sec - previous

        # Dò MÃI, không chỉ trong lúc học GOP: mất keyframe là chuyện xảy ra bất kỳ lúc nào,
        # thường lúc đĩa hoặc mạng nghẽn — tức muộn hơn nhiều so với vài giây đầu.
        self._check_keyframe_gap(camera_code, gap)
        if camera_code in self._gop_reported:
            return

        gaps = self._gop_gaps.setdefault(camera_code, [])
        gaps.append(gap)
        if len(gaps) < _GOP_SAMPLES:
            return

        gop = statistics.median(gaps)
        self._gop[camera_code] = gop
        self._gop_reported.add(camera_code)
        limit = self._segment_sec
        real = gop if gop >= limit else (int(limit / gop) + (limit % gop > 0)) * gop
        print(  # noqa: T201
            f"[record] {camera_code}: GOP nguồn {gop:.2f}s, giới hạn {limit:.0f}s "
            f"⇒ đoạn dài thật ~{real:.1f}s",
            flush=True,
        )

    # ------------------------------------------------------------------ tên file
    def _on_new_fragment(self, _sink: Any, _fragment_id: int, sample: Any, camera_code: str) -> str:
        opened = self._fragment_unix(sample, camera_code)
        if self._file_format == "epoch":
            # Epoch nguyên, không phải chuỗi giờ: `evidenced` cần một CON SỐ để so với dấu
            # thời gian khung. Tên dạng giờ buộc nó phân tích ngược, mà phân tích ngược thì
            # phải đoán múi giờ — đoán sai một lần là lệch cả giờ.
            #
            # Mốc CHÍNH XÁC của đoạn vẫn là `opened` (có phần thập phân) và đi qua
            # `on_fragment` + `FragmentIndex`; tên file chỉ để người đọc và để sắp xếp.
            # Đoạn cách nhau ~10 s nên giây nguyên không đụng nhau.
            stamp = str(int(opened))
        else:
            stamp = datetime.datetime.fromtimestamp(opened, tz=datetime.timezone.utc).strftime(
                self._file_format
            )
        final = self._root / camera_code / f"{stamp}.mp4"
        path = str(final) + PART_SUFFIX
        self._pending[path] = final
        if self._fragments is not None:
            # Sổ đoạn giữ đường dẫn CUỐI: nơi tiêu thụ tra theo tên ổn định, và
            # `Fragment.live_path` mới là chỗ đọc khi đoạn chưa chốt.
            self._fragments.open_fragment(camera_code, str(final), opened, self._segment_sec)
        if self._on_fragment is not None:
            self._on_fragment(camera_code, str(final), opened)
        return path

    def _fragment_unix(self, sample: Any, camera_code: str) -> float:
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
        base = self._time_sync.anchor(camera_code, pts_sec, now)
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
