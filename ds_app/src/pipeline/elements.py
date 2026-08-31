"""Hằng số tinh chỉnh GStreamer, và ba hàm trợ giúp dựng/nối element.

**Mọi con số ở đây đều đã trả giá.** Chúng không phải mặc định của DeepStream — mặc định
của DS sai cho một pipeline vừa ghi hình vừa suy luận, và sai theo kiểu im lặng. Mỗi hằng
số kèm lý do; đổi nó mà không đọc lý do là cách nhanh nhất để dựng lại một lỗi cũ.

Đặt hết ở một chỗ thay vì rải trong code dựng pipeline: khi cần chỉnh vì tải thực địa,
người vận hành phải tìm được chúng trong một file.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEC_QUEUE",
    "MUXER",
    "NVURISRCBIN",
    "RECORD_QUEUE",
    "SOURCE_QUEUE",
    "SPLITMUX",
    "link",
    "link_pads",
    "make",
]


NVURISRCBIN: dict[str, Any] = {
    # 4 = TCP. Camera cảng đi qua Internet công cộng; UDP mất gói thành vệt nhiễu kéo dài
    # tới keyframe kế tiếp, mà keyframe cách nhau ~1,7 s.
    "select-rtp-protocol": 4,
    # Độ sâu jitterbuffer (ms). Sâu hơn thì chịu jitter tốt hơn nhưng dồn thêm bộ đếm
    # gói-mất, và tăng độ trễ đầu-cuối.
    "latency": 1500,
    # ⚠️ DeepStream mặc định TRUE. Khi TRUE, rtpjitterbuffer VỨT mọi gói tới trễ hơn
    # `latency` — ở phía TRƯỚC tee, nên nó làm hỏng CẢ nhánh ghi lẫn nhánh model. FALSE
    # nghĩa là để buffer phình ra và đẩy gói trễ đi tiếp; phần phình được chặn bởi `leaky`
    # của dec_que bên dưới.
    "drop-on-latency": False,
    # ⚠️ DeepStream mặc định 0 (không rò rỉ) cho dec_que. Với 0, khi suy luận chậm lại thì
    # dec_que đầy và **chặn tee dùng chung** — kéo theo nhánh ghi đứng luôn. 2 =
    # downstream: bỏ khung ở nhánh decode để tee không bao giờ nghẽn. Mất khung suy luận
    # chấp nhận được; mất đoạn ghi hình thì không, vì nó là bằng chứng.
    "leaky": 2,
    "max-size-buffers": 0,  # 0 = không giới hạn theo số buffer; DEC_QUEUE chặn theo thời gian/byte
    # Số giây KHÔNG có dữ liệu thì buộc kết nối lại.
    "rtsp-reconnect-interval": 30,
    # -1 = thử lại mãi. Đếm số lần thất bại LIÊN TIẾP, nên đặt giới hạn hữu hạn chỉ có
    # nghĩa là "chịu được N lần interval giây mất mạng"; quá đó camera chết vĩnh viễn cho tới
    # khi restart process. Với hệ chạy 24/7 ở cảng, thử lại mãi là đúng.
    "rtsp-reconnect-attempts": -1,
}
"""Thuộc tính của ``nvurisrcbin`` — nguồn RTSP kèm decode.

Dùng ``nvurisrcbin`` chứ **không** tự ghép ``rtspsrc ! depay ! parse ! decoder``: nó đã có
sẵn ``tee_rtsp_pre_decode`` bên trong, đúng chỗ nhánh ghi cần cắm vào, và nó tự lo phần
kết nối lại RTSP."""

DEC_QUEUE: dict[str, Any] = {
    # ⚠️ `nvurisrcbin` chỉ đặt `leaky` và `max-size-buffers` cho dec_que, để nguyên
    # `max-size-time` ở mặc định gst là **1 GIÂY**. Một giây tồn đọng là đủ chặn tee dùng
    # chung và kéo nhánh ghi đứng theo. Nới ra để `leaky` mới là van xả, không phải chặn.
    "max-size-time": 4_000_000_000,  # 4 s
    "max-size-bytes": 33_554_432,  # 32 MiB
}
"""Chặn cho hàng đợi nội bộ của ``nvurisrcbin``. Ghép vào sau khi nó tự dựng — xem
``sources._bound_dec_queue``."""

RECORD_QUEUE: dict[str, Any] = {
    # Chặn theo THỜI GIAN, không theo số buffer: một ngưỡng tính bằng buffer được viết cho
    # 30 fps sẽ âm thầm thành một ngân sách khác khi nguồn đổi fps.
    "max-size-buffers": 0,
    "max-size-time": 30_000_000_000,  # 30 s video đã nén
    "max-size-bytes": 67_108_864,  # 64 MiB
    # 2 = downstream. Nhánh ghi xả SAU CÙNG: mất một khung P làm nhoè hình tới tận keyframe
    # kế tiếp. Nhưng chặn còn tệ hơn — tee đẩy tuần tự, nên nhánh ghi đứng là nhánh model
    # đứng theo.
    "leaky": 2,
}
"""Hàng đợi của nhánh ghi, cắm vào ``tee_rtsp_pre_decode``."""

SOURCE_QUEUE: dict[str, Any] = {
    # Tách từng nguồn khỏi nvstreammux DÙNG CHUNG: một camera phun dữ liệu đột biến không
    # đẩy thẳng vào muxer, và một muxer đang nghẽn không làm đứng decoder của camera này.
    "max-size-buffers": 4,
    "max-size-time": 0,
    "max-size-bytes": 0,
    "leaky": 2,
}
"""Hàng đợi giữa mỗi nguồn và ``nvstreammux``."""

SPLITMUX: dict[str, Any] = {
    # MẶC ĐỊNH cho độ dài đoạn — `RecordingBranch(segment_sec=...)` ghi đè nó, và luôn ghi
    # đè, nên đổi con số ở đây chỉ đổi giá trị dự phòng.
    #
    # ⚠️ splitmuxsink CHỈ cắt tại keyframe, nên độ dài thật là bội số của GOP:
    #     độ dài thật = ceil(giới hạn / GOP) lần GOP
    # Nguồn cảng có GOP ≈ 1,7 s (đo 2026-08-30) ⇒ 10 s cho ra ~10,2 s. `evidenced` tính cửa
    # sổ theo độ dài THẬT (FragmentIndex.observed_duration), không theo con số này.
    "max-size-time": 10_000_000_000,
    # Đóng file ở luồng riêng: đóng đồng bộ chặn nhánh ghi đúng lúc chuyển file.
    "async-finalize": True,
    "async-handling": True,
    # 0 = giữ hết. Việc dọn file cũ là của sweeper, không phải của splitmuxsink: nó chỉ
    # đếm được số file, không biết `evidenced` còn cần đoạn nào.
    "max-files": 0,
}
"""Thuộc tính ``splitmuxsink`` của nhánh ghi."""

MUXER: dict[str, Any] = {
    "factory": "mp4mux",
    # Làm mới `moov` định kỳ để đoạn ĐANG GHI vẫn đọc được. Không có nó, `evidenced` phải
    # chờ file đóng mới cắt được — mà cửa sổ bằng chứng thường chạm vào đoạn hiện tại.
    #
    # ⚠️ `use-robust-muxing` MỘT MÌNH không làm gì: phải đặt `reserved-moov-update-period`
    # tường minh. Và phải đặt `reserved-max-duration` rộng — độ dài thật do GOP quyết định
    # nên không biết trước; đặt thiếu thì mất cả đoạn, đặt dư chỉ tốn ~vài trăm byte header.
    "reserved-max-duration-sec": 300,
    "reserved-moov-update-period-sec": 1,
}
"""Cấu hình muxer bên trong ``splitmuxsink``.

⚠️ **Phải cấu hình qua ``muxer-factory`` + ``muxer-properties``, KHÔNG phải ``muxer``.**
Đặt thuộc tính ``muxer`` bằng một element cho ra fragment **0 byte** ("moov atom not
found") — đo được, và không có thông báo lỗi nào."""


def make(Gst: Any, factory: str, name: str) -> Any:
    """Dựng một element, hỏng thì ném lỗi nói rõ thiếu gì.

    ``Gst.ElementFactory.make`` trả ``None`` khi thiếu plugin — im lặng. Để ``None`` đi
    tiếp thì lỗi nổ ở chỗ khác, thường là một ``AttributeError`` không liên quan.
    """
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(
            f"không dựng được element {factory!r} (tên {name!r}) — thiếu plugin DeepStream? "
            f"Kiểm bằng: gst-inspect-1.0 {factory}"
        )
    return element


def link(a: Any, b: Any, a_name: str, b_name: str) -> None:
    if not a.link(b):
        raise RuntimeError(f"không nối được {a_name} -> {b_name}")


def link_pads(Gst: Any, src_pad: Any, sink_pad: Any, a_name: str, b_name: str) -> None:
    if src_pad is None or sink_pad is None:
        raise RuntimeError(
            f"thiếu pad khi nối {a_name} -> {b_name} (src={src_pad!r}, sink={sink_pad!r})"
        )
    result = src_pad.link(sink_pad)
    if result != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"không nối được pad {a_name} -> {b_name}: {result}")


def apply_props(element: Any, props: dict[str, Any]) -> None:
    """Đặt một loạt thuộc tính, bỏ qua khoá không phải thuộc tính GStreamer.

    Khoá có hậu tố ``-sec`` là giá trị dẫn xuất (cần nhân với ``Gst.SECOND``) nên nơi gọi
    tự xử lý; ``factory`` cũng vậy.
    """
    for key, value in props.items():
        if key.endswith("-sec") or key == "factory":
            continue
        element.set_property(key, value)
