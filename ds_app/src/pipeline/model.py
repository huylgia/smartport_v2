"""Nhánh model: decode → gộp batch theo role → chép khung ra host → Triton.

**Một ``nvstreammux`` cho mỗi role, không phải một cái chung.** Muxer cho ra MỘT batch và
không có cách nào bảo tầng sau chỉ xử lý vài nguồn trong đó; các role lại dùng model khác
nhau và nhịp khác nhau. ``nvdsmetamux`` giải bài toán ngược lại — nhiều model trên *cùng*
một luồng — nên không dùng được ở đây.

Khác kế hoạch ban đầu ở một chỗ, và đây là lý do: kế hoạch định dùng
``nvinferserver`` với pattern PGIE→SGIE. Nó cần ``nvinferserver`` tự parse được đầu ra
detector để dựng ``NvDsObjectMeta``, mà PicoDet trả tensor thô và ``DetectionParams.nms``
trong ``nvdsinferserver_common.proto`` ghi *"reserved, not supported yet"*. Toàn bộ nhánh
suy luận vì thế nằm trong model BLS của Triton (``triton/bls/pico.py``), và ở đây chỉ còn
việc lấy khung ra rồi gửi đi.

Probe **không gọi Triton**. Nó chép khung rồi trả pad ngay; lời gọi ~18 ms diễn ra trên
thread của :class:`ds_app.src.pipeline.inference.InferenceClient`. Probe chặn một khung là
``nvstreammux`` chặn cả batch, và với ``leaky`` ở hàng đợi phía trên thì hệ quả là mất
khung chứ không phải chậm.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from common.config import CameraConfig, CraneConfig
from common.enum import CameraRole
from ds_app.src.pipeline.elements import NVVIDEOCONVERT, STREAMMUX, apply_props, link, make
from ds_app.src.pipeline.inference import FrameJob
from ds_app.src.pipeline.timesync import TimeSync

if TYPE_CHECKING:
    from internal.pkg.nptypes import Image

__all__ = ["ModelBranch", "roles_with_cameras"]


def roles_with_cameras(crane: CraneConfig) -> dict[CameraRole, list[CameraConfig]]:
    """Các role ds_app **chạy được ngay bây giờ**, theo thứ tự ổn định.

    Hai bộ lọc, và cái thứ hai quan trọng hơn vẻ ngoài của nó:

    * bỏ role rỗng — dựng một muxer không có nguồn nào thì pipeline không bao giờ PREROLL
      xong, và nó biểu hiện thành "treo lúc khởi động" chứ không thành lỗi;
    * bỏ role **chưa có model BLS**. ``CameraRole.runs_model`` nói role đó *rốt cuộc* sẽ
      chạy model, còn :data:`~ds_app.src.pipeline.inference.BLS_FOR_ROLE` nói hôm nay đã có
      model chưa. Trộn hai câu đó lại thì ``ccode`` (Phase 3b) lọt vào: 5 camera decode
      suốt phiên chạy để rồi mỗi khung ném ``KeyError`` — đo được 1 503 lỗi trong 60 giây,
      trong khi hai role kia vẫn chạy đúng nên bảng tổng kết trông gần như bình thường.
    """
    from ds_app.src.pipeline.inference import BLS_FOR_ROLE

    out: dict[CameraRole, list[CameraConfig]] = {}
    for role in CameraRole:
        if not role.runs_model or role not in BLS_FOR_ROLE:
            continue
        cams = [c for c in crane.cameras.get(role, []) if c.decodes]
        if cams:
            out[role] = cams
    return out


class ModelBranch:
    """Muxer + chuyển đổi + probe cho MỘT role.

    Nơi gọi dựng một cái cho mỗi role rồi nối nguồn của role đó vào :attr:`muxer`.
    """

    def __init__(
        self,
        role: CameraRole,
        cameras: list[CameraConfig],
        crane: CraneConfig,
        *,
        submit: Callable[[FrameJob], bool],
        time_sync: TimeSync,
        segment_hint: Callable[[str, float], str | None] | None = None,
    ) -> None:
        self.role = role
        self.cameras = cameras
        self.crane = crane
        self._submit = submit
        # DÙNG CHUNG với nhánh ghi. Hai nhánh neo thời gian riêng thì chúng trôi khỏi nhau
        # và cửa sổ cắt clip lệch dần — không có gì báo. Xem timesync.py.
        self._sync = time_sync
        self._segment_hint = segment_hint
        self.muxer: Any = None
        self._pad_of_camera: dict[int, CameraConfig] = {}

    def build(self, Gst: Any, pipeline: Any) -> Any:
        """Dựng muxer → nvvideoconvert → capsfilter → fakesink, gắn probe. Trả muxer."""
        name = self.role.value
        self.muxer = make(Gst, "nvstreammux", f"mux_{name}")
        apply_props(self.muxer, STREAMMUX)
        # batch-size = số camera của role: một batch gom trọn một "lát thời gian" của role
        # đó. Đặt nhỏ hơn thì muxer chia thành nhiều batch và nhịp mỗi camera tụt theo.
        self.muxer.set_property("batch-size", len(self.cameras))

        convert = make(Gst, "nvvideoconvert", f"conv_{name}")
        apply_props(convert, NVVIDEOCONVERT)

        caps = make(Gst, "capsfilter", f"caps_{name}")
        # RGBA: định dạng DUY NHẤT mà `pyds.get_nvds_buf_surface()` map ra numpy được.
        # NV12 (mặc định của muxer) map ra sẽ là dữ liệu phẳng Y/UV, không phải ảnh màu.
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

        sink = make(Gst, "fakesink", f"sink_{name}")
        sink.set_property("sync", False)
        # ⚠️ async=0 BẮT BUỘC: pipeline có tee (nhánh ghi) và nguồn động. Thiếu nó thì
        # pipeline kẹt ở PAUSED và không có gì báo — chỉ là không có khung nào chạy.
        sink.set_property("async", False)
        # Không giữ buffer cuối: nó neo một buffer NVMM lại mãi, và pool chỉ có 8 cái.
        sink.set_property("enable-last-sample", False)

        for element in (self.muxer, convert, caps, sink):
            pipeline.add(element)
        chain = ((self.muxer, "mux"), (convert, "conv"), (caps, "caps"), (sink, "sink"))
        for (a, a_name), (b, b_name) in itertools.pairwise(chain):
            link(a, b, f"{a_name}_{name}", f"{b_name}_{name}")

        caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._probe, None)
        return self.muxer

    def attach(self, pad_index: int, camera: CameraConfig) -> None:
        """Ghi nhận camera nào ngồi ở pad nào.

        Metadata của DeepStream chỉ mang ``pad_index``; danh tính camera phải tra bằng
        bảng này, nếu không kết quả bị gán cho nhầm camera mà không có gì báo.
        """
        self._pad_of_camera[pad_index] = camera

    # -- probe ---------------------------------------------------------------

    def _probe(self, pad: Any, info: Any, _user: Any) -> Any:
        import gi

        gi.require_version("Gst", "1.0")
        import pyds
        from gi.repository import Gst

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if batch is None:
            return Gst.PadProbeReturn.OK

        node = batch.frame_meta_list
        while node is not None:
            try:
                frame = pyds.NvDsFrameMeta.cast(node.data)
            except StopIteration:
                break

            camera = self._pad_of_camera.get(frame.pad_index)
            if camera is not None:
                self._dispatch(pyds, buf, frame, camera)
            node = node.next

        return Gst.PadProbeReturn.OK

    def _dispatch(self, pyds: Any, buf: Any, frame: Any, camera: CameraConfig) -> None:
        import time

        from internal.pkg.timebase import restore_frame_id

        pts_sec = frame.buf_pts / 1e9
        base = self._sync.anchor(camera.code, pts_sec, time.time(), series="model")
        if base is None:
            # Vài buffer đầu của nguồn RTSP chưa có PTS hợp lệ. Bỏ khung thay vì đoán một
            # mốc: một dấu thời gian sai đi thẳng vào cửa sổ cắt clip.
            return
        frame_ts = base.to_unix(pts_sec)

        # `frame_num` đếm khung ĐÃ QUA decimate; khôi phục trước khi dùng làm chỉ số, nếu
        # không nó co lại đúng theo tỉ lệ decimate và không khớp với chỉ số của nhánh ghi.
        frame_id = restore_frame_id(frame.frame_num, camera.drop_frame_interval)

        # `get_nvds_buf_surface` trả một VIEW vào buffer GStreamer — nó chỉ sống tới khi
        # probe trả về. Phải CHÉP, và chép trước khi bỏ kênh alpha: cắt view rồi mới chép
        # thì vẫn là chép, chỉ ít byte hơn.
        surface: Image = pyds.get_nvds_buf_surface(hash(buf), frame.batch_id)
        image = np.array(surface[:, :, :3][:, :, ::-1], copy=True, order="C")  # RGBA -> BGR
        _dump_once(surface, camera.code)

        job = FrameJob(
            camera_code=camera.code,
            role=self.role,
            frame_id=frame_id,
            frame_ts=frame_ts,
            # `base_unix`, KHÔNG phải `to_unix(0.0)`. Cái sau là gốc PTS của NGUỒN,
            # còn `frame_id` đếm từ khung đầu tiên PIPELINE thấy — hai gốc khác nhau,
            # và hệ quả là `start_ts + frame_id/fps` lệch `frame_ts` một khoảng cố
            # định bằng PTS của khung đầu (đo được 0,475 s). Nhịp thì vẫn đúng, nên
            # không có gì báo — chỉ là mọi cửa sổ thời gian trượt đi nửa giây.
            start_ts=base.base_unix,
            source_fps=camera.source_fps,
            image=image,
            segment_hint=(
                self._segment_hint(camera.code, frame_ts) if self._segment_hint else None
            ),
        )
        self._submit(job)


_DUMPED: set[str] = set()


def _dump_once(surface: Image, camera_code: str) -> None:
    """Ghi MỘT khung thô ra ``.npy`` khi ``CRANEOPS_DUMP_FRAME`` được đặt.

    Giả định "surface là RGBA" không kiểm được bằng test đơn vị: nếu nó thật ra là BGRA
    thì model vẫn trả về hộp, chỉ kém đi, và không có gì báo. Cách duy nhất là nhìn.

    Ghi mảng thô chứ không ghi PNG: image này KHÔNG có OpenCV (xem ``build/ds_app.Dockerfile``
    — thêm nó chỉ để debug sẽ tạo ra một bản cv2 thứ hai có thể lệch phiên bản với bản
    trong Triton). Soi bằng công cụ ở host.
    """
    import os

    out = os.environ.get("CRANEOPS_DUMP_FRAME")
    if not out or camera_code in _DUMPED:
        return
    _DUMPED.add(camera_code)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{camera_code}.npy", np.ascontiguousarray(surface[:, :, :3]))
