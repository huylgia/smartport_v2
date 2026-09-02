"""Cấu hình cẩu — nguồn sự thật cho `ds_app` và mọi service đọc `configs/cranes/*.yaml`.

Ba việc file này làm, và **chỉ** ba việc đó (không I/O ngoài đọc YAML, không dựng
pipeline, không biết gì về GStreamer):

1. **Validate fail-fast.** Gõ sai một khoá là lỗi lúc load, không phải một `.get(k, default)`
   im lặng lúc chạy. ``extra="forbid"`` cho **config** — ngược với message contract, nơi
   dùng ``extra="ignore"`` để chịu được nâng cấp lệch pha (xem ``common/message.py``).
2. **Nội suy secret từ môi trường.** URL RTSP có mật khẩu nên **không bao giờ** nằm trong
   YAML. YAML **không nhắc tới URL**: camera nhóm theo chức năng, và tên biến môi trường
   là vai trò + số thứ tự trong nhóm — camera tcode thứ hai đọc ``TCODE2``. Không có
   trường nào khai tay, nên không có gì để trôi khỏi nhau.

   Thêm một camera cho một chức năng đang có = **một mục trong YAML + một dòng env**.
   Không đặt tên, không cấp phát số, không đụng docker-compose.
3. **Suy ra thứ pipeline cần** từ vai trò camera — quan trọng nhất là *camera nào được
   decode*.

⚠️ **Không phải camera nào cũng được decode, và đó là ràng buộc phần cứng chứ không phải
tối ưu.** Cả 10 camera đều 2688x1520@30. Decode hết là ~1 226 Mpixel/s ≈ 4,9 lần một luồng
4K30 — vượt trần một NVDEC của GA106 (RTX 3060). Hai vai trò ``bottom`` và
``evidence_only`` không chạy model nào, nên chúng **chỉ ghi hình**: bỏ chúng khỏi nhánh
decode là cách rẻ nhất để về trong ngân sách. Xem ``docs/HARDWARE_BUDGET.md`` §2.2.

Nhưng **mọi** camera đều được ghi hình, kể cả camera không decode — ảnh bằng chứng 6 mặt
cần chúng. Đó là lý do nhánh ghi tách ở tầng bitstream (DN-014).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.enum import CameraRole, ContainerDim, Lane

__all__ = [
    "CameraConfig",
    "ConfigError",
    "CraneConfig",
    "OcrRoi",
    "load_crane",
]


Relative = Annotated[float, Field(ge=0.0, le=1.0)]
"""Toạ độ tương đối trong khung ảnh. **Không phải pixel** — xem DN-002."""


class ConfigError(ValueError):
    """Config sai. Luôn kèm đường dẫn file và chỗ sai."""


class ShapeParams(BaseModel):
    """Tham số tiền xử lý cho MỘT hình dạng mã trên một vùng.

    Vùng dùng chung, tham số thì không: ``ccode_det_h`` và ``ccode_det_v`` nhận kích thước
    khác nhau và cần nới khác nhau. Đo trên v1, cùng một vùng: ``input_size`` 800x992
    (ngang) so 480x608 (dọc).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_size: tuple[int, int]
    """``(cao, rộng)`` đưa vào detector. Thứ tự này ngược ``cv2.resize`` — dễ nhầm."""

    expand_ratio: tuple[float, float] = (1.0, 1.0)
    """``(rộng, cao)``. Nới vùng trước khi cắt — khác nhau theo từng vùng VÀ từng hình
    dạng; đo trên v1 thấy từ 1,0/1,1 tới 1,3/1,15."""

    @field_validator("input_size")
    @classmethod
    def _positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"input_size phải dương, nhận {v}")
        return v


class OcrRoi(BaseModel):
    """Một vùng OCR tĩnh trên camera ``ccode``: cắt ở đâu, và vùng đó **nghĩa là gì**.

    Cả hai nằm ở đây, và đây là config của ds_app — không tách sang rule. Lý do là hợp
    đồng message: :class:`~common.message.OcrResult` mang sẵn ``lane`` và ``cont_dim``, nên
    **probe của ds_app phải biết chúng để điền**. Để chúng ở tầng rule thì ds_app hoặc phải
    đọc ngược config của rule, hoặc không điền nổi message — cả hai đều tệ hơn.

    Thứ duy nhất KHÔNG ở đây là ngưỡng chấp nhận (``ocr_threshold`` của rule ``CCODE01``):
    nó là bộ lọc áp *sau* khi đã đọc xong, nên probe cứ phát mọi kết quả kèm confidence và
    rule quyết định. Đo trên v1: ngưỡng giống hệt nhau (0,95) ở cả 8 vùng, nên nó là một
    giá trị của rule chứ không phải thuộc tính của vùng.

    Vùng là **tĩnh, khai trong config**, không phải đầu ra của detector — đó là lý do nhánh
    ccode dùng ``nvdspreprocess`` (nó nhận ROI theo từng nguồn) thay vì để PGIE tự tìm vùng.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane: Lane
    """Vùng này phủ làn nào. Đi thẳng vào ``OcrResult.lane``."""

    cont_dim: ContainerDim
    """Kích thước container vùng này ứng với. Đi thẳng vào ``OcrResult.cont_dim``."""

    roi: tuple[Relative, Relative, Relative, Relative]
    """``(x1, y1, x2, y2)`` tương đối."""

    shapes: dict[Literal["horizontal", "vertical"], ShapeParams] = Field(min_length=1)
    """Hình dạng mã chạy trên vùng này → tham số của nó. Chọn cặp model
    ``ccode_{det,rec}_{h,v}``.

    **Một vùng, nhiều hình dạng.** v1 khai mỗi (vùng, hình dạng) thành một mục riêng, nên
    cùng một toạ độ bị chép hai lần — hai bản có thể trôi khỏi nhau, và đã trôi: ở
    ``..._1508`` lane 1 / 20 feet, bản ngang là (0,138)-(535,720) còn bản dọc là
    (0,161)-(560,683). v2 gộp thành **hợp** của hai vùng, nên không mất diện tích mà model
    nào đang có.

    Rỗng là lỗi: một vùng không chạy hình dạng nào chỉ tốn một lần cắt ảnh."""

    @field_validator("roi")
    @classmethod
    def _ordered(cls, v: tuple[float, float, float, float]) -> tuple[float, ...]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"roi lật ngược hoặc rỗng: {v}")
        return v


class CameraConfig(BaseModel):
    """Một camera của một cẩu."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crane_id: str = ""
    """Mã cẩu, **được bơm xuống** từ :class:`CraneConfig` lúc load — không khai trong YAML.

    Camera cần biết cẩu của nó để dựng :attr:`code`. Bơm xuống thay vì bắt khai lại: khai
    hai chỗ là hai chỗ có thể lệch nhau."""

    role: CameraRole
    """Vai trò, **bơm xuống từ khoá nhóm** trong ``cameras:`` — không khai trong thân."""

    index: int = Field(default=1, ge=1)
    """Vị trí trong nhóm cùng vai trò, đếm từ 1. **Bơm xuống** theo thứ tự khai báo."""

    desc: str = ""
    """Mô tả cho người đọc ("Mặt phải trước"). **Không phải định danh** — đừng dùng nó để
    khớp dữ liệu; nó đổi khi ai đó sửa cho dễ hiểu hơn. Để trống được."""

    stream: str
    """URL RTSP **không kèm credential**: ``rtsp://113.160.225.15:1508//CH001.sdp``.

    Toàn bộ định danh của luồng nằm ở đây, trong config — host, cổng, path. Đó là thứ
    quyết định :attr:`code`, nên mã camera đọc được từ chính file này mà không cần biến môi
    trường nào. Trước đây URL nằm ở env và hệ quả là mã camera **không tái tạo được** khi
    review một diff hay chạy CI.

    Credential thì KHÔNG ở đây: nó là bí mật, không phải cấu hình. Xem
    :attr:`CraneConfig.rtsp_credential`.

    **Một luồng cho cả ghi lẫn model.** Nhánh ghi tách ở tầng bitstream từ chính luồng này
    (DN-014), nên không cần luồng thứ hai. Mở luồng riêng cho nhánh model là nhân đôi số
    kết nối RTSP (10 → 20) chỉ để đổi lấy độ phân giải khác — chưa có nhu cầu đó, và khi
    có thì thêm một trường ở đây, đừng thêm một trường luôn rỗng để chờ.
    """

    @field_validator("stream")
    @classmethod
    def _no_credential(cls, v: str) -> str:
        """URL trong config không được mang credential — file này nằm trong git."""
        url = v.strip()
        if "${" in url:
            raise ValueError(f"stream còn tham chiếu chưa nội suy: {url!r}")
        if not url.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"stream phải bắt đầu bằng rtsp:// hoặc rtsps://, nhận {url!r}")
        if "@" in url:
            raise ValueError(
                f"stream chứa credential: {url!r}. Bỏ phần `user:pass@` ra — config nằm "
                f"trong git. Credential đặt ở `rtsp_credential` cấp cẩu, lấy từ env."
            )
        for junk in ("|", " ", "\t"):
            if junk in url:
                raise ValueError(
                    f"stream chứa ký tự {junk!r} — gần như chắc chắn là trích thiếu từ một "
                    f"định dạng có phân tách: {url!r}"
                )
        return url

    code: str = ""
    """Mã camera, **sinh ra rồi kiểm lại** — không viết tay.

    ``make codes`` ghi nó vào YAML để mã hiện ngay trong file, cạnh camera nó thuộc về; đó
    là thứ các service khác khoá config theo (``configs/rules/<cẩu>/<rule>/config.json``).
    Nhưng để trống thì nó vẫn tự suy từ :attr:`stream`, và khai **sai** thì load báo lỗi:
    hiện ra được nhưng không trôi được.
    """

    credential: str = ""
    """``user:pass``, **bơm xuống** từ :class:`CraneConfig` lúc load. Không khai trong YAML."""

    @model_validator(mode="after")
    def _code_matches_stream(self) -> CameraConfig:
        derived = self._derive_code()
        if self.code and self.code != derived:
            raise ValueError(
                f"camera {self.key!r}: code khai {self.code!r} nhưng URL cho {derived!r}.\n"
                f"   Mã suy từ crane_id + host + cổng — sửa `stream` hoặc chạy `make codes`.\n"
                f"   Để lệch thì các service khác khoá config theo một mã không tồn tại."
            )
        if not self.code:
            object.__setattr__(self, "code", derived)
        return self

    def _derive_code(self) -> str:
        host = self.stream.split("://", 1)[1].split("/")[0]
        ip, _, port = host.partition(":")
        parts = [self.crane_id, ip.replace(".", "_")]
        if port:
            parts.append(port)
        return "_".join(p for p in parts if p)

    @property
    def rtsp_record(self) -> str:
        """URL đầy đủ để kết nối — :attr:`stream` có chèn credential."""
        if not self.credential:
            return self.stream
        scheme, rest = self.stream.split("://", 1)
        return f"{scheme}://{self.credential}@{rest}"

    @property
    def key(self) -> str:
        """Tên ngắn của camera: vai trò + số thứ tự trong vai trò. ``tcode2``, ``ccode5``.

        Đây là thứ người và CLI dùng (``--cam tcode2``). Nó nói đúng cái người vận hành
        quan tâm — *camera này làm việc gì* — chứ không phải một số tự cấp phát.
        """
        return f"{self.role.value}{self.index}"

    model_fps: float | None = Field(default=None, gt=0.0)
    """Nhịp khung đưa vào nhánh model. ``None`` = giữ nguyên fps nguồn.

    Đây là nhịp mà **rule cần** (HARDWARE_BUDGET §2.7), không phải nhịp nguồn. ds_app quy
    ra ``drop-frame-interval`` của decoder — xem :attr:`drop_frame_interval`.

    ⚠️ **Không giảm được tải NVDEC.** Nguồn là IPPP nên mọi khung vẫn phải giải mã; cái
    này vứt output *sau* decode. Thứ nó tiết kiệm là mọi thứ phía sau: gộp batch ở
    ``nvstreammux``, copy buffer, request gửi Triton, công việc trong probe, message lên
    Kafka. Không đặt thì tải suy luận là fps **nguồn** (30) chứ không phải fps mục tiêu
    (5) — gấp 6 lần, và trần Triton mới chỉ đo trên máy dev.

    Không đặt cho camera không chạy model: nhánh decode của chúng không ai kéo."""

    ocr_rois: list[OcrRoi] = Field(default_factory=list)
    """Vùng OCR tĩnh. Chỉ camera ``ccode``.

    ⚠️ **Vùng làn (`laneN_zone`) KHÔNG nằm ở đây** — đó là config của rule
    (``configs/rules/<cẩu>/``), ds_app không đọc nó. Khác nhau ở chỗ: vùng OCR quyết định
    ds_app cắt gì đưa cho Triton, còn vùng làn chỉ dùng để suy ra xe đang ở làn nào."""

    @property
    def drop_frame_interval(self) -> int:
        """Giá trị ``drop-frame-interval`` cho decoder. ``0`` = không bỏ khung nào.

        Ngữ nghĩa của property (đọc từ chính element): ``N`` nghĩa là **giữ 1 khung mỗi N
        khung**. Nên chia fps nguồn cho fps mục tiêu, làm tròn.

        Làm tròn nghĩa là nhịp thật hiếm khi đúng bằng ``model_fps``:
        :attr:`effective_fps` cho số thật, và ds_app in nó ra lúc dựng pipeline để chênh
        lệch nhìn thấy được thay vì âm thầm.
        """
        if self.model_fps is None or self.source_fps <= 0:
            return 0
        return max(1, round(self.source_fps / self.model_fps))

    @property
    def effective_fps(self) -> float:
        """Nhịp THẬT sau khi làm tròn. Bằng fps nguồn nếu không giảm nhịp."""
        n = self.drop_frame_interval
        return self.source_fps if n <= 1 else self.source_fps / n

    source_fps: float = Field(default=0.0, ge=0.0)
    """fps của nguồn camera này.

    Mặc định lấy từ :attr:`CraneConfig.source_fps`; **khai trong dòng camera để ghi đè**
    khi camera đó chạy nhịp khác. Đo 2026-09-02: camera ``..._1517`` phát 18 fps còn mọi
    camera khác 30 — dùng chung một số làm ``drop_frame_interval`` của nó lệch 40 %, và
    ``PerceptionMessage.fps`` báo sai ra ngoài. Xem ``docs/HARDWARE_BUDGET.md`` §6.3."""

    @model_validator(mode="after")
    def _role_consistency(self) -> CameraConfig:
        if self.model_fps is not None and not self.decodes:
            raise ValueError(
                f"camera {self.key!r} vai trò {self.role} không chạy model nhưng khai "
                f"model_fps — nhánh decode của nó không ai kéo, đặt gì cũng vô nghĩa"
            )
        if self.model_fps is not None and self.source_fps and self.model_fps > self.source_fps:
            raise ValueError(
                f"camera {self.key!r}: model_fps={self.model_fps} lớn hơn fps nguồn "
                f"({self.source_fps}) — không tạo thêm khung được"
            )
        if self.ocr_rois and self.role is not CameraRole.CCODE:
            raise ValueError(
                f"camera {self.key!r} vai trò {self.role} nhưng khai ocr_rois; "
                f"chỉ vai trò 'ccode' mới có vùng OCR"
            )
        return self

    @property
    def decodes(self) -> bool:
        """Camera này có đi vào nhánh model (tức có tốn NVDEC) không.

        ``bottom`` và ``evidence_only`` trả ``False``: chúng chỉ ghi hình. Xem docstring
        module về ngân sách NVDEC."""
        return self.role.runs_model


class CraneConfig(BaseModel):
    """Cấu hình một cẩu — nội dung một file ``configs/cranes/<id>.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crane_id: str = Field(min_length=1)
    berth_no: str = Field(min_length=1)
    num_lane: int = Field(ge=1, le=9)
    source_fps: float = Field(default=30.0, gt=0.0)
    """fps của **nguồn**, đo được — dùng để quy ``model_fps`` ra ``drop-frame-interval``.

**Vì sao phải khai thay vì tự dò** — hai lý do độc lập, cả hai đã kiểm chứng:

    * ``nvv4l2decoder.drop-frame-interval`` khai ``changeable only in NULL or READY
      state`` (đọc thẳng từ ``gst-inspect``), nên nhịp phải quyết TRƯỚC khi khung đầu tiên
      về — không thể chờ rồi mới tính.
    * Caps của nguồn khai ``framerate=0/1`` trên **cả 10 camera GC03** (đo 2026-09-02), tức
      "biến thiên, tự đo lấy". Không có gì để đọc.

    ds_app **đo lại lúc chạy** và báo nếu lệch — xem ``ds_app/src/pipeline/ratecheck.py``.
    Đó là hậu kiểm, không phải tự dò: nhịp model vẫn tính theo con số khai ở đây.

    Đây là **mặc định cho mọi camera**; camera nào chạy nhịp khác thì khai ``source_fps``
    ngay trong dòng của nó.

    Đo 2026-08-29: cả 10 camera GC03 là 30 fps. Đo lại 2026-09-02 thì ``..._1517`` chỉ còn
    **18 fps** — đếm từ bảng sample của chính đoạn ghi passthrough, nên đó là bitstream
    thật đã tới, không phải suy đoán. Xem HARDWARE_BUDGET §6.3.
    """

    rtsp_credential: str = ""
    """``user:pass`` dùng chung cho mọi camera của cẩu, **lấy từ env lúc load**.

    Đây là thứ DUY NHẤT không nằm trong file này, vì nó là bí mật chứ không phải cấu hình.
    Mọi thứ định danh luồng — host, cổng, path — nằm trong ``stream`` của từng camera, nên
    ``camera_code`` đọc được từ chính config mà không cần biến môi trường nào.

    Cả 10 camera GC03 dùng chung một credential (đo 2026-09-01). Nếu về sau có camera cần
    credential riêng thì thêm trường ở tầng camera; đừng đưa cả URL ngược lại vào env.
    """

    cameras: dict[CameraRole, list[CameraConfig]] = Field(min_length=1)
    """Camera **nhóm theo chức năng**: ``cameras.tcode`` là danh sách camera làm việc tcode.

    Hình dạng này chọn theo câu hỏi hay gặp nhất khi vận hành — *"thêm một camera nữa cho
    tính năng này"*. Câu trả lời là thêm một mục vào đúng danh sách, không phải đặt tên,
    không phải cấp phát số, không phải đụng file nào khác.

    Khoá nhóm là :class:`CameraRole`, nên gõ sai vai trò là lỗi lúc load kèm danh sách vai
    trò hợp lệ — không phải một nhóm rỗng bị bỏ qua im lặng.
    """

    @model_validator(mode="before")
    @classmethod
    def _stamp_identity(cls, data: Any) -> Any:
        """Bơm ``crane_id``, ``role`` và ``index`` xuống từng camera.

        Cả ba là trường **dẫn xuất** từ vị trí trong file. Ghi đè bất cứ giá trị nào có
        sẵn: để người dùng khai đè nghĩa là mở đường cho một camera tự nhận thuộc cẩu khác,
        hoặc tự nhận vai trò khác vai trò của nhóm chứa nó — và khi đó nó đọc URL của
        camera khác.
        """
        if isinstance(data, dict) and isinstance(data.get("cameras"), dict):
            crane_id = data.get("crane_id", "")
            for role, group in data["cameras"].items():
                if not isinstance(group, list):
                    continue
                for i, cam in enumerate(group, start=1):
                    if isinstance(cam, dict):
                        cam["crane_id"] = crane_id
                        cam["role"] = role
                        cam["index"] = i
                        cam["credential"] = data.get("rtsp_credential", "")
                        # Camera khai riêng thì GIỮ: `source_fps` của cẩu chỉ là mặc
                        # định. Đo 2026-09-02 thấy camera 1517 phát 18 fps trong khi mọi
                        # camera khác 30 — một giá trị chung không diễn đạt nổi điều đó, và
                        # hệ quả là `drop_frame_interval` sai 40 % cho camera đó.
                        cam.setdefault("source_fps", data.get("source_fps", 30.0))

        # Vùng OCR khai ở mục RIÊNG, khoá theo mã camera — không nhét vào dòng camera.
        #
        # Lý do là hình thức, và nó có giá trị thật: một camera = MỘT dòng, đếm được bằng
        # mắt, và một diff đổi camera nào thì thấy ngay camera đó. Nhét 8 vùng vào trong
        # cặp ngoặc của dòng đó sẽ cho ra một dòng dài vài trăm ký tự; tách camera thành
        # khối nhiều dòng thì mất luôn tính chất kia. Mục riêng giữ được cả hai: mỗi camera
        # một dòng, mỗi vùng một dòng.
        #
        # Cùng khuôn với config rule (`configs/rules/<cẩu>/<rule>/config.json`), vốn cũng
        # khoá theo mã camera — nên chỗ nào cần tra theo camera thì tra cùng một kiểu.
        # `pop`, không phải `get`: `CraneConfig` là `extra="forbid"`, nên khoá này
        # phải BIẾN MẤT sau khi đã bơm xuống camera.
        rois = data.pop("ocr_rois", None) or {}
        if rois:
            known = {
                cam["code"]
                for group in data["cameras"].values()
                if isinstance(group, list)
                for cam in group
                if isinstance(cam, dict) and "code" in cam
            }
            unknown = set(rois) - known
            if unknown:
                # Gõ sai mã camera ở đây sẽ làm vùng biến mất không dấu vết, và camera đó
                # chạy OCR trên không có vùng nào.
                raise ValueError(
                    f"ocr_rois khai cho camera không tồn tại: {sorted(unknown)}; "
                    f"mã hợp lệ: {sorted(known)}"
                )
            for group in data["cameras"].values():
                if not isinstance(group, list):
                    continue
                for cam in group:
                    if isinstance(cam, dict) and cam.get("code") in rois:
                        cam["ocr_rois"] = rois[cam["code"]]
        return data

    @model_validator(mode="after")
    def _consistent(self) -> CraneConfig:
        codes = [c.code for c in self.record_cameras]
        if len(codes) != len(set(codes)):
            dup_codes = sorted({c for c in codes if codes.count(c) > 1})
            raise ValueError(
                f"mã camera trùng nhau: {dup_codes}. Mã suy từ host+cổng của URL, nên trùng "
                f"nghĩa là hai camera cùng một điểm cuối — hoặc URL sai. Để nguyên thì dữ "
                f"liệu của camera này bị gán cho camera kia mà không có gì báo."
            )

        # Không có camera nào decode nghĩa là cấu hình này không sinh ra suy luận nào —
        # gần như chắc chắn là lỗi gõ vai trò, và nó sẽ biểu hiện thành "hệ chạy mà không
        # bao giờ ra kết quả", loại lỗi tốn nhiều giờ nhất để lần.
        if not any(c.decodes for c in self.record_cameras):
            raise ValueError(
                "không camera nào chạy model — kiểm lại các nhóm trong `cameras:`; "
                f"đang có: {sorted(r.value for r in self.cameras)}"
            )
        return self

    @property
    def model_cameras(self) -> list[CameraConfig]:
        """Camera đi vào nhánh model, theo thứ tự khai báo.

        Thứ tự này là **chỉ số nguồn của ``nvstreammux``**, nên nó phải ổn định: đổi thứ
        tự trong YAML là đổi ``pad_index``, và probe dùng chỉ số đó để biết khung thuộc
        camera nào."""
        return [c for c in self.record_cameras if c.decodes]

    @property
    def record_cameras(self) -> list[CameraConfig]:
        """Camera được ghi hình — **tất cả**. Ảnh bằng chứng 6 mặt cần cả camera không decode."""
        return [cam for group in self.cameras.values() for cam in group]

    def by_role(self, role: CameraRole) -> list[CameraConfig]:
        """Mọi camera làm một chức năng, theo thứ tự khai báo. Không có thì trả danh sách rỗng."""
        return list(self.cameras.get(role, ()))

    def camera(self, key: str) -> CameraConfig:
        """Camera theo tên ngắn (``tcode2``). Không có thì báo kèm danh sách đang có."""
        for cam in self.record_cameras:
            if cam.key == key:
                return cam
        known = ", ".join(c.key for c in self.record_cameras)
        raise KeyError(f"cẩu {self.crane_id} không có camera {key!r}; đang có: {known}") from None


_CRED_ENV = "CRANEOPS_RTSP_CRED"


def _inject_credential(raw: Any, *, env: Mapping[str, str]) -> None:
    """Bơm credential RTSP từ môi trường vào config đã đọc.

    Chỉ MỘT biến cho cả cẩu. Bản trước có một biến cho mỗi camera, và hệ quả là
    ``camera_code`` — thứ các service khác khoá config theo — **không tái tạo được** nếu
    không có file env: CI không xác thực nổi config đã commit, và người review một diff
    không biết mã nào ứng với camera nào.
    """
    if not isinstance(raw, dict):
        return
    if raw.get("rtsp_credential"):
        return  # đã khai tường minh (test); không đụng
    cred = env.get(_CRED_ENV, "").strip()
    if cred:
        raw["rtsp_credential"] = cred
    # Không có credential thì vẫn load được: `camera_code`, vai trò, vùng OCR đều đọc được
    # mà không cần nó. Chỉ lúc thật sự kết nối RTSP mới thiếu — và lúc đó GStreamer báo rõ.


def load_crane(path: str | Path, *, env: Mapping[str, str] | None = None) -> CraneConfig:
    """Đọc và validate một file cấu hình cẩu.

    Args:
        path: Đường dẫn tới ``configs/cranes/<id>.yaml``.
        env: Nguồn biến môi trường; mặc định ``os.environ``. Truyền dict để test.

    Raises:
        ConfigError: file không đọc được, YAML hỏng, thiếu biến môi trường, hoặc nội dung
            không hợp lệ. Thông báo luôn kèm đường dẫn file.
    """
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"không đọc được {p}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: YAML hỏng — {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: nội dung phải là một ánh xạ, nhận {type(raw).__name__}")

    src = os.environ if env is None else env
    _inject_credential(raw, env=src)
    try:
        return CraneConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
