"""Cấu hình cẩu — nguồn sự thật cho `ds_app` và mọi service đọc `configs/cranes/*.yaml`.

Ba việc file này làm, và **chỉ** ba việc đó (không I/O ngoài đọc YAML, không dựng
pipeline, không biết gì về GStreamer):

1. **Validate fail-fast.** Gõ sai một khoá là lỗi lúc load, không phải một `.get(k, default)`
   im lặng lúc chạy. ``extra="forbid"`` cho **config** — ngược với message contract, nơi
   dùng ``extra="ignore"`` để chịu được nâng cấp lệch pha (xem ``common/message.py``).
2. **Nội suy secret từ môi trường.** URL RTSP có mật khẩu nên **không bao giờ** nằm trong
   YAML. YAML viết ``${CAM01_RTSP}``; giá trị đến từ env lúc load.
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
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.enum import CameraRole, Lane

__all__ = [
    "CameraConfig",
    "ConfigError",
    "CraneConfig",
    "OcrRoi",
    "load_crane",
]

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

Relative = Annotated[float, Field(ge=0.0, le=1.0)]
"""Toạ độ tương đối trong khung ảnh. **Không phải pixel** — xem DN-002."""


class ConfigError(ValueError):
    """Config sai. Luôn kèm đường dẫn file và chỗ sai."""


class OcrRoi(BaseModel):
    """Một vùng OCR tĩnh trên camera ``ccode``.

    Vùng là **tĩnh, khai trong config**, không phải đầu ra của detector — đó là lý do
    nhánh ccode dùng ``nvdspreprocess`` (nó nhận ROI theo từng nguồn) thay vì để PGIE tự
    tìm vùng.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    shape: str = Field(pattern="^(horizontal|vertical)$")
    """Mã container nằm ngang hay dọc — chọn cặp model ``ccode_{det,rec}_{h,v}``."""

    lane: Lane
    roi: tuple[Relative, Relative, Relative, Relative]
    """``(x1, y1, x2, y2)`` tương đối."""

    input_size: tuple[int, int] = Field()
    """``(cao, rộng)`` đưa vào detector. Thứ tự này ngược ``cv2.resize`` — dễ nhầm."""

    expand_ratio: tuple[float, float] = (1.0, 1.0)
    ocr_threshold: Relative = 0.95

    @field_validator("roi")
    @classmethod
    def _ordered(cls, v: tuple[float, float, float, float]) -> tuple[float, ...]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"roi lật ngược hoặc rỗng: {v}")
        return v

    @field_validator("input_size")
    @classmethod
    def _positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"input_size phải dương, nhận {v}")
        return v


class CameraConfig(BaseModel):
    """Một camera của một cẩu."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1)
    name: str
    role: CameraRole

    rtsp_record: str
    """Nguồn RTSP. Trong YAML là ``${CAM01_RTSP}``; :func:`load_crane` thay bằng giá trị env."""

    @field_validator("rtsp_record", "rtsp_model")
    @classmethod
    def _looks_like_rtsp(cls, v: str | None) -> str | None:
        """URL phải là URL, không phải một dòng cấu hình bị cắt dở.

        ⚠️ Đã xảy ra: URL trích từ một định dạng phân tách bằng ``|`` mà quên dừng ở dấu
        phân tách, cho ra ``rtsp://host:1508//CH001.sdp|h265|10|||``. GStreamer **không**
        báo lỗi — nó giữ nguyên phần thừa trong path và gửi ``SETUP //CH001.sdp|h265|10|||``
        cho camera. Camera đang dùng tình cờ bỏ qua phần thừa nên mọi thứ vẫn chạy; một
        firmware khác sẽ trả 404, và lúc đó không có gì trỏ về nguyên nhân.
        """
        if v is None:
            return None
        url = v.strip()
        # Kiểm tham chiếu chưa nội suy TRƯỚC: nó là chẩn đoán cụ thể nhất. Một `${VAR}`
        # trần cũng trượt kiểm scheme, và khi đó thông báo "phải bắt đầu bằng rtsp://" chỉ
        # tổ làm người đọc đi sửa nhầm chỗ.
        if "${" in url:
            raise ValueError(
                f"URL còn tham chiếu chưa nội suy: {url!r} — biến môi trường không được nạp?"
            )
        if not url.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"URL phải bắt đầu bằng rtsp:// hoặc rtsps://, nhận {url!r}")
        for junk in ("|", " ", "\t"):
            if junk in url:
                raise ValueError(
                    f"URL chứa ký tự {junk!r} — gần như chắc chắn là trích thiếu từ một "
                    f"định dạng có phân tách: {url!r}"
                )
        return url

    rtsp_model: str | None = None
    """Luồng riêng cho nhánh model. ``None`` (mặc định) ⇒ **tee từ ``rtsp_record``**, chỉ
    MỘT kết nối RTSP cho cả ghi lẫn model. Chỉ đặt khi camera có sub-stream và ta thật sự
    muốn model chạy ở độ phân giải khác — mỗi giá trị khác ``None`` là thêm một kết nối
    RTSP nữa, nhân với 10 camera thì đó là 20 kết nối."""

    lane_zones: dict[Lane, list[tuple[Relative, Relative]]] = Field(default_factory=dict)
    """Vùng lane dạng đa giác, toạ độ tương đối.

    Khai **theo từng camera** chứ không suy từ một bộ đường chung: camera 3 và camera 10
    nhìn cùng khu vực từ hai hướng nên vùng của chúng ngược chiều nhau. Xem DN-002."""

    ocr_rois: list[OcrRoi] = Field(default_factory=list)

    @model_validator(mode="after")
    def _role_consistency(self) -> CameraConfig:
        if self.ocr_rois and self.role is not CameraRole.CCODE:
            raise ValueError(
                f"camera {self.id} vai trò {self.role} nhưng khai ocr_rois; "
                f"chỉ vai trò 'ccode' mới có vùng OCR"
            )
        if self.lane_zones and not self.role.runs_model:
            raise ValueError(
                f"camera {self.id} vai trò {self.role} không chạy model nhưng khai "
                f"lane_zones — vùng lane sẽ không bao giờ được dùng"
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
    cameras: list[CameraConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> CraneConfig:
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"id camera trùng: {dup}")

        for cam in self.cameras:
            for lane in cam.lane_zones:
                if int(lane.value) > self.num_lane:
                    raise ValueError(
                        f"camera {cam.id} khai lane {lane.value} nhưng cẩu chỉ có "
                        f"{self.num_lane} lane"
                    )
        # Không có camera nào decode nghĩa là cấu hình này không sinh ra suy luận nào —
        # gần như chắc chắn là lỗi gõ vai trò, và nó sẽ biểu hiện thành "hệ chạy mà không
        # bao giờ ra kết quả", loại lỗi tốn nhiều giờ nhất để lần.
        if not any(c.decodes for c in self.cameras):
            raise ValueError(
                "không camera nào chạy model — kiểm lại trường `role`; "
                f"đang có: {sorted({c.role.value for c in self.cameras})}"
            )
        return self

    @property
    def model_cameras(self) -> list[CameraConfig]:
        """Camera đi vào nhánh model, theo thứ tự khai báo.

        Thứ tự này là **chỉ số nguồn của ``nvstreammux``**, nên nó phải ổn định: đổi thứ
        tự trong YAML là đổi ``pad_index``, và probe dùng chỉ số đó để biết khung thuộc
        camera nào."""
        return [c for c in self.cameras if c.decodes]

    @property
    def record_cameras(self) -> list[CameraConfig]:
        """Camera được ghi hình — **tất cả**. Ảnh bằng chứng 6 mặt cần cả camera không decode."""
        return list(self.cameras)

    def camera(self, cam_id: int) -> CameraConfig:
        try:
            return next(c for c in self.cameras if c.id == cam_id)
        except StopIteration:
            known = ", ".join(str(c.id) for c in self.cameras)
            raise KeyError(
                f"cẩu {self.crane_id} không có camera {cam_id}; đang có: {known}"
            ) from None


def _resolve_env(value: Any, *, where: str, env: Mapping[str, str]) -> Any:
    """Thay ``${TÊN}`` bằng giá trị môi trường.

    Chỉ nhận **toàn bộ** chuỗi là một tham chiếu, không nội suy giữa chuỗi: một URL RTSP
    ghép từ nhiều mảnh là cách dễ nhất để lộ mật khẩu vào log khi chỉ một mảnh thiếu.
    """
    if not isinstance(value, str):
        return value
    match = _ENV_REF.match(value.strip())
    if match is None:
        return value
    name = match.group(1)
    resolved = env.get(name)
    if not resolved:
        raise ConfigError(
            f"{where}: cần biến môi trường {name} (config ghi '{value}') nhưng nó trống. "
            f"Secret không nằm trong YAML — đặt {name} trong file env của service."
        )
    return resolved


def _walk(node: Any, *, where: str, env: Mapping[str, str]) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, where=f"{where}.{k}", env=env) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, where=f"{where}[{i}]", env=env) for i, v in enumerate(node)]
    return _resolve_env(node, where=where, env=env)


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

    resolved = _walk(raw, where=str(p), env=os.environ if env is None else env)
    try:
        return CraneConfig.model_validate(resolved)
    except ValueError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
