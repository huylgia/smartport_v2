"""Model config của từng rule — **nguồn sự thật** cho schema JSON và giá trị mặc định.

Mỗi rule một model. Nó quyết định ba thứ, và cả ba đều sinh ra từ đây chứ không viết tay:

* nội dung hợp lệ của ``configs/rules/<CẨU>/<RULE>/config.json``
* ``schema.json`` cạnh nó (``craneops-rules schema``)
* khung config mặc định cho một cẩu mới (``craneops-rules init <CẨU>``)

Ranh giới với config ds_app: xem docstring ``common/rule_config.py``. Ngắn gọn — thứ gì
chỉnh được mà **không** dựng lại pipeline thì thuộc về đây.

⚠️ Các model ở đây **chưa có rule đi kèm** (Phase 4-5). Chúng tồn tại vì dữ liệu chúng mô
tả đã có thật: vùng lane đo được cho GC03. Để dữ liệu đó nằm trong config ds_app thì mỗi
lần chỉnh một đa giác là một lần cả 10 camera ngừng ghi hình.

Số mặc định lấy từ ``docs/RULES.md``. Đổi ở đây thì sửa cả bảng bên đó — hai chỗ trôi khỏi
nhau thì bản ít được chạy hơn là bản người ta tin.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)

from common.enum import CameraRole, Lane
from internal.pkg.geometry import Anchor

__all__ = [
    "RULE_CONFIGS",
    "CCode01Config",
    "Crane01Config",
    "Crane02Config",
    "RuleSpec",
    "TCode01Config",
]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Relative = Annotated[float, Field(ge=0.0, le=1.0)]

Polygon = list[tuple[Relative, Relative]]
"""Một đa giác, toạ độ tương đối."""


class _LaneZones(BaseModel):
    """Vùng của từng làn — **một làn đúng một đa giác**.

    Ba trường phẳng chứ không phải ánh xạ ``{"1": …}``: ``Lane`` là tập đóng đúng ba giá
    trị, nên ba trường nói đúng điều đó. Ánh xạ thì gợi ý một quan hệ nhiều-nhiều không có
    thật, và cho phép khai làn "4" mà chỉ bắt được bằng một phép kiểm riêng.

    Khai **theo từng camera** chứ không suy từ một bộ đường chung: camera tcode và camera
    crane nhìn cùng khu vực từ hai hướng nên vùng của chúng ngược chiều nhau. Đa giác còn
    trả lời được "nằm ngoài mọi làn" — với hai đường phân chia thì mọi điểm đều rơi vào một
    dải. Xem DN-002.

    Để trống một làn nghĩa là camera này không nhìn thấy làn đó.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane1_zone: Polygon = Field(default_factory=list)
    lane2_zone: Polygon = Field(default_factory=list)
    lane3_zone: Polygon = Field(default_factory=list)

    def zones(self) -> dict[Lane, Polygon]:
        """Các làn **có** vùng, dạng ánh xạ — cho ``LaneZones.from_config``."""
        pairs = (
            (Lane.ONE, self.lane1_zone),
            (Lane.TWO, self.lane2_zone),
            (Lane.THREE, self.lane3_zone),
        )
        return {lane: poly for lane, poly in pairs if poly}


class _RuleConfig(BaseModel):
    """Nền chung. ``extra="forbid"``: gõ sai một khoá là lỗi lúc load, không phải mặc định
    im lặng — cùng lý do như config ds_app."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _anchor_from_name(v: Anchor | str | int) -> Anchor:
    """Nhận ``"CENTER"``, ``"CENTER|BOTTOM"`` — không nhận số trần.

    ``Anchor`` là ``Flag`` nên mặc định nó vào JSON dưới dạng số nguyên, và ``1`` thì
    người sửa config không đoán được là gì. Đọc/ghi bằng tên để file tự giải thích.
    """
    if isinstance(v, Anchor):
        return v
    if isinstance(v, int):
        raise ValueError(f"lane_anchor phải là tên ({_anchor_names()}), không phải số {v}")
    out = Anchor(0)
    for part in str(v).split("|"):
        name = part.strip().upper()
        member = Anchor.__members__.get(name)
        if member is None:
            raise ValueError(f"lane_anchor không hợp lệ: {part!r}; nhận: {_anchor_names()}")
        out |= member
    if not out:
        raise ValueError("lane_anchor rỗng")
    return out


def _anchor_names() -> str:
    return " | ".join(Anchor.__members__)


def _anchor_to_name(v: Anchor) -> str:
    return "|".join(m.name for m in Anchor if m in v and m.name)


AnchorName = Annotated[
    Anchor,
    BeforeValidator(_anchor_from_name),
    PlainSerializer(_anchor_to_name, return_type=str),
    WithJsonSchema({"type": "string", "examples": ["CENTER", "BOTTOM", "CENTER|BOTTOM"]}),
]
"""``Anchor`` đọc/ghi bằng tên trong config, không phải số."""


class Crane01Config(_LaneZones):
    """``CRANE01`` — gán lane từ bbox đầu kéo trên camera nhìn xuống."""

    lane_anchor: AnchorName = Anchor.CENTER
    """Điểm mốc của bbox dùng để xét nằm trong vùng nào."""

    head_thresh: Confidence = 0.6
    """Ngưỡng tin cậy tối thiểu cho bbox đầu kéo."""


class TCode01Config(_LaneZones):
    """``TCODE01`` — nhận dạng số đầu kéo (classifier tập đóng ~130 lớp)."""

    lane_anchor: AnchorName = Anchor.CENTER

    head_thresh: Confidence = 0.8
    head_code_thresh: Confidence = 0.93
    """Cao hơn ``head_thresh`` vì tập lớp đóng — đoán bừa thì rẻ, đoán sai thì đắt."""

    min_streak: int = Field(default=3, ge=1)
    """Số khung liên tiếp cùng một số xe mới phát signal."""


class Crane02Config(_RuleConfig):
    """``CRANE02`` — xe vào vị trí ổn định, và **ổn định ở đúng chỗ**.

    Hai điều kiện, không phải một:

    1. bbox đầu kéo không dịch quá ``stable_move_ratio`` trong ``stable_duration``
    2. tâm bbox nằm trong ``stop_band`` tính từ **một** mép ảnh

    Chỉ điều kiện 1 là không đủ: xe kẹt hoặc dừng chờ giữa khung hình cũng đứng yên, và nếu
    bỏ qua vị trí thì nó mở cổng OCR cho một lane không có xe nào ở đúng chỗ. Mép mà xe dừng
    lại cũng cho biết luôn chiều — xem :func:`internal.pkg.geometry.stop_side`.
    """

    stable_duration: float = Field(default=3.0, gt=0.0)
    """Giây. Đếm theo **thời gian**, không theo số khung: đếm khung trói định nghĩa "ổn
    định" vào fps của nguồn, và đổi cấu hình camera là vô tình đổi luôn ngưỡng nghiệp vụ."""

    stable_move_ratio: float = Field(default=0.02, gt=0.0)
    """Dịch chuyển tối đa, **tỉ lệ so với đường chéo bbox**. Ngưỡng pixel tuyệt đối phụ
    thuộc độ phân giải và khoảng cách xe tới camera. Xem DN-002 Q3."""

    stop_band: Relative = 0.35
    """Đầu xe khi dừng phải nằm trong dải này tính từ một mép ảnh, tương đối theo chiều
    rộng. Ngoài dải ⇒ **không phát signal**, dù bbox đứng yên."""

    head_thresh: Confidence = 0.6


class CCode01Config(_RuleConfig):
    """``CCODE01`` — nhận dạng mã container. Rule nặng nhất."""

    ocr_threshold: Confidence = 0.95
    """Ngưỡng chấp nhận một chuỗi đọc được.

    Ở đây chứ không ở từng vùng OCR: đo trên v1 thấy cả 8 vùng dùng chung 0,95, nên nó là
    một tham số hiệu chỉnh của rule chứ không phải thuộc tính của vùng. Và nó là bộ lọc áp
    **sau** khi đọc — probe cứ phát mọi kết quả kèm confidence, rule quyết định."""

    top_k: int = Field(default=5, ge=1)
    sharpness_min: float = Field(default=1000.0, ge=0.0)
    """Dưới ngưỡng này thì bỏ crop, không đưa vào OCR."""

    pair_distance_px: float = Field(default=60.0, gt=0.0)
    """Khoảng cách tâm tối đa để ghép part-1 với part-2."""

    bitmap_threshold: Confidence = 0.1
    box_threshold: Confidence = 0.2
    character_threshold: Confidence = 0.3
    iso_threshold: Confidence = 0.95
    min_streak: int = Field(default=3, ge=1)


class RuleSpec(BaseModel):
    """Một rule: mã, model config, và vai trò camera nó tiêu thụ."""

    model_config = ConfigDict(frozen=True)

    code: str
    config_model: type[BaseModel]
    roles: tuple[CameraRole, ...]


RULE_CONFIGS: tuple[RuleSpec, ...] = (
    RuleSpec(code="CCODE01", config_model=CCode01Config, roles=(CameraRole.CCODE,)),
    RuleSpec(code="TCODE01", config_model=TCode01Config, roles=(CameraRole.TCODE,)),
    RuleSpec(code="CRANE01", config_model=Crane01Config, roles=(CameraRole.CRANE,)),
    RuleSpec(code="CRANE02", config_model=Crane02Config, roles=(CameraRole.CRANE,)),
)
"""Các rule đã có model config.

Chưa đủ 8 rule trong ``docs/RULES.md``: chỉ khai những rule mà **dữ liệu đã có thật**.
Thêm một rule ở đây là nó tự vào ``craneops-rules init`` và ``schema`` — không có danh sách
thứ hai để trôi khỏi thực tế.
"""
