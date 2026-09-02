"""Hợp đồng message giữa các service — nguồn sự thật duy nhất.

Tài liệu giải thích *vì sao*: ``docs/MESSAGE_CONTRACT.md``. File này định nghĩa *cái gì*.

Bốn quy tắc, mỗi quy tắc chặn một lớp lỗi **im lặng** — loại lỗi mà hệ vẫn chạy, vẫn trả
kết quả, chỉ là kết quả sai:

1. **Mọi message có ``schema_version``.** Không có nó thì khi nâng cấp lệch pha, không ai
   biết một consumer cũ đang đọc payload mới hay ngược lại — chỉ thấy trường bị thiếu.

2. **Validate ở CẢ producer lẫn consumer.** Một JSON Schema không được enforce lúc chạy
   chỉ là tài liệu: gõ sai key sẽ rơi im lặng vào ``.get(key, default)`` và service âm
   thầm chạy bằng giá trị mặc định. :func:`encode` và :func:`decode` không cho bỏ qua.

3. **Thời gian là trục suy từ frame**, không phải ``time.time()`` tại điểm xử lý. Xem
   ``internal/pkg/timebase.py``.

4. **``extra="ignore"`` cho message, KHÔNG phải ``extra="forbid"``.** Đây là lựa chọn có
   cân nhắc và ngược với model *config* (ở đó dùng ``forbid`` để bắt lỗi gõ sai). Message
   đi qua ranh giới process có thể nâng cấp lệch pha: nếu ``ds_app`` được cập nhật trước
   ``ruled`` và thêm một trường mới, ``forbid`` sẽ làm ``ruled`` chết hàng loạt. Bỏ qua
   trường lạ là hành vi đúng cho tương thích tiến. Trường *thiếu* hoặc *sai kiểu* vẫn báo lỗi.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.enum import (
    CameraRole,
    ContainerDim,
    ContainerPosition,
    Direction,
    IxCd,
    Lane,
    SignalKind,
    StrEnum,
)

__all__ = [
    "SCHEMA_VERSION",
    "BBox",
    "ContainerSlot",
    "ControlAction",
    "ControlMessage",
    "Detection",
    "EventMessage",
    "EvidenceJob",
    "EvidenceJobMessage",
    "EvidenceKind",
    "ManifestEntry",
    "ManifestMessage",
    "Message",
    "OcrResult",
    "PerceptionMessage",
    "Signal",
    "Topic",
    "decode",
    "encode",
    "model_for_topic",
    "perception_topic",
]

SCHEMA_VERSION = "1.0"
"""Tăng số major khi đổi/xoá một trường bắt buộc. Thêm trường tuỳ chọn thì giữ nguyên."""


class Topic(StrEnum):
    """Tên topic Kafka. Không được gõ chuỗi thô ở nơi khác."""

    PERCEPTION_CCODE = "craneops.perception.ccode"
    PERCEPTION_TCODE = "craneops.perception.tcode"
    PERCEPTION_CRANE = "craneops.perception.crane"
    SIGNALS = "craneops.signals"
    MANIFEST = "craneops.manifest"
    EVIDENCE_FAST = "craneops.evidence.fast"
    EVIDENCE_SLOW = "craneops.evidence.slow"
    EVENTS = "craneops.events"
    CONTROL = "craneops.control"


def perception_topic(role: CameraRole) -> Topic:
    """Topic perception ứng với một vai trò camera.

    Raises:
        ValueError: nếu vai trò không chạy model. ``bottom`` và ``evidence_only`` chỉ được
            ghi hình passthrough, không decode và không sinh message perception nào —
            hỏi topic của chúng gần như chắc chắn là lỗi logic ở nơi gọi.
    """
    if not role.runs_model:
        raise ValueError(
            f"vai trò {role} không chạy model nên không có topic perception; "
            f"camera loại này chỉ được ghi hình (xem CameraRole.runs_model)"
        )
    return _PERCEPTION_BY_ROLE[role]


# ---------------------------------------------------------------------------- nền


class _Msg(BaseModel):
    """Lớp nền cho mọi message. Xem quy tắc 4 ở docstring module về ``extra``."""

    model_config = ConfigDict(extra="ignore", frozen=True, use_enum_values=False)

    topic: ClassVar[Topic]

    schema_version: str = SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _known_major(cls, v: str) -> str:
        want, got = SCHEMA_VERSION.split(".")[0], v.split(".")[0]
        if want != got:
            raise ValueError(
                f"schema_version {v!r} không tương thích với {SCHEMA_VERSION!r} "
                f"(major khác nhau) — producer và consumer lệch phiên bản"
            )
        return v


Message = _Msg
T = TypeVar("T", bound=_Msg)

Timestamp = Annotated[float, Field(gt=0, description="epoch giây")]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class BBox(BaseModel):
    """Hộp bao ``[x1, y1, x2, y2]`` theo pixel, gốc toạ độ ở góc trên trái.

    Là một model chứ không phải ``list[float]``: bbox đi qua nhiều tầng hình học, và một
    hộp lật ngược (``x2 < x1``) hay âm sẽ không làm gì nổ — nó chỉ cho ra diện tích âm,
    IoU vô nghĩa, và một lane gán sai. Validator ở đây là chỗ duy nhất chặn được.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"bbox lật ngược: {self.as_tuple()}")
        return self

    @classmethod
    def from_xyxy(cls, xyxy: tuple[float, float, float, float]) -> BBox:
        return cls(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


# ---------------------------------------------------------------- perception


class Detection(BaseModel):
    """Một vật thể do PGIE/SGIE phát hiện."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    bbox: BBox
    class_name: str
    confidence: Confidence
    attrs: dict[str, float] = Field(default_factory=dict)
    """Thuộc tính phẳng từ SGIE, ví dụ ``{"headcode_12": 0.97}``.

    Phẳng, không lồng: DeepStream gắn kết quả SGIE lên chính object meta của bbox, nên
    giữ nguyên hình dạng đó thì probe chỉ việc chép ra, không phải dựng lại cấu trúc."""


class OcrResult(BaseModel):
    """Một chuỗi đọc được trong một ROI của camera ``ccode``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    roi_index: int = Field(ge=0)
    """Chỉ số vào ``ocr_rois`` của camera trong config — để truy ngược cấu hình nào sinh ra nó."""

    shape: Literal["horizontal", "vertical"]
    lane: Lane
    cont_dim: ContainerDim
    bbox: BBox
    text: str
    confidence: Confidence


class PerceptionMessage(_Msg):
    """ds_app → ruled. Một khung đã qua model."""

    topic: ClassVar[Topic] = Topic.PERCEPTION_CCODE  # ghi đè theo role, xem model_for_topic

    crane_id: str
    """Mã cẩu. Giữ riêng chứ không tách từ ``camera_code``: mã cẩu có thể chứa gạch dưới,
    và khi đó tách ngược là đoán."""

    camera_code: str = Field(min_length=1)
    """Định danh camera, dạng ``<mã cẩu>_<ip>_<cổng>`` (``GC03_113_160_225_15_1508``).

    Suy từ URL trong config (:attr:`common.config.CameraConfig.code`), không khai tay — một
    trường khai tay là một trường có thể trôi khỏi URL, và khi đó dữ liệu bị gán cho nhầm
    camera mà không có gì báo. Cùng một chuỗi được dùng làm tên thư mục ghi hình, nên
    ``segment_hint`` và trường này luôn khớp nhau."""

    role: CameraRole

    frame_id: int = Field(ge=0)
    """Chỉ số khung **gốc**, đã khôi phục qua ``restore_frame_id``. Xem timebase.

    Đọc là *"số khung nguồn này đã gửi tới được"*, **không phải** *"khung thứ mấy camera đã
    phát"*. Bộ đếm nằm ở trạng thái per-pad của ``nvstreammux``, nên nó sống sót qua cả
    việc nối lại RTSP lẫn việc dựng lại hẳn source bin — nhưng nó **đứng yên** trong lúc
    camera mất kết nối. Xem ``docs/HARDWARE_BUDGET.md`` §6.1.

    Dùng cho thứ tự và dedup **trong một trục thời gian** (xem :attr:`start_ts`); đừng dùng
    để suy ra thời gian."""

    start_ts: Timestamp
    """Mốc neo **đang có hiệu lực** cho camera này — thời điểm unix ứng với khung đầu tiên
    của trục thời gian hiện tại, không phải gốc PTS của nguồn RTSP.

    Giá trị này **đổi** khi producer phải neo lại, tức khi PTS của nguồn lùi (RTSP nối lại
    và phát PTS từ đầu). Đó là công dụng chính của trường này: hai message có ``start_ts``
    khác nhau nằm trên **hai trục thời gian khác nhau**, và ``frame_id`` của chúng không so
    sánh được với nhau. Không có trường này thì consumer không có cách nào biết điều đó.

    ⚠️ **Đừng dùng nó để tính thời gian.** ``start_ts + frame_id / fps`` chỉ đúng khi không
    khung nào mất — xem :attr:`frame_ts`."""

    fps: float = Field(gt=0)
    """FPS của **nguồn** (30 với smartport), không phải fps sau decimate."""

    frame_ts: Timestamp
    """Thời điểm khung được **chụp**, lấy từ PTS của chính khung đó.

    **Dùng trường này, đừng tự tính lại từ ``start_ts + frame_id / fps``.**

    Trong một đoạn chạy liền mạch hai cách cho kết quả y hệt — đo trên GC03: lệch
    **0,0 ms** qua 585 khung, vì ``nvstreammux`` cấp PTS đều tuyệt đối. Nhưng ``frame_id``
    đếm **số khung nhận được**, không đếm thời gian: lúc camera mất kết nối, đồng hồ chạy
    còn nó thì đứng, rồi khi có lại nó chỉ **+1** — không nhảy qua những khung không tới.
    Đo với đợt mất mạng 30 s: ``frame_num`` 65 → 66 trong khi ``frame_ts`` nhảy 30,3 s,
    nên công thức tụt lại đúng **30,000 s** và giữ nguyên khoảng đó về sau.

    Nghĩa là sai lệch **tích luỹ** theo từng lần rớt mạng. Trường này không bị vậy."""

    segment_hint: str | None = None
    """Đường dẫn segment mp4 chứa khung này — ``evidenced`` dùng để biết cắt clip ở đâu.

    Producer biết chính xác segment nào đang được ghi lúc nó xử lý khung này, nên nó nói
    ra; tìm lại bằng cách quét thư mục theo dấu thời gian là suy đoán, và suy đoán sai ở
    ranh giới giữa hai segment."""

    detections: list[Detection] = Field(default_factory=list)
    ocr: list[OcrResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ocr_only_for_ccode(self) -> PerceptionMessage:
        if self.ocr and self.role is not CameraRole.CCODE:
            raise ValueError(f"role {self.role} không được mang kết quả OCR")
        return self


# ---------------------------------------------------------------- signals


class Signal(_Msg):
    """ruled → orchestratord. Một quan sát nghiệp vụ từ một rule.

    ``kind`` là khoá mà composition spec tham chiếu trong ``configs/operations/*.yaml``.
    Nó là enum chứ không phải chuỗi tự do: gõ sai sẽ khiến một trường của sự kiện âm thầm
    không bao giờ được điền, và không có gì báo lỗi.
    """

    topic: ClassVar[Topic] = Topic.SIGNALS

    rule_code: str
    crane_id: str
    camera_code: str = Field(min_length=1)
    """Định danh camera, dạng ``<mã cẩu>_<ip>_<cổng>`` (``GC03_113_160_225_15_1508``).

    Suy từ URL trong config (:attr:`common.config.CameraConfig.code`), không khai tay — một
    trường khai tay là một trường có thể trôi khỏi URL, và khi đó dữ liệu bị gán cho nhầm
    camera mà không có gì báo. Cùng một chuỗi được dùng làm tên thư mục ghi hình, nên
    ``segment_hint`` và trường này luôn khớp nhau."""

    lane: Lane
    direction: Direction = Direction.RIGHT_TO_LEFT
    kind: SignalKind
    frame_ts: Timestamp
    confidence: Confidence = 1.0
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- manifest


class ManifestEntry(BaseModel):
    """Một container trong kế hoạch làm hàng lấy từ Oracle CATOS."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    container_no: str
    ix_cd: IxCd
    cont_dim: ContainerDim
    sztp: str = ""


class ManifestMessage(_Msg):
    """syncd → ruled. Topic **compacted**: chỉ bản mới nhất mới có ý nghĩa.

    Thay ``ContainerCodeCamera.DATABASE`` — biến class global mà ``Service`` ghi và ``crane_camera/data.py:41`` đọc, tạo phụ thuộc vòng giữa hai
    package.
    """

    topic: ClassVar[Topic] = Topic.MANIFEST

    crane_id: str
    berth_no: str
    synced_at: Timestamp
    vsl_cd: str = ""
    call_seq: str = ""
    call_year: str = ""
    containers: list[ManifestEntry] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Rỗng nghĩa là **chưa có tàu tại bến**, KHÔNG phải lỗi.

        Bến trống là trạng thái vận hành bình thường và xảy ra hàng ngày. Hệ chuyển sang
        chế độ không-đối-chiếu (combinator ``fuzzy_dedup``) và chờ — không dừng, không
        báo động.
        """
        return not self.containers


# ---------------------------------------------------------------- evidence


class EvidenceKind(StrEnum):
    IMAGE = "image"
    CLIP = "clip"
    MOSAIC = "mosaic"


class EvidenceJob(BaseModel):
    """Một việc cần làm cho một camera."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: EvidenceKind
    camera_code: str = Field(min_length=1)
    """Định danh camera, dạng ``<mã cẩu>_<ip>_<cổng>`` (``GC03_113_160_225_15_1508``).

    Suy từ URL trong config (:attr:`common.config.CameraConfig.code`), không khai tay — một
    trường khai tay là một trường có thể trôi khỏi URL, và khi đó dữ liệu bị gán cho nhầm
    camera mà không có gì báo. Cùng một chuỗi được dùng làm tên thư mục ghi hình, nên
    ``segment_hint`` và trường này luôn khớp nhau."""

    from_ts: Timestamp | None = None
    to_ts: Timestamp | None = None
    """Khoảng video cần cắt, **thời điểm tuyệt đối** (epoch giây). Chỉ ``clip`` và
    ``mosaic`` có; ``image`` thì không — nó chụp đúng một khoảnh khắc, và khoảnh khắc đó là
    :attr:`EvidenceJobMessage.anchor_ts`.

    Tuyệt đối chứ không phải độ lệch so với mốc neo, vì như vậy **một job tự đủ**: đưa nó
    cho worker không phải kèm theo mốc neo, và người đọc message không phải cộng trừ. Quy
    đổi từ cửa sổ trong ``configs/operations/*.yaml`` là việc của orchestrator, làm một lần
    ở chỗ nó đọc config — và cửa sổ phải nằm ở config chứ không phải hằng số trong code,
    vì các con số ấy được chỉnh theo thực địa.

    Nằm trong message chứ không phải hằng số trong ``evidenced``: mỗi camera một cửa sổ
    khác nhau (camera đáy cần lùi xa hơn nhiều), và cửa sổ là quyết định của orchestrator
    — nó mới biết sự kiện thuộc loại nào."""

    grid: tuple[int, int] | None = None
    count: int = Field(default=1, ge=1)

    @property
    def span(self) -> tuple[float, float]:
        """``(từ, đến)`` — đưa thẳng vào :meth:`internal.pkg.fragments.FragmentIndex.plan`.

        Raises:
            ValueError: với job ``image``, vốn không có khoảng. Dùng ``anchor_ts``.
        """
        if self.from_ts is None or self.to_ts is None:
            raise ValueError(f"job {self.kind} không có khoảng video; dùng anchor_ts")
        return self.from_ts, self.to_ts

    @model_validator(mode="after")
    def _span_matches_the_kind(self) -> EvidenceJob:
        needs_span = self.kind in (EvidenceKind.CLIP, EvidenceKind.MOSAIC)
        has_span = self.from_ts is not None and self.to_ts is not None

        if needs_span and not has_span:
            raise ValueError(f"job {self.kind.value} phải có from_ts và to_ts")
        if not needs_span and (self.from_ts is not None or self.to_ts is not None):
            # `image` chụp MỘT khoảnh khắc — khoảnh khắc đó là `anchor_ts`. Một khoảng ở
            # đây nghĩa là ai đó tưởng nó cắt video, và sẽ ngạc nhiên khi nhận một tấm ảnh.
            raise ValueError(f"job {self.kind.value} không nhận khoảng; nó chụp tại anchor_ts")
        if has_span and self.to_ts <= self.from_ts:  # type: ignore[operator]
            raise ValueError(f"khoảng lật ngược hoặc rỗng: [{self.from_ts}, {self.to_ts}]")
        if self.kind is EvidenceKind.MOSAIC and self.grid is None:
            raise ValueError("mosaic phải có grid")
        return self


class EvidenceJobMessage(_Msg):
    """orchestratord → evidenced.

    Tách hai lane ``fast``/``slow``: ảnh chụp gần như tức thì, còn clip phải chờ
    ``delay`` 20-40 s sau thao tác cẩu. Trộn chung một topic thì job ảnh sẽ kẹt sau job
    clip đang chờ hết ``delay`` — hai lane là cách rẻ nhất để tách hai hạng thời gian.
    """

    topic: ClassVar[Topic] = Topic.EVIDENCE_FAST

    event_id: str
    crane_id: str
    lane: Lane
    anchor_ts: Timestamp
    """Mốc neo — thời điểm rule ``CRANE03`` báo "cẩu đang thao tác".

    Hai việc, đừng nhầm:

    * **Job ``image`` chụp tại đúng khoảnh khắc này.** Nó không có khoảng, và không cần.
    * Job ``clip``/``mosaic`` mang khoảng **tuyệt đối** riêng (``from_ts``/``to_ts``); mốc
      neo ở đây để overlay đánh dấu khoảnh khắc sự kiện bên trong clip, và để ``delay``
      đo từ nó.

    Quy đổi cửa sổ trong config sang khoảng tuyệt đối là việc của orchestrator — làm một
    lần ở chỗ nó đọc config, thay vì mỗi consumer tự cộng trừ."""

    delay: float = Field(default=0.0, ge=0)
    jobs: list[EvidenceJob] = Field(min_length=1)


# ---------------------------------------------------------------- events


class ContainerSlot(BaseModel):
    """Một container trong sự kiện. Twin-lift 20 ft cho hai slot.

    Thay toàn bộ việc nhân đôi thủ công cũ — ``containerNo``/``containerNo2``,
    ``ixCd``/``ixCd_2``, ``shortVideo``/``shortVideo2``, ``sztp``/``sztp2``,
    ``chassisPosition``/``chassisPosition2`` (`utils/data_utils.py:95-274`, ~180 dòng).

    Lưu ý: **không có trường ``chassis_position``**. Nó suy ra 1:1 từ ``cont_position``
    (xem :class:`~common.enum.ContainerPosition`), nên để nó lên wire chỉ tạo thêm một
    đường cho hai giá trị mâu thuẫn nhau. ``gateway/contract/dashboard.py`` đọc
    :attr:`chassis_code` khi dàn phẳng sang payload của e-port.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    container_no: str = ""
    ix_cd: IxCd | None = None
    sztp: str = ""
    cont_position: ContainerPosition | None = None
    container_image: str = ""
    short_video: str = ""
    confidence: Confidence = 0.0

    @property
    def chassis_code(self) -> str:
        """Mã ``chassisPosition`` gửi dashboard. Rỗng khi chưa biết vị trí hoặc là 40 ft.

        Rỗng, không phải ``None`` và không phải lỗi: dashboard coi chuỗi rỗng là "không
        áp dụng", đúng với container 40 ft.
        """
        return self.cont_position.chassis_code if self.cont_position else ""


class EventMessage(_Msg):
    """evidenced → syncd. Sự kiện thao tác cẩu đã hoàn chỉnh.

    Đây là payload mà ``syncd`` dàn phẳng rồi POST lên dashboard e-port
    (``/admin/berth/support/detection``). Việc dàn phẳng sang tên trường cũ
    (``containerNo``/``containerNo2``…) xảy ra ở ``gateway/contract/dashboard.py`` —
    biên tương thích ngược, không rò rỉ vào trong.
    """

    topic: ClassVar[Topic] = Topic.EVENTS

    event_id: str
    crane_id: str
    lane: Lane
    direction: Direction
    anchor_ts: Timestamp
    berth_no: str = ""
    vsl_cd: str = ""
    call_seq: str = ""
    call_year: str = ""
    truck_no: str = ""
    slots: list[ContainerSlot] = Field(default_factory=list, max_length=2)


# ---------------------------------------------------------------- control


class ControlAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    RELOAD_CONFIG = "reload_config"
    RELOAD_RULE = "reload_rule"


class ControlMessage(_Msg):
    """syncd / craneopsctl → mọi service.

    Thay ``BaseCamera.PAUSE_HANDLE_FRAME_SIGNAL`` — một ``threading.Event`` cấp class vốn chỉ có tác dụng trong một process.
    """

    topic: ClassVar[Topic] = Topic.CONTROL

    crane_id: str
    action: ControlAction
    rule_code: str | None = None
    issued_at: Timestamp

    @model_validator(mode="after")
    def _rule_code_required_for_reload_rule(self) -> ControlMessage:
        if self.action is ControlAction.RELOAD_RULE and not self.rule_code:
            raise ValueError("action=reload_rule bắt buộc có rule_code")
        return self


# ---------------------------------------------------------------- mã hoá


_PERCEPTION_BY_ROLE: dict[CameraRole, Topic] = {
    CameraRole.CCODE: Topic.PERCEPTION_CCODE,
    CameraRole.TCODE: Topic.PERCEPTION_TCODE,
    CameraRole.CRANE: Topic.PERCEPTION_CRANE,
}

_MODEL_BY_TOPIC: dict[Topic, type[_Msg]] = {
    Topic.PERCEPTION_CCODE: PerceptionMessage,
    Topic.PERCEPTION_TCODE: PerceptionMessage,
    Topic.PERCEPTION_CRANE: PerceptionMessage,
    Topic.SIGNALS: Signal,
    Topic.MANIFEST: ManifestMessage,
    Topic.EVIDENCE_FAST: EvidenceJobMessage,
    Topic.EVIDENCE_SLOW: EvidenceJobMessage,
    Topic.EVENTS: EventMessage,
    Topic.CONTROL: ControlMessage,
}


def model_for_topic(topic: Topic) -> type[_Msg]:
    """Model pydantic ứng với một topic.

    Raises:
        KeyError: nếu topic chưa được khai báo — không thể xảy ra nếu dùng :class:`Topic`,
            nhưng chặn trường hợp thêm topic mới mà quên map.
    """
    try:
        return _MODEL_BY_TOPIC[topic]
    except KeyError:
        raise KeyError(f"topic {topic!r} chưa có model trong _MODEL_BY_TOPIC") from None


def encode(msg: _Msg) -> bytes:
    """Serialise sang UTF-8 JSON cho Kafka.

    Validate lại trước khi gửi. Nghe thừa vì pydantic đã validate lúc khởi tạo, nhưng
    model là ``frozen`` chứ không phải bất biến sâu — một ``dict`` lồng bên trong vẫn có
    thể bị sửa sau khi tạo. Đây là chốt chặn cuối trước khi dữ liệu rời process.
    """
    validated = type(msg).model_validate(msg.model_dump())
    return validated.model_dump_json().encode("utf-8")


def decode(topic: Topic, data: bytes) -> _Msg:
    """Parse và validate một message nhận từ Kafka.

    Raises:
        ValueError: JSON hỏng, sai kiểu, thiếu trường bắt buộc, hoặc lệch major
            ``schema_version``. Consumer nên bắt lỗi này, log kèm offset, rồi **bỏ qua
            message** — không được để một payload hỏng giết cả worker. Bắt trần rồi
            ``break`` khỏi vòng lặp chính là cách biến một message lỗi thành một sự cố.
    """
    model = model_for_topic(topic)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload của {topic} không phải JSON hợp lệ: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"payload của {topic} phải là object JSON, nhận {type(payload).__name__}")
    return model.model_validate(payload)
