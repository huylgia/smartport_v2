"""Enum dùng chung.

Các giá trị này đi vào payload gửi dashboard, nên **chuỗi là hợp đồng đối ngoại**, không
phải chi tiết cài đặt: đổi ``"40feet"`` thành ``"40ft"`` là đổi API. Dùng ``StrEnum`` để
serialize ra đúng chuỗi đó mà vẫn gõ có kiểm.

Gom một chỗ để gõ sai thành lỗi lúc **load config**, không phải lỗi lúc chạy — nửa đêm,
giữa một chu kỳ cẩu.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "CameraRole",
    "ContainerDim",
    "ContainerPosition",
    "Direction",
    "IxCd",
    "Lane",
    "SignalKind",
]


class StrEnum(str, Enum):
    """``enum.StrEnum`` của Python 3.11, viết lại để chạy được trên 3.10.

    ``common/`` phải tương thích 3.10 vì ``ds_app`` dùng chung ``common/message.py``
    và nó chạy trong container DeepStream (Ubuntu 22.04 ⇒ Python 3.10).

    ``__str__`` bắt buộc phải có: ``class F(str, Enum)`` trần cho ``f"{F.A}"`` ra
    ``"F.A"`` chứ không phải ``"a"``, và lỗi đó chỉ lộ ra khi message đã lên Kafka.
    """

    def __str__(self) -> str:
        return str(self.value)


class CameraRole(StrEnum):
    """Vai trò của camera — quyết định nhánh pipeline nào xử lý nó.

    trước đây hardcode ánh xạ này theo ID camera (``BOT_CAM_ID = 9``, ``CRANE_CAM_ID = 10``,
    ``CCODE_CAM_IDS = [1,4,6,7,8,11]``, ``HEAD_CAM_IDS = [3,5]`` — `service.py:28-32`),
    nên lắp thêm camera hoặc đổi số camera là phải sửa code. v2 để trong config.
    """

    CCODE = "ccode"
    """Nhận dạng mã container. DB text det → SVTR rec."""

    TCODE = "tcode"
    """Nhận dạng số đầu kéo. PicoDet → FastViT classifier."""

    CRANE = "crane"
    """Nhìn xuống khu vực dưới cẩu. PicoDet đầu kéo + container."""

    BOTTOM = "bottom"
    """Soi đáy container. Không chạy model, chỉ ghi hình."""

    EVIDENCE_ONLY = "evidence_only"
    """Chỉ ghi hình cho bộ ảnh 6 mặt, không chạy model và không decode.

    GC03 camera 2 (`Hông trái - Trước`) thuộc loại này. Vai trò tồn tại riêng vì nó tiết
    kiệm thật: decode 2688x1520@30 chỉ để lấy một JPEG mỗi 5 giây là lãng phí cả NVDEC lẫn
    VRAM — trên RTX 3060 với 11 camera thì đó không phải khoản dư ra."""

    @property
    def runs_model(self) -> bool:
        """Vai trò này có nhánh model trong pipeline DeepStream không.

        Quyết định camera nào cần decode. Xem ``docs/HARDWARE_BUDGET.md`` §2.3: bỏ decode
        cho `bottom` và `evidence_only` giúp giảm từ 10 xuống 8 luồng.
        """
        return self in (CameraRole.CCODE, CameraRole.TCODE, CameraRole.CRANE)


class Lane(StrEnum):
    """Làn xe dưới cẩu. Tối đa 3."""

    ONE = "1"
    TWO = "2"
    THREE = "3"


class Direction(StrEnum):
    """Chiều xe chạy, **đo trên trục x của ảnh**.

    Tên nêu đủ ba thứ để kiểm chứng được bằng một khung hình: trục, mép xuất phát, và
    chiều. Probe tính được từ toạ độ; người review đối chiếu được với ảnh.

    ⚠️ Tên cũ là ``RIGHT``/``COUNTER`` và cả hai đều không mô tả gì: ``RIGHT`` không nói
    phải-của-cái-gì lẫn đi hướng nào, còn ``COUNTER`` giả định có một chiều "chuẩn" — đó là
    thiên kiến vận hành lọt vào một cái tên kỹ thuật. Hai chiều đối xứng nhau về mặt hình
    học; chỉ tần suất là khác.

    Chiều là **tham số**, không phải nhánh: hai chiều dùng chung một engine điều phối, chỉ
    khác ngưỡng và mốc neo. Nhân đôi đường xử lý cho mỗi chiều nghĩa là mọi sửa lỗi sau
    này phải nhớ sửa hai chỗ.
    """

    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"
    """Xe vào từ mép PHẢI ảnh, chạy sang trái. Toạ độ x giảm dần."""

    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    """Xe vào từ mép TRÁI ảnh, chạy sang phải. Toạ độ x tăng dần.

    ⚠️ **Chưa được sinh ra ở đâu cả** — chiều này đang hoãn (xem ``docs/RULES.md``). Giữ
    trong hợp đồng vì gỡ rồi thêm lại là hai lần đổi schema cho cùng một thứ. Ai đọc một
    ``Signal`` hôm nay có thể yên tâm ``direction`` luôn là ``RIGHT_TO_LEFT``; đừng viết
    code *dựa* vào điều đó."""


class IxCd(StrEnum):
    """Nhập hay xuất. Lấy từ manifest Oracle CATOS."""

    IMPORT = "I"
    EXPORT = "X"


class ContainerDim(StrEnum):
    FT20 = "20feet"
    FT40 = "40feet"


class ContainerPosition(StrEnum):
    """Vị trí container trên rơ-moóc — gói trọn *một* khái niệm.

    Cách thường gặp: thông tin này nằm rải ba chỗ:

    * quy tắc dẫn xuất từ ``cont_dim`` + ``truck_position`` — `crane_camera/data.py:64-91`
    * bảng tra sang mã dashboard — `utils/data_utils.py:299-308`
      (``{"40feet": "", "20feet-1": "F", "20feet-2": "A"}``)
    * hai trường riêng biệt trên ``Event``: ``cont_position`` và ``chassisPosition``

    Ba chỗ đó là **cùng một thứ**, ánh xạ 1:1, và không có đường nào khác gán
    ``chassisPosition`` (mọi call site đều qua ``update_chassisPosition()``). Giữ chúng
    tách rời chỉ tạo ra khả năng cho hai giá trị mâu thuẫn nhau.

    Ở đây mỗi thành viên mang cả tên nội bộ lẫn mã gửi dashboard, và quy tắc dẫn xuất là
    một classmethod ngay cạnh. Thêm một vị trí mới ⇒ sửa đúng một dòng.
    """

    FT40 = ("40feet", "")
    FT20_1 = ("20feet-1", "F")
    FT20_2 = ("20feet-2", "A")

    chassis_code: str
    """Mã ``chassisPosition`` mà dashboard e-port mong đợi. Rỗng nghĩa là container 40 ft."""

    def __new__(cls, value: str, chassis_code: str) -> ContainerPosition:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.chassis_code = chassis_code
        return obj

    @classmethod
    def for_slot(cls, cont_dim: ContainerDim, slot_index: int) -> ContainerPosition:
        """Vị trí container theo thứ tự slot tính từ đầu xe.

        ``slot_index`` là hạng của container khi **sắp xếp theo khoảng cách tới đầu xe**:
        ``0`` là gần nhất. Việc *đo* khoảng cách nằm ở rule ``CRANE04``
        (``internal/rules/modules/``), không phải ở đây — enum chỉ ánh xạ thứ tự sang tên.

        Container 40 ft chiếm cả rơ-moóc nên chỉ có một slot; ``slot_index`` bị bỏ qua.

        Suy từ khoảng cách container ↔ đầu xe, **không** từ dải dọc mà tâm đầu xe rơi
        vào: dải dọc là hiệu chỉnh tuyệt đối theo pixel, phải làm lại mỗi khi camera xê
        dịch. Lý do đổi và các câu hỏi còn mở: ``docs/DESIGN_NOTES.md`` DN-001.

        Raises:
            ValueError: nếu ``slot_index`` ngoài ``0``/``1`` với container 20 ft — một
                rơ-moóc chỉ chở tối đa hai container 20 ft.
        """
        if cont_dim is ContainerDim.FT40:
            return cls.FT40
        if slot_index == 0:
            return cls.FT20_1
        if slot_index == 1:
            return cls.FT20_2
        raise ValueError(f"slot_index phải là 0 hoặc 1 với container 20 ft, nhận {slot_index}")


class SignalKind(StrEnum):
    """Loại signal mà rule phát ra.

    Đây là khoá mà composition spec tham chiếu trong ``configs/operations/*.yaml``.
    Thêm một loại mới nghĩa là thêm một hằng ở đây — spec không được dùng chuỗi tự do,
    vì gõ sai sẽ âm thầm khiến một trường không bao giờ được điền.
    """

    LANE_ACTIVE = "lane_active"
    TRUCK_STABLE = "truck_stable"
    CRANE_OP = "crane_op"
    CONT_DIM = "cont_dim"
    CONT_POSITION = "cont_position"
    CONTAINER_NO = "container_no"
    TRUCK_NO = "truck_no"
    BOTTOM_READY = "bottom_ready"
