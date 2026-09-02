"""Đường ống PicoDet: ảnh vào, hộp ra — thuần, không biết gì về Triton.

Hai model dùng chung mã ở đây, khác nhau đúng bảng lớp:

======================== =================== =========================================
model                    lớp                 dùng cho
======================== =================== =========================================
``truckitems_pico``      ``head``,           ``CRANE01`` gán lane, ``CRANE04`` suy kích
                         ``container``       thước container
``truckhead_pico``       ``head``            ``TCODE01`` khoanh vùng đầu kéo trước khi
                                             phân loại số xe
======================== =================== =========================================

**Model đã tự giải mã hộp.** ``tmp_16`` ra thẳng ``x1, y1, x2, y2`` trong khung 416x416;
không có bước ``distance2bbox`` nào phải làm ở đây. Chỉ còn NMS rồi đưa toạ độ về ảnh gốc.

⚠️ ``INTER_CUBIC``, không phải ``INTER_LINEAR``. Đây là phép nội suy PicoDet được huấn
luyện với, và hai phép cho hộp **lệch nhau tới 17 px** trên cùng một ảnh — trong khi recall
đo được y hệt nhau, nên bảng số tổng không phát hiện ra. ``CRANE01`` gán lane bằng điểm mốc
của hộp, nên sai lệch cỡ đó lật được phán quyết ở sát biên vùng. Xem
``docs/HARDWARE_BUDGET.md`` §6.2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from internal.pkg.vision.nms import multiclass

if TYPE_CHECKING:
    from internal.pkg.nptypes import Array, Image

__all__ = ["INPUT_SIZE", "Detection", "PicoParams", "detect", "to_tensor"]

INPUT_SIZE = (416, 416)
"""``(cao, rộng)`` đầu vào của cả hai model pico. Cố định trong engine TensorRT."""


@dataclass(frozen=True, slots=True)
class PicoParams:
    """Ngưỡng hậu xử lý. Mặc định lấy từ ``PicoConfig`` đang chạy production."""

    score_threshold: float = 0.3
    nms_threshold: float = 0.3


DEFAULT_PARAMS = PicoParams()
"""Dùng làm mặc định cho :func:`detect`. Là singleton module-level chứ không gọi
``PicoParams()`` ngay trong chữ ký — ruff B008."""


@dataclass(frozen=True, slots=True)
class Detection:
    """Một vật đã phát hiện, toạ độ trên **ảnh gốc**."""

    label: str
    score: float
    box: tuple[int, int, int, int]
    """``x_min, y_min, x_max, y_max``, đã cắt về trong ảnh."""


def to_tensor(image: Image) -> Array:
    """Ảnh BGR ``(H, W, 3)`` uint8 → tensor ``(1, 3, 416, 416)`` float32.

    BGR **thô** ``[0, 255]``: phép chia 255 và đảo kênh sang RGB đã gấp vào đồ thị ONNX
    (DN-012). Đừng chuẩn hoá thêm ở đây — model đã gấp vẫn chạy trên dữ liệu đã chuẩn hoá
    và vẫn trả về hộp, chỉ là hộp rác, không có exception nào để lần ra.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"ảnh đầu vào rỗng: {image.shape}")
    target_h, target_w = INPUT_SIZE
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    tensor: Array = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    return tensor


def detect(
    image: Image,
    infer: Callable[[Array], tuple[Array, Array]],
    labels: dict[int, str],
    params: PicoParams = DEFAULT_PARAMS,
) -> list[Detection]:
    """Chạy detector trên một ảnh và trả hộp theo toạ độ ảnh gốc.

    Args:
        image: ảnh BGR gốc, kích thước tuỳ ý.
        infer: nhận tensor ``(1, 3, 416, 416)`` float32, trả ``(hộp, điểm)`` với hộp
            ``(3598, 4)`` và điểm ``(C, 3598)`` — đã bỏ chiều batch.
        labels: ``{chỉ_số_lớp: tên}``. Lớp **không có trong bảng bị bỏ**, đúng như v1:
            ``truckhead_pico`` trả 2 lớp nhưng chỉ lớp 0 có nghĩa.
        params: ngưỡng.

    Returns:
        Danh sách rỗng nếu không hộp nào vượt ngưỡng.
    """
    height, width = image.shape[:2]
    boxes, scores = infer(to_tensor(image))

    dets = multiclass(
        boxes,
        scores,
        nms_threshold=params.nms_threshold,
        score_threshold=params.score_threshold,
    )
    if len(dets) == 0:
        return []

    # Về toạ độ ảnh gốc. Hệ số là kích_thước_đích / kích_thước_gốc nên ở đây CHIA.
    target_h, target_w = INPUT_SIZE
    scale = np.array([target_w / width, target_h / height] * 2, dtype=np.float32)
    dets[:, :4] /= scale

    # ⚠️ GIỮ NGUYÊN: chỉ kẹp cận DƯỚI về 0, để cận trên trôi ra ngoài mép ảnh.
    #
    # Kẹp cả hai đầu thì hợp lý hơn, nhưng nó đổi toạ độ hộp so với bản đang chạy ở cảng,
    # và toạ độ đó đi thẳng vào phép gán lane của `CRANE01`. Đổi ở đây là đổi kết quả
    # nghiệp vụ mà không có phép đo nào đi kèm — để dành cho một thay đổi riêng có đo lại.
    # Nơi cắt ảnh không hỏng vì việc này: numpy tự cắt bớt khi chỉ số vượt biên.
    dets[dets < 0] = 0

    found: list[Detection] = []
    for det in dets:
        label = labels.get(int(det[5]))
        if label is None:
            # v1 bỏ qua lớp không khai trong `class_mapping`. `truckhead_pico` trả 2 lớp
            # nhưng chỉ lớp 0 (`head`) có nghĩa — lớp 1 là rác từ lúc huấn luyện chung.
            continue
        x1, y1, x2, y2 = (int(v) for v in det[:4])
        found.append(Detection(label=label, score=float(det[4]), box=(x1, y1, x2, y2)))
    return found


def crop(image: Image, box: tuple[int, int, int, int]) -> Image:
    """Cắt một hộp ra khỏi ảnh.

    Hộp có thể vượt mép phải/dưới (xem ghi chú "GIỮ NGUYÊN" trong :func:`detect`); numpy
    tự cắt bớt, nên crop thu được nhỏ hơn hộp. Trả về mảng **rỗng** nếu hộp nằm trọn ngoài
    ảnh — nơi gọi phải kiểm ``.size`` trước khi đưa vào model.
    """
    x1, y1, x2, y2 = box
    region: Image = image[y1:y2, x1:x2]
    return region
