"""Hậu xử lý DB/DB++: bitmap xác suất → hộp chữ.

Vài bước ở đây **trông thừa nhưng không được bỏ** — ngưỡng đang chạy đã hiệu chỉnh trên
chúng. Mỗi chỗ như vậy có ghi chú ⚠️ tại chỗ; đọc trước khi dọn.

Vì sao là module thuần: phần này chạy trong **Python backend của Triton**
(``triton/repo/…``), một tiến trình riêng không có onnxruntime lẫn ``cv2.dnn``. Nên nó chỉ
phụ thuộc numpy + cv2 + pyclipper + shapely, không biết gì về model, không đọc config,
không log — nhờ vậy test được bằng ``pytest`` trên máy không có GPU.

Chi phí thật, **đo chứ không ước lượng** (bitmap 512x576 thật từ model, 2026-08-30):
**0,151 ms/ảnh**. Nhỏ hơn nhiều so với con số 3-6 ms mà bản đầu của ghi chú này phỏng
đoán — bitmap của DB rất thưa (vùng vượt ngưỡng chỉ chiếm 0,17 % ảnh) nên
``cv2.findContours`` gần như không có việc gì làm.

Hệ quả cho thiết kế: bước này **không phải** nút thắt, và viết lại bằng C++ sẽ tiết kiệm
gần như không có gì. Trong một request 6,8 ms thì nó chiếm 2,2 %. Chi phí thật nằm ở
phép chuẩn hoá đầu vào (``preprocess.to_tensor``, ~1,9 ms) — xem ``docs/DESIGN_NOTES.md``
DN-010.

Việc tách module này ra khỏi luồng streaming vẫn đúng, nhưng lý do là ``instance_group``
(3 tiến trình thật, đo được 2,7x thông lượng — DN-009), không phải vì bản thân nó nặng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from internal.pkg.nptypes import Array

# Kernel giãn nở mask. Tạo nó ở cấp module qua giá trị mặc định của dataclass
# — một mảng dùng chung có thể bị ghi đè. Ở đây tạo một lần, chỉ đọc.
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Ngưỡng dưới cho hộp sau khi nới rộng, tính bằng pixel. trước đây hardcode 15 ở hai chỗ
# .
_MIN_EXPANDED_SIDE = 15


@dataclass(frozen=True, slots=True)
class DbPostConfig:
    """Tham số hậu xử lý.

    Giá trị mặc định lấy từ ``DBConfig`` cũ. ``expand_ratio`` và
    ``box_threshold`` bị ``CCRecognizer`` ghi đè theo từng ROI, nên chúng nằm trong config chứ không
    hardcode.
    """

    bitmap_threshold: float = 0.1
    box_threshold: float = 0.2
    max_candidates: int = 1000
    unclip_ratio: float = 1.5
    min_size: int = 3
    # (ngang, dọc). Nới hộp ra vì DB bám sát nét chữ, cắt đúng hộp thì mất chân chữ.
    expand_ratio: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True, slots=True)
class TextBox:
    """Một vùng chữ đã phát hiện, toạ độ trên ảnh GỐC (đã chia scale_factor)."""

    bbox: tuple[int, int, int, int]
    """``(x_min, y_min, x_max, y_max)``, đã nới theo ``expand_ratio``."""

    quad: Array = field(repr=False)
    """4 điểm ``(x, y)`` theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái."""

    score: float
    """Xác suất trung bình của bitmap bên trong hộp xoay (trước khi nới)."""

    @property
    def area(self) -> int:
        x_min, y_min, x_max, y_max = self.bbox
        return (x_max - x_min) * (y_max - y_min)


def decode(
    bitmap: Array,
    cfg: DbPostConfig,
    *,
    scale_factor: tuple[float, float],
    image_size: tuple[int, int],
) -> list[TextBox]:
    """Bitmap xác suất ``(H, W)`` → danh sách hộp chữ trên ảnh gốc.

    Args:
        bitmap: đầu ra model, giá trị ``[0, 1]``, kích thước bằng ảnh ĐÃ resize.
        cfg: ngưỡng.
        scale_factor: ``(sx, sy)`` = kích thước đã resize / kích thước gốc. Toạ độ
            được chia cho hệ số này để về ảnh gốc.
        image_size: ``(width, height)`` của ảnh GỐC — dùng để kẹp hộp sau khi nới.

    Returns:
        Danh sách hộp, thứ tự theo ``cv2.findContours`` (KHÔNG sắp xếp — nơi gọi tự
        chọn top-k, xem :func:`internal.pkg.vision.textcrop.top_k_by_area`).
    """
    if scale_factor[0] <= 0 or scale_factor[1] <= 0:
        # Chia cho 0 ở dưới sẽ cho toạ độ NaN, rồi `astype(np.int32)` biến chúng thành
        # số nguyên rác — hộp trông hợp lệ nhưng ở vị trí vô nghĩa, và không ai báo lỗi.
        msg = f"scale_factor phải dương, nhận {scale_factor}"
        raise ValueError(msg)

    mask: Array = (bitmap > cfg.bitmap_threshold).astype(np.uint8)
    mask = cv2.dilate(mask, _DILATE_KERNEL)

    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: list[TextBox] = []
    for contour in contours[: min(len(contours), cfg.max_candidates)]:
        points, min_side = _mini_box(contour)
        if min_side < cfg.min_size:
            continue

        score = _box_score(bitmap, points)
        if score < cfg.box_threshold:
            continue

        # unclip: nới đa giác ra ngoài một khoảng tỉ lệ với diện tích/chu vi. Nếu phép
        # nới làm đa giác vỡ thành nhiều mảnh thì bỏ — hộp đó không đáng tin.
        if len(_unclip(points, cfg.unclip_ratio)) > 1:
            continue

        # ⚠️ CỐ Ý: kết quả unclip ở trên bị VỨT ĐI, hộp nhỏ nhất được tính
        # lại từ `points` cũ. Nghĩa là unclip chỉ đóng vai trò bộ lọc
        # "đa giác có vỡ không", không thực sự nới hộp — việc nới do `expand_ratio` làm.
        # Trông như lỗi, nhưng ngưỡng expand_ratio hiện tại đã được chỉnh theo hành vi
        # này, nên sửa ở đây sẽ làm lệch mọi ROI đang chạy. Nếu muốn đổi: đổi ở một PR
        # riêng, đo lại toàn bộ golden set.
        points, min_side = _mini_box(points)
        if min_side < cfg.min_size + 2:
            continue

        quad = (np.asarray(points) / scale_factor).astype(np.int32)
        x, y, w, h = cv2.boundingRect(quad)
        bbox, quad = _expand(
            image_size,
            (x, y, x + w - 1, y + h - 1),
            quad,
            cfg.expand_ratio,
        )
        boxes.append(TextBox(bbox=bbox, quad=quad, score=float(score)))

    return boxes


def _mini_box(contour: Array) -> tuple[Array, float]:
    """Hình chữ nhật xoay nhỏ nhất bao contour, 4 đỉnh theo thứ tự TL, TR, BR, BL."""
    rect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rect), key=lambda p: p[0])

    # Hai điểm trái nhất: điểm có y nhỏ hơn là trên-trái. Tương tự cho hai điểm phải.
    tl, bl = (pts[0], pts[1]) if pts[1][1] > pts[0][1] else (pts[1], pts[0])
    tr, br = (pts[2], pts[3]) if pts[3][1] > pts[2][1] else (pts[3], pts[2])

    return np.array([tl, tr, br, bl]), min(rect[1])


def _box_score(bitmap: Array, quad: Array) -> float:
    """Xác suất trung bình bên trong đa giác — dùng làm độ tin cậy của hộp."""
    h, w = bitmap.shape[:2]
    box = quad.copy()
    x_min = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    x_max = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    y_min = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    y_max = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))

    mask: Array = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    box[:, 0] -= x_min
    box[:, 1] -= y_min
    cv2.fillPoly(mask, [box.reshape(-1, 2).astype(np.int32)], (1,))

    return float(cv2.mean(bitmap[y_min : y_max + 1, x_min : x_max + 1], mask)[0])


def _unclip(quad: Array, ratio: float) -> list[Array]:
    """Nới đa giác ra ``area * ratio / length`` pixel theo pháp tuyến."""
    poly = Polygon(quad)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(quad, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded: list[Array] = offset.Execute(poly.area * ratio / poly.length)
    return expanded


def _expand(
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    quad: Array,
    ratio: tuple[float, float],
) -> tuple[tuple[int, int, int, int], Array]:
    """Nới hộp và 4 đỉnh theo tỉ lệ, kẹp trong khung ảnh.

    DB được huấn luyện để bám sát nét chữ. Cắt đúng hộp đó rồi đưa vào OCR thì mất chân
    chữ và dấu, nên phải nới. Tỉ lệ khác nhau giữa mã ngang ``(1.1, 1.7)`` và mã dọc
    ``(1.0, 1.1)`` vì chữ dọc đã cao sẵn.

    ⚠️ **Đừng thêm lối tắt ``if ratio == (1.0, 1.0): return bbox``.** Nó trông như một
    tối ưu hiển nhiên và nó **đổi kết quả**: ngay cả khi không nới theo tỉ lệ, hàm này vẫn
    áp sàn ``_MIN_EXPANDED_SIDE`` và vẫn kẹp hộp vào khung ảnh. Bỏ qua hai bước đó sẽ cho
    ra hộp khác — với hộp nhỏ thì khác đáng kể.
    """
    img_w, img_h = image_size
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    dw = w * (ratio[0] - 1.0) / 2
    dh = h * (ratio[1] - 1.0) / 2
    # Sàn tuyệt đối: hộp quá nhỏ thì nới theo tỉ lệ vẫn quá nhỏ để OCR đọc được.
    dw = max(dw, (_MIN_EXPANDED_SIDE - w) / 2)
    dh = max(dh, (_MIN_EXPANDED_SIDE - h) / 2)

    p0, p1, p2, p3 = quad
    expanded_quad = np.array(
        [
            [max(p0[0] - dw, 0), max(p0[1] - dh, 0)],
            [min(p1[0] + dw, img_w), max(p1[1] - dh, 0)],
            [min(p2[0] + dw, img_w), min(p2[1] + dh, img_h)],
            [max(p3[0] - dw, 0), min(p3[1] + dh, img_h)],
        ],
        dtype=np.int32,
    )
    expanded_bbox = (
        max(int(x_min - dw), 0),
        max(int(y_min - dh), 0),
        min(int(x_max + dw), img_w),
        min(int(y_max + dh), img_h),
    )
    return expanded_bbox, expanded_quad
