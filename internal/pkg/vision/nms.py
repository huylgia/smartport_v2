"""Triệt tiêu phi cực đại (NMS) cho đầu ra PicoDet.

PicoDet trả về ~3 598 hộp neo cho mỗi lớp, phần lớn chồng lên nhau. NMS giữ lại hộp có
điểm cao nhất rồi loại những hộp trùng nó quá nhiều.

Giữ **đúng từng bước** —
kể cả những chỗ trông lạ, xem ghi chú "GIỮ NGUYÊN".

Dùng ở hai nơi: đo độ chính xác hai model PicoDet (``tools/golden/accuracy.py``) và, ở
Phase 3, hậu xử lý trong ``ds_app``. Với DeepStream, nhiều khả năng bản C++ trong
``nvdsinfer`` sẽ thay chỗ này — NMS rẻ, vừa khuôn API của parser, và tránh được một vòng
gRPC. Xem ``docs/DESIGN_NOTES.md`` DN-010.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from internal.pkg.nptypes import Array

DEFAULT_KEEP_TOP_K = 1000
DEFAULT_NMS_TOP_K = 100
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_SCORE_THRESHOLD = 0.3
"""Mặc định của PicoDet, KHÔNG phải từ
``multiclass_nms`` (vốn để 0.4) — nơi gọi truyền cfg xuống nên giá trị của ``PicoConfig``
mới là giá trị thật khi chạy."""


def iou(box: Array, others: Array) -> Array:
    """IoU giữa một hộp và một mảng hộp.

    ⚠️ CỐ Ý: diện tích tính ``(x2 - x1 + 1)``, tức coi toạ độ là **chỉ số
    pixel bao gồm cả hai đầu**. Bỏ ``+1`` sẽ đổi IoU vài phần nghìn và có thể lật quyết
    định ở ngưỡng — đo lại toàn bộ trước khi đổi.
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (others[:, 2] - others[:, 0] + 1) * (others[:, 3] - others[:, 1] + 1)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    overlap: Array = inter / (box_area + areas - inter)
    return overlap


def suppress(
    dets: Array,
    threshold: float,
    *,
    keep_top_k: int = DEFAULT_KEEP_TOP_K,
    nms_top_k: int = DEFAULT_NMS_TOP_K,
) -> list[int]:
    """Chỉ số các hộp giữ lại. ``dets`` là ``(N, 5)``: ``x1, y1, x2, y2, điểm``."""
    boxes, scores = dets[:, 0:4], dets[:, 4]
    order = scores.argsort()[::-1][:keep_top_k]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        remaining = np.where(iou(boxes[best], boxes[order[1:]]) <= threshold)[0]
        order = order[remaining + 1]

    return keep[:nms_top_k]


def multiclass(
    boxes: Array,
    class_scores: Array,
    *,
    keep_top_k: int = DEFAULT_KEEP_TOP_K,
    nms_top_k: int = DEFAULT_NMS_TOP_K,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Array:
    """NMS từng lớp một, rồi ghép lại.

    Args:
        boxes: ``(N, 4)`` — ``x1, y1, x2, y2`` của N hộp neo.
        class_scores: ``(C, N)`` — điểm của N hộp cho từng lớp trong C lớp.

    Returns:
        ``(M, 6)`` — ``x1, y1, x2, y2, điểm, chỉ_số_lớp``. Mảng **rỗng** nếu không hộp nào
        vượt ngưỡng; nơi gọi phải kiểm ``len(...) > 0`` trước khi lập chỉ mục.
    """
    kept = []
    for class_index, scores in enumerate(class_scores):
        dets = np.hstack([boxes, scores[:, np.newaxis]])
        dets = dets[dets[:, -1] > score_threshold]
        dets = dets[suppress(dets, nms_threshold, keep_top_k=keep_top_k, nms_top_k=nms_top_k)]
        if dets.shape[0] > 0:
            labels = np.full((dets.shape[0], 1), class_index, dtype=dets.dtype)
            kept.append(np.hstack([dets, labels]))

    return np.vstack(kept) if kept else np.empty((0, 6), dtype=np.float32)
