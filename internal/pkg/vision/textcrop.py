"""Cắt và nắn vùng chữ từ ảnh gốc để đưa vào OCR.

Đây là bước nằm GIỮA detector và recognizer. Nó không phải một phép cắt đơn thuần mà gồm:

1. Chọn **top-5 hộp theo diện tích** — mã container thật bao giờ cũng là vùng chữ lớn
   nhất trong ROI; phần còn lại là số hiệu hãng, nhãn dán, vệt bẩn.
2. Cắt theo hộp.
3. Mã **ngang**: nắn phối cảnh về hình chữ nhật bằng 4 đỉnh (container nhìn chéo nên
   chữ bị nghiêng). Mã **dọc**: chỉ xoay 90° ngược chiều kim đồng hồ.
4. Cân sáng bằng CLAHE — mặt container phơi nắng nên nửa sáng nửa tối.
5. **Cổng nét ảnh**: bỏ crop có độ nét < ngưỡng. Xe đang chạy làm chữ nhoè; đọc ảnh nhoè
   sinh ra mã sai mà vẫn có độ tin cậy cao, tệ hơn là không đọc.

Bước 5 là lý do nhánh ccode **không thể** là ``ensemble`` của Triton: nó bỏ bớt phần tử,
nên số crop ra khác số hộp vào. Ensemble là đồ thị tĩnh, không diễn đạt được. Xem
``docs/DESIGN_NOTES.md`` DN-007.

Module thuần: chỉ numpy + cv2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from internal.pkg.nptypes import Array, Image
    from internal.pkg.vision.dbpost import TextBox

# Ngưỡng nét mặc định. Trung bình
# biên độ phổ Fourier, không có đơn vị vật lý, chỉ so tương đối được.
DEFAULT_SHARPNESS_MIN = 1000.0

DEFAULT_TOP_K = 5


def top_k_by_area(boxes: Sequence[TextBox], k: int = DEFAULT_TOP_K) -> list[TextBox]:
    """``k`` hộp có diện tích lớn nhất, giảm dần."""
    return sorted(boxes, key=lambda b: b.area, reverse=True)[:k]


def sharpness(image: Image) -> float:
    """Độ nét = biên độ phổ Fourier trung bình. Càng cao càng nét.

    Ảnh nhoè mất thành phần tần số cao nên biên độ trung bình tụt xuống. Rẻ hơn nhiều
    so với việc chạy OCR rồi vứt kết quả.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(np.fft.fftshift(np.fft.fft2(gray)))))


def equalize_brightness(
    image: Image,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image:
    """Cân sáng cục bộ bằng CLAHE trên kênh L của LAB.

    Chỉ đụng vào độ sáng, giữ nguyên màu — cân bằng histogram trên cả ba kênh BGR sẽ
    làm lệch màu.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


def warp_quad(image: Image, quad: Array) -> Image:
    """Nắn tứ giác về hình chữ nhật thẳng.

    Che phần ngoài tứ giác bằng đen trước khi nắn (dùng ``bitwise_and`` với mask):
    hộp bao của một tứ giác nghiêng chứa cả góc của vùng bên cạnh, để nguyên thì OCR
    đọc lẫn ký tự của dòng kế.

    ⚠️ **Chiều rộng đích lấy đúng cạnh DƯỚI, KHÔNG phải trung bình hai cạnh** — và đó là
    cố ý, không phải lỗi gõ.

    Nhìn qua thì ``(norm(br - bl) + norm(br - bl)) / 2`` trông như copy-paste hỏng, và
    "sửa" nó thành trung bình cạnh trên/dưới là phản xạ đầu tiên của bất kỳ ai đọc. Đừng.
    Phép đó đổi kích thước **mọi** ảnh crop ở **mọi** tứ giác không phải hình bình hành:
    đo được crop lệch tới 15 mức xám, đủ để đổi điểm tin cậy OCR từ 0,9653 xuống 0,9229 và
    làm một mã container hợp lệ rớt khỏi ngưỡng 0,95.

    Muốn đổi thì đổi có chủ đích ở một thay đổi riêng, kèm đo lại toàn bộ golden set.
    Chiều cao thì tính đúng (trung bình hai cạnh trái/phải).
    """
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    x1, y1 = quad.min(axis=0)
    x2, y2 = quad.max(axis=0)

    cropped = image[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1].copy()
    local = (quad - [x1, y1]).astype(np.int64)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [local], -1, (255,), -1, cv2.LINE_AA)
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)

    # Tính kích thước đích trên int64/float64, KHÔNG phải float32: float32 chỉ có ~7 chữ
    # số nên với toạ độ hàng nghìn pixel, ma trận phối cảnh lệch đủ để crop lệch 2-4 mức xám.
    tl, tr, br, bl = local
    # Cạnh dưới đếm hai lần — CỐ Ý. Xem cảnh báo ở docstring trước khi "sửa".
    width = (np.linalg.norm(br - bl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2
    if width < 1 or height < 1:
        return masked

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(local.astype(np.float32), dst)
    return cv2.warpPerspective(masked, matrix, (int(width), int(height)))


def prepare_crop(
    image: Image,
    box: TextBox,
    *,
    vertical: bool,
) -> tuple[Image, float]:
    """Cắt + nắn + cân sáng một hộp. Trả về ``(ảnh crop, độ nét)``.

    Thứ tự các bước khác nhau giữa hai hướng:
    :

    * **ngang**: nắn phối cảnh → cân sáng → đo nét
    * **dọc**: cân sáng → xoay 90° CCW → đo nét

    Mã dọc không nắn phối cảnh vì chữ xếp theo chiều cao container, tứ giác gần như
    luôn là chữ nhật thẳng; nắn chỉ thêm nhiễu nội suy.
    """
    x1, y1, x2, y2 = box.bbox
    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return crop, 0.0

    if vertical:
        crop = equalize_brightness(crop)
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Toạ độ 4 đỉnh đang ở hệ ảnh gốc; dời về hệ của chính crop.
        crop = warp_quad(crop, np.asarray(box.quad, dtype=np.float64) - [x1, y1])
        if crop.size == 0:
            return crop, 0.0
        crop = equalize_brightness(crop)

    return crop, sharpness(crop)
