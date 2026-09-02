"""Chuẩn bị ảnh cho model — và bảng hằng số chuẩn hoá mà model được huấn luyện với.

**Hợp đồng đầu vào chỉ có một**: mọi model đều nhận pixel **BGR thô ``[0,255]``**. Phép
chuẩn hoá đã được gấp vào chính đồ thị ONNX (``tools/fold_preprocess.py``, DN-012), nên
ở đây không còn phép tính nào ngoài ``cv2.resize``.

Các hằng số ``*_NORM`` bên dưới **không được áp lúc chạy**. Chúng là bản ghi lại thang số
mà từng model được huấn luyện với, và là đầu vào để ``fold_preprocess`` tính ra hai hằng
số ``A``/``B`` chèn vào đồ thị. Giữ chúng ở đây vì chúng mô tả *model*, không mô tả công
cụ.

⚠️ **Đừng thêm lại một nhánh "áp chuẩn hoá bằng Python" cho model chưa gấp.** Hai đường
vào cho cùng một model là một lớp lỗi: đưa dữ liệu đã chuẩn hoá vào model đã gấp thì nó
vẫn chạy, vẫn trả về chuỗi, chỉ là chuỗi rác — không có exception nào để lần ra. Mọi model
trong ``triton/repo`` đều đã gấp; nếu thêm model mới, gấp nó, đừng thêm nhánh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from internal.pkg.nptypes import Image


@dataclass(frozen=True, slots=True)
class Normalization:
    """Tham số chuẩn hoá theo kênh, thứ tự BGR."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


# Hai bộ hằng số dưới đây quyết định model chạy đúng hay ra rác. Chúng KHÔNG hoán đổi
# được cho nhau — lấy từ `DBConfig` theo `model_type`:
#
#   model_type="db"   -> hằng số kiểu ImageNet   -> mã container DỌC
#   model_type="db++" -> mean riêng, std = 1     -> mã container NGANG
#
# ⚠️ Ánh xạ này ngược trực giác: "db++" nghe như bản nâng cấp của "db", nhưng nó chỉ là
# nhãn chọn bộ hằng số — cả hai file model đều là DB. Đừng suy ra kiến trúc từ cái tên.
DET_NORM_VERTICAL = Normalization(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
DET_NORM_HORIZONTAL = Normalization(
    mean=(0.48109378172549, 0.45752457890196, 0.40787054090196),
    std=(1.0, 1.0, 1.0),
)
# SVTR dùng chuẩn hoá đối xứng về [-1, 1].
REC_NORM = Normalization(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def to_tensor(
    image: Image,
    size: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[Image, tuple[float, float]]:
    """Ảnh BGR ``(H, W, 3)`` uint8 → tensor ``(1, H, W, 3)`` uint8, kèm hệ số co giãn.

    Không đổi kiểu, không hoán trục, không chuẩn hoá: ảnh sau ``resize`` đã đúng thứ model
    cần. Đây là đường rẻ nhất có thể — xem DN-012 để biết nó tiết kiệm bao nhiêu.

    Args:
        image: ảnh gốc, BGR.
        size: ``(cao, rộng)`` — **không** phải ``(rộng, cao)``. Ngược với quy ước của
            ``cv2.resize``, nên đây là chỗ dễ nhầm nhất trong module; hoán nhầm hai số
            này thì model vẫn chạy và vẫn trả về chuỗi.
        interpolation: ``INTER_CUBIC`` cho detector, ``INTER_LINEAR`` cho recognizer —
            đúng phép nội suy mà từng model được huấn luyện với.

    Returns:
        ``(tensor, scale_factor)`` với ``scale_factor = (sx, sy)`` = kích thước đích /
        kích thước gốc. Hậu xử lý chia toạ độ cho hệ số này để về ảnh gốc.
    """
    target_h, target_w = size
    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        msg = f"ảnh đầu vào rỗng: {image.shape}"
        raise ValueError(msg)
    if target_h <= 0 or target_w <= 0:
        # Kích thước 0 tới từ config sai. Không chặn ở đây thì hệ số co giãn thành 0, và
        # hậu xử lý DB chia cho nó rồi cho ra toạ độ NaN — hộp rác mà không ai báo lỗi.
        msg = f"kích thước đích phải dương, nhận (cao={target_h}, rộng={target_w})"
        raise ValueError(msg)

    scale = (1.0, 1.0)
    if (src_h, src_w) != (target_h, target_w):
        scale = (target_w / src_w, target_h / src_h)
        # Dùng fx/fy chứ không truyền dsize: hai đường cho kích
        # thước ra có thể lệch 1 pixel so với dsize do làm tròn. Giữ nguyên để khớp.
        image = cv2.resize(image, None, None, fx=scale[0], fy=scale[1], interpolation=interpolation)

    return image[np.newaxis, ...], scale


def batch_to_tensor(
    images: list[Image],
    size: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> Image:
    """Nhiều ảnh → một tensor ``(N, cao, rộng, 3)`` uint8.

    Gộp crop thành một tensor để ``dynamic_batching`` của Triton còn gom tiếp với crop từ
    các camera khác. Gọi model từng crop một là nguồn lãng phí lớn nhất có thể có ở đây.
    """
    if not images:
        return np.empty((0, *size, 3), dtype=np.uint8)
    return np.concatenate([to_tensor(img, size, interpolation=interpolation)[0] for img in images])


DET_LONG_SIDE = 640
"""Cạnh dài mặc định khi resize một vùng OCR cho detector.

Vì sao có công thức thay vì khai từng vùng: v1 chỉnh tay 12 giá trị khác nhau cho nhánh
ngang, và mỗi lần vẽ một vùng mới lại phải dò lại. Đo trên dữ liệu thật cho thấy việc dò
đó **không mua được gì**: một công thức duy nhất đọc đúng bằng bản chỉnh tay.

Con số 640 nằm giữa một **vùng phẳng**, không phải một đỉnh nhọn — đo trên 4 mẫu có đồng
thuận, mọi cạnh dài từ 576 tới 832 cho 3-4/4, còn từ 544 trở xuống tụt về 1/4. Có sàn, và
trên sàn thì chọn gì cũng gần như nhau. Xem ``docs/HARDWARE_BUDGET.md`` §6.2.

⚠️ Cơ sở đo còn mỏng (4 mẫu). Điều đã chắc là **sàn ~576** và **không có đỉnh nhọn**; con
số chính xác nhất thì chưa. Đo lại khi có thêm ảnh có mã đọc được.
"""


def fit_long_side(height: int, width: int, long_side: int = DET_LONG_SIDE) -> tuple[int, int]:
    """``(cao, rộng)`` cho detector: giữ tỉ lệ, đưa cạnh dài về ``long_side``, làm tròn 32.

    Làm tròn về **bội số 32** vì detector DB hạ mẫu 5 lần; kích thước lẻ khiến nó tự đệm
    và bản đồ xác suất lệch so với ảnh vào.

    Đây chính là quy luật v1 đã tuân theo mà không viết ra: đo lại 20 vùng của v1 thấy tỉ
    lệ ``input_size`` khớp tỉ lệ vùng, lệch trung vị **2,0 %** — toàn bộ phần lệch là do
    làm tròn. Thứ v1 chỉnh tay chỉ là cạnh dài.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"kích thước vùng phải dương, nhận {(height, width)}")
    if long_side <= 0:
        raise ValueError(f"long_side phải dương, nhận {long_side}")
    scale = long_side / max(height, width)
    return (
        max(32, round(height * scale / 32) * 32),
        max(32, round(width * scale / 32) * 32),
    )
