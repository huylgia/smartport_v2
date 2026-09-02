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

from collections.abc import Sequence
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


CODE_CONTEXT = 3.0
"""Cạnh vùng crop, tính theo **bội cạnh dài của chính mã container**.

Người vẽ vùng chỉ cần khoanh 4 điểm của mã; phần nới ra suy từ đây. Đó là việc lặp lại
được — khác hẳn việc vẽ một vùng rồi dò ``input_size`` cho nó.

**3 là mốc khởi đầu, chọn để bắt đầu gán nhãn**, không phải giá trị đã tối ưu. Nó cho crop
chỉ bằng **36 %** diện tích của 5, tức rẻ hơn nhiều — và số đo hiện có chưa đủ để bác bỏ nó.

⚠️ Nhưng số đo hiện có **nghiêng về giá trị lớn hơn**, và chỗ đáng lo là kiểu hỏng:

======  ======  =======  ======  ==========  ==================
``k``   đúng    **SAI**  sót     mã chiếm    chữ cao trong det
======  ======  =======  ======  ==========  ==================
3       1       **2**    11      33 %        **34 px**
4       3       1        10      25 %        26 px
5       4       1        9       20 %        21 px
======  ======  =======  ======  ==========  ==================

Cột "đúng" thiên lệch (mẫu chuẩn lấy từ chính vùng của v1), nhưng cột SAI thì ít thiên lệch
hơn — và ``k=3`` cho nhiều lần đọc **sai** nhất. Đọc sai nguy hiểm hơn sót: một mã sai qua
được ISO 6346 sẽ đi thẳng lên dashboard như sự thật.

Nghi phạm là **tỉ lệ chữ**: ở ``k=3`` mã cao 34 px trong đầu vào, so với **23 px** đo được
ở điểm vận hành. Phần lớn mức phóng đó đến từ việc vùng bị **cắt ở mép ảnh** — vùng nhỏ đi
nhưng ``fit_long_side`` vẫn kéo cạnh dài về 640, nên chữ to lên. Sửa được bằng cách suy
kích thước từ vùng DANH NGHĨA (trước khi cắt) thay vì vùng thực; chưa làm.

Hai con số đo được cho ngữ cảnh mà detector **đang dùng khi chạy đúng** là 4,89 và 5,0:

* Tính ngược 20 vùng của v1 (đang chạy ở cảng): vùng rộng gấp **4,89 lần** bề rộng mã.
* Điểm vận hành đo trên mẫu đọc được: mã chiếm **20 %** bề rộng đầu vào detector, tức
  vùng gấp 5 lần — và ở đó mã cao ~23 px trong đầu vào.

**Một nhãn cho mỗi (camera, lane, cont_dim) là đủ.** Mã nằm ở vị trí cố định trên
container, và container đỗ vào chỗ lặp lại được, nên vị trí mã chỉ dao động **0,09-0,67
lần** cạnh dài của nó giữa các chuyến xe (đo trên 4 nhóm mẫu GC03). Lề của ``k=5`` là 2
lần mỗi phía — dư gấp ba. Kiểm trực tiếp: vùng suy từ **một** nhãn phủ trọn **5/5** vị trí
mã của 5 chuyến khác nhau.

⚠️ Hai con số 4,89/5,0 nói vùng cần **bao nhiêu ngữ cảnh**, không nói mức tối thiểu. Chưa
đo được mức tối thiểu: mọi phép so hiện có đều lấy mẫu từ chính vùng của v1, nên vùng nào
khác nó cũng thua sẵn — thiên lệch chọn mẫu. Đo lại khi có nhãn 4 điểm gán tay, độc lập
với đường ống hiện tại. Xem ``docs/HARDWARE_BUDGET.md`` §6.2.
"""


def roi_from_code(
    quad: Sequence[tuple[float, float]],
    frame_width: int,
    frame_height: int,
    context: float = CODE_CONTEXT,
) -> tuple[float, float, float, float]:
    """4 điểm của mã container → vùng crop tương đối ``(x1, y1, x2, y2)``.

    Vùng **gần vuông** chứ không theo tỉ lệ dẹt của mã: đo trên cấu hình đang chạy thấy
    vùng 776x722 cho một mã 174x28. Mã dịch lên xuống giữa các chuyến xe nhiều hơn là
    dịch ngang, nên phần chừa theo chiều dọc phải tính theo cạnh DÀI của mã.

    Args:
        quad: 4 đỉnh ``(x, y)`` theo **pixel của khung**, thứ tự nào cũng được.
        context: cạnh vùng bằng bao nhiêu lần cạnh dài của mã.

    Vùng bị **cắt về trong khung**, nên mã nằm sát mép cho ra vùng lệch tâm — đúng như
    mong muốn: không có gì để lấy ngoài mép ảnh.
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if len(quad) != 4:
        raise ValueError(f"cần đúng 4 điểm, nhận {len(quad)}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"kích thước khung phải dương, nhận {(frame_width, frame_height)}")

    long_side = max(max(xs) - min(xs), max(ys) - min(ys))
    if long_side <= 0:
        raise ValueError(f"4 điểm suy biến thành một điểm/đường: {list(quad)}")

    half = long_side * context / 2.0
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    x1 = max(0.0, cx - half) / frame_width
    y1 = max(0.0, cy - half) / frame_height
    x2 = min(float(frame_width), cx + half) / frame_width
    y2 = min(float(frame_height), cy + half) / frame_height
    return (x1, y1, x2, y2)
