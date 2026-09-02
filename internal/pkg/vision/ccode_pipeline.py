"""Đường ống nhận dạng mã container: det → hậu xử lý → cắt/nắn → rec → CTC.

Đây là phần "solution" của nhánh ccode, tách khỏi mọi thứ liên quan tới Triton. Nó nhận
hai hàm suy luận qua tham số, nên test được bằng model giả trên máy không có GPU; bản
chạy thật truyền vào hai hàm gọi BLS (``triton/repo/craneops/craneops_ccode_*/1/model.py``).

**Vì sao không phải ``ensemble`` của Triton:**

Giữa det và rec có ba bước mà đồ thị tĩnh không diễn đạt được —

1. chọn top-5 hộp theo diện tích (số hộp vào thay đổi từng khung),
2. nắn phối cảnh theo 4 đỉnh của **từng** hộp,
3. **cổng nét ảnh loại bỏ bớt crop** ⇒ số crop ra ≠ số hộp vào.

Ensemble truyền tensor qua các bước với hình dạng cố định theo lược đồ; bước (3) làm số
phần tử thay đổi theo dữ liệu. BLS (Business Logic Scripting) sinh ra đúng cho việc này.

**Vẫn giữ được cái lợi lớn nhất.** Điều lo ngại khi bỏ ensemble là mất ``dynamic_batching``.
Không mất: mỗi lần chạy gom TOÀN BỘ crop còn sống thành **một** lời gọi rec duy nhất, và
các lời gọi từ những instance BLS khác nhau — tức các camera khác nhau — vẫn được bộ gom
batch của Triton nhập lại. Xem ``docs/DESIGN_NOTES.md`` DN-007.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from internal.pkg.vision import dbpost, preprocess, textcrop
from internal.pkg.vision.ctc import CtcConfig, decode
from internal.pkg.vision.preprocess import batch_to_tensor, to_tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from internal.pkg.nptypes import Array, Image

import cv2

REC_INPUT_SIZE = (64, 256)
"""``(cao, rộng)`` đầu vào recognizer SVTR — khớp ``Shape("x", (3, 64, 256))`` trong SPECS."""

DEFAULT_DET_SIZE = (352, 640)
"""``(cao, rộng)`` mặc định cho detector, từ ``CCRecognizer.input_size``. Mỗi ROI trong config ghi đè giá trị riêng."""


@dataclass(frozen=True, slots=True)
class RoiParams:
    """Tham số của MỘT vùng OCR. Tương ứng một phần tử ``ocr_rois`` trong config cẩu."""

    det_size: tuple[int, int] | None = None
    """Bỏ trống ⇒ suy từ vùng bằng :func:`~internal.pkg.vision.preprocess.fit_long_side`."""

    det_long_side: int = preprocess.DET_LONG_SIDE
    bitmap_threshold: float = 0.1
    box_threshold: float = 0.2
    expand_ratio: tuple[float, float] = (1.0, 1.0)
    character_threshold: float = 0.3
    score_threshold: float = 0.8
    sharpness_min: float = textcrop.DEFAULT_SHARPNESS_MIN
    top_k: int = textcrop.DEFAULT_TOP_K

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> RoiParams:
        """Dựng từ dict (JSON đi kèm request). Khoá thiếu ⇒ dùng mặc định."""

        def number(key: str, default: float) -> float:
            value = raw.get(key, default)
            if not isinstance(value, (int, float)):
                msg = f"{key} phải là số, nhận được {value!r}"
                raise TypeError(msg)
            return float(value)

        def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            value = raw.get(key)
            if value is None:
                return default
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                msg = f"{key} phải là cặp hai số, nhận được {value!r}"
                raise TypeError(msg)
            return (float(value[0]), float(value[1]))

        raw_size = raw.get("det_size")
        return cls(
            # Bỏ trống ⇒ để `run` tự suy từ vùng. KHÔNG đặt mặc định cứng ở đây: một giá
            # trị cố định áp cho mọi vùng sẽ bóp méo tỉ lệ của phần lớn chúng.
            det_size=(lambda p: (int(p[0]), int(p[1])))(pair("det_size", (0.0, 0.0)))
            if raw_size is not None
            else None,
            det_long_side=int(number("det_long_side", float(preprocess.DET_LONG_SIDE))),
            bitmap_threshold=number("bitmap_threshold", 0.1),
            box_threshold=number("box_threshold", 0.2),
            expand_ratio=pair("expand_ratio", (1.0, 1.0)),
            character_threshold=number("character_threshold", 0.3),
            score_threshold=number("score_threshold", 0.8),
            sharpness_min=number("sharpness_min", textcrop.DEFAULT_SHARPNESS_MIN),
            top_k=int(number("top_k", textcrop.DEFAULT_TOP_K)),
        )


@dataclass(frozen=True, slots=True)
class TextResult:
    """Một vùng chữ đã đọc được."""

    text: str
    score: float
    """Độ tin cậy OCR (trung bình xác suất ký tự)."""
    box: tuple[int, int, int, int]
    quad: Array = field(repr=False)
    det_score: float
    sharpness: float


@dataclass(frozen=True, slots=True)
class Stats:
    """Đếm số hộp rơi rụng ở từng cửa — để chẩn đoán khi hệ thống "không đọc được gì"."""

    detected: int = 0
    after_top_k: int = 0
    after_sharpness: int = 0
    recognized: int = 0


class CCodePipeline:
    """Chạy trọn một khung ROI. Không giữ trạng thái giữa các lần gọi.

    Args:
        vertical: mã dọc hay ngang. Quyết định bộ hằng số chuẩn hoá của detector và
            cách nắn ảnh crop — hai thứ này phải khớp nhau, nên chỉ có MỘT cờ.
        char_dict: bảng ký tự, chỉ số bắt đầu từ 1 (0 là blank).
        det_infer: ``(1, H, W, 3) uint8 → bitmap`` hình dạng bất kỳ, sẽ được ép về
            ``(H, W)``.
        rec_infer: ``(N, 64, 256, 3) uint8 → (N, 25, 37) float32``.
    """

    def __init__(
        self,
        *,
        vertical: bool,
        char_dict: Mapping[int, str],
        det_infer: Callable[[NDArray[np.float32]], Array],
        rec_infer: Callable[[NDArray[np.float32]], Array],
    ) -> None:
        self._vertical = vertical
        self._char_dict = char_dict
        self._det_infer = det_infer
        self._rec_infer = rec_infer

    def run(self, image: Image, params: RoiParams) -> tuple[list[TextResult], Stats]:
        """Ảnh ROI (BGR, uint8) → danh sách vùng chữ đọc được."""
        if image.size == 0:
            return [], Stats()

        # Suy kích thước từ CHÍNH vùng đã cắt, không lấy từ config: nó tự đúng khi camera
        # đổi độ phân giải, và không ai phải dò lại mỗi lần vẽ một vùng mới.
        det_size = params.det_size or preprocess.fit_long_side(
            image.shape[0], image.shape[1], params.det_long_side
        )
        tensor, scale = to_tensor(image, det_size, interpolation=cv2.INTER_CUBIC)
        bitmap = np.asarray(self._det_infer(tensor), dtype=np.float32).reshape(det_size)

        boxes = dbpost.decode(
            bitmap,
            dbpost.DbPostConfig(
                bitmap_threshold=params.bitmap_threshold,
                box_threshold=params.box_threshold,
                expand_ratio=params.expand_ratio,
            ),
            scale_factor=scale,
            image_size=(image.shape[1], image.shape[0]),
        )
        detected = len(boxes)
        if not boxes:
            return [], Stats(detected=0)

        boxes = textcrop.top_k_by_area(boxes, params.top_k)

        # Cổng nét ảnh: chỉ crop đủ nét mới được đưa vào OCR. Đây là chỗ số phần tử
        # thay đổi theo dữ liệu — lý do đường ống này là BLS chứ không phải ensemble.
        kept: list[tuple[dbpost.TextBox, float]] = []
        crops: list[Image] = []
        for box in boxes:
            crop, sharp = textcrop.prepare_crop(image, box, vertical=self._vertical)
            if crop.size == 0 or sharp < params.sharpness_min:
                continue
            kept.append((box, sharp))
            crops.append(crop)

        if not crops:
            return [], Stats(detected, len(boxes), 0, 0)

        # MỘT lời gọi cho tất cả crop — xem DN-007. Gọi từng crop một ở đây sẽ vô hiệu
        # hoá cả dynamic_batching của Triton lẫn batch của chính engine.
        logits = np.asarray(
            self._rec_infer(batch_to_tensor(crops, REC_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)),
            dtype=np.float32,
        )

        ctc_cfg = CtcConfig(
            character_threshold=params.character_threshold,
            score_threshold=params.score_threshold,
        )
        results = []
        for (box, sharp), row in zip(kept, logits, strict=True):
            out = decode(row, self._char_dict, ctc_cfg)
            if not out.text:
                continue
            results.append(
                TextResult(
                    text=out.text,
                    score=out.score,
                    box=box.bbox,
                    quad=box.quad,
                    det_score=box.score,
                    sharpness=sharp,
                )
            )

        return results, Stats(detected, len(boxes), len(crops), len(results))
