"""Helper dựng dữ liệu giả, dùng chung cho test nhánh thị giác.

Đặt ở ``conftest.py`` thay vì import chéo giữa các file test: pytest tự nạp nó cho mọi
test trong thư mục, và không file test nào phải biết file test khác tồn tại.
"""

from __future__ import annotations

import numpy as np

from internal.pkg.nptypes import Array, Image

NUM_CLASSES = 37
"""Recognizer SVTR: 1 lớp blank (chỉ số 0) + 36 ký tự 0-9A-Z."""

CHAR_DICT = {i + 1: c for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def bitmap_with_box(
    shape: tuple[int, int], box: tuple[int, int, int, int], value: float = 0.9
) -> Array:
    """Bitmap xác suất giả với đúng một vùng chữ hình chữ nhật."""
    bitmap = np.zeros(shape, dtype=np.float32)
    x1, y1, x2, y2 = box
    bitmap[y1:y2, x1:x2] = value
    return bitmap


def make_logits(sequence: list[int], prob: float = 0.9, length: int = 25) -> Array:
    """Ma trận logit giả cho một chuỗi chỉ số ký tự; phần đuôi là blank.

    Tên có tiền tố ``make_`` để không che biến cục bộ tên ``logits`` trong test — đã từng
    va phải chuyện đó.
    """
    out = np.full((length, NUM_CLASSES), (1.0 - prob) / (NUM_CLASSES - 1), np.float32)
    for pos, idx in enumerate(sequence):
        out[pos] = (1.0 - prob) / (NUM_CLASSES - 1)
        out[pos, idx] = prob
    for pos in range(len(sequence), length):
        out[pos, 0] = prob
    return out


def noisy_image(shape: tuple[int, int]) -> Image:
    """Ảnh nhiễu — cần độ nét cao để qua được cổng lọc trong ``prepare_crop``."""
    return np.random.default_rng(1).integers(0, 255, (*shape, 3), dtype=np.uint8)
