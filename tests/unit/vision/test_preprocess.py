"""Kiểm chuẩn bị đầu vào cho model — chỉ resize, không đụng giá trị pixel."""

from __future__ import annotations

import numpy as np
import pytest

from internal.pkg.vision.preprocess import batch_to_tensor, to_tensor

# ---------------------------------------------------------------- chuẩn hoá


def test_to_tensor_tra_ve_nhwc_uint8_va_scale_factor() -> None:
    img = np.full((100, 200, 3), 255, dtype=np.uint8)

    tensor, scale = to_tensor(img, (50, 100))

    assert tensor.shape == (1, 50, 100, 3)
    assert tensor.dtype == np.uint8
    assert scale == pytest.approx((0.5, 0.5))


def test_batch_to_tensor_gom_nhieu_anh_khac_kich_thuoc() -> None:
    images = [
        np.full((30, 90, 3), 100, np.uint8),
        np.full((70, 120, 3), 200, np.uint8),
    ]

    out = batch_to_tensor(images, (64, 256))

    assert out.shape == (2, 64, 256, 3)
    assert out.dtype == np.uint8


def test_batch_to_tensor_rong_van_dung_hinh_dang() -> None:
    assert batch_to_tensor([], (64, 256)).shape == (0, 64, 256, 3)


def test_anh_rong_bao_loi_thay_vi_chia_cho_khong() -> None:
    with pytest.raises(ValueError, match="rỗng"):
        to_tensor(np.zeros((0, 0, 3), np.uint8), (64, 64))


def test_kich_thuoc_dich_bang_0_bi_chan() -> None:
    """Config sai cho ra hệ số co giãn 0, rồi hậu xử lý DB sinh toạ độ NaN trong im lặng."""
    with pytest.raises(ValueError, match="phải dương"):
        to_tensor(np.zeros((10, 10, 3), np.uint8), (0, 64))
