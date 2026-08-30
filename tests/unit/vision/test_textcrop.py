"""Kiểm bước cắt/nắn ảnh chữ giữa detector và recognizer."""

from __future__ import annotations

import cv2
import numpy as np

from internal.pkg.vision import dbpost, textcrop

# ---------------------------------------------------------------- cắt / nắn ảnh


def test_sharpness_anh_net_cao_hon_anh_nhoe() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    assert textcrop.sharpness(np.asarray(sharp)) > textcrop.sharpness(blurred)


def test_equalize_brightness_giu_nguyen_kich_thuoc_va_kieu() -> None:
    img = np.full((32, 64, 3), 40, dtype=np.uint8)
    out = textcrop.equalize_brightness(img)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_warp_quad_nan_tu_giac_nghieng_ve_chu_nhat() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.fillPoly(img, [np.array([[10, 20], [80, 10], [90, 60], [20, 70]])], (255, 255, 255))
    quad = np.array([[10, 20], [80, 10], [90, 60], [20, 70]], dtype=np.float64)

    out = textcrop.warp_quad(img, quad)

    assert out.ndim == 3
    assert out.shape[0] > 1 and out.shape[1] > 1
    # Sau khi nắn, vùng trắng phải chiếm gần hết khung.
    assert (out > 128).mean() > 0.85


def test_prepare_crop_doc_xoay_90_do() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 30:50] = 255  # dải dọc: cao 60, rộng 20
    box = dbpost.TextBox(
        bbox=(30, 20, 49, 79),
        quad=np.array([[30, 20], [49, 20], [49, 79], [30, 79]], dtype=np.int32),
        score=0.9,
    )

    crop, _ = textcrop.prepare_crop(img, box, vertical=True)

    # Xoay 90 độ ⇒ cao và rộng đổi chỗ.
    assert crop.shape[0] < crop.shape[1]


def test_top_k_by_area_sap_giam_dan() -> None:
    def make(w: int) -> dbpost.TextBox:
        return dbpost.TextBox((0, 0, w, 10), np.zeros((4, 2), np.int32), 0.5)

    got = textcrop.top_k_by_area([make(10), make(50), make(30)], k=2)

    assert [b.bbox[2] for b in got] == [50, 30]
