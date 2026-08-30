"""Kiểm hậu xử lý DB: bitmap xác suất → hộp chữ."""

from __future__ import annotations

import numpy as np
import pytest

from internal.pkg.vision import dbpost, textcrop
from tests.unit.vision.conftest import bitmap_with_box

# ---------------------------------------------------------------- hậu xử lý DB


def test_dbpost_tim_duoc_hop_chu_nhat_don_gian() -> None:
    bitmap = bitmap_with_box((200, 300), (50, 60, 150, 100))
    boxes = dbpost.decode(
        bitmap,
        dbpost.DbPostConfig(),
        scale_factor=(1.0, 1.0),
        image_size=(300, 200),
    )

    assert len(boxes) == 1
    x1, _, x2, _ = boxes[0].bbox
    # Giãn nở mask nới thêm ~1 px mỗi phía; chấp nhận sai số nhỏ.
    assert x1 == pytest.approx(50, abs=3)
    assert x2 == pytest.approx(150, abs=3)
    assert boxes[0].score > 0.8
    assert boxes[0].quad.shape == (4, 2)


def test_dbpost_bo_hop_duoi_nguong_box_threshold() -> None:
    """Vùng có xác suất thấp vẫn qua được ngưỡng bitmap nhưng phải rớt ở ngưỡng hộp."""
    bitmap = bitmap_with_box((200, 300), (50, 60, 150, 100), value=0.15)
    cfg = dbpost.DbPostConfig(bitmap_threshold=0.1, box_threshold=0.5)

    assert dbpost.decode(bitmap, cfg, scale_factor=(1.0, 1.0), image_size=(300, 200)) == []


def test_dbpost_quy_doi_ve_anh_goc_theo_scale_factor() -> None:
    """Toạ độ phải chia cho scale_factor để về khung ảnh gốc."""
    bitmap = bitmap_with_box((200, 300), (50, 60, 150, 100))
    boxes = dbpost.decode(
        bitmap,
        dbpost.DbPostConfig(),
        scale_factor=(0.5, 0.5),  # ảnh đã resize xuống một nửa
        image_size=(600, 400),
    )

    assert len(boxes) == 1
    x1, _, x2, _ = boxes[0].bbox
    assert x1 == pytest.approx(100, abs=6)
    assert x2 == pytest.approx(300, abs=6)


def test_dbpost_ap_san_15px_ngay_ca_khi_ty_le_bang_1() -> None:
    """Sàn 15 px được áp KỂ CẢ khi expand_ratio = (1,1). Xem dbpost._expand."""
    bitmap = bitmap_with_box((100, 100), (40, 40, 46, 46))
    boxes = dbpost.decode(
        bitmap,
        dbpost.DbPostConfig(min_size=1, expand_ratio=(1.0, 1.0)),
        scale_factor=(1.0, 1.0),
        image_size=(100, 100),
    )

    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0].bbox
    assert x2 - x1 >= 14, "sàn 15 px không được áp — hộp nhỏ sẽ bị cắt mất chân chữ"
    assert y2 - y1 >= 14


def test_dbpost_khong_sap_xep_ket_qua() -> None:
    """Thứ tự do findContours quyết định; việc chọn top-k là của textcrop."""
    bitmap = bitmap_with_box((200, 400), (20, 20, 60, 50))
    bitmap += bitmap_with_box((200, 400), (200, 100, 380, 180))
    boxes = dbpost.decode(
        bitmap, dbpost.DbPostConfig(), scale_factor=(1.0, 1.0), image_size=(400, 200)
    )

    assert len(boxes) == 2
    top = textcrop.top_k_by_area(boxes, 1)
    assert top[0].area == max(b.area for b in boxes)


def test_scale_factor_bang_0_bao_loi_thay_vi_toa_do_rac() -> None:
    """Chia cho 0 cho ra NaN, rồi ``astype(int32)`` biến nó thành hộp trông hợp lệ."""
    with pytest.raises(ValueError, match="phải dương"):
        dbpost.decode(
            np.ones((32, 32), np.float32),
            dbpost.DbPostConfig(),
            scale_factor=(0.0, 1.0),
            image_size=(32, 32),
        )
