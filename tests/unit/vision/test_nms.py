"""Kiểm NMS cho đầu ra PicoDet."""

from __future__ import annotations

import numpy as np

from internal.pkg.vision.nms import iou, multiclass, suppress


def test_iou_hop_trung_khit_bang_1() -> None:
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    assert iou(box, box[np.newaxis]) == np.array([1.0])


def test_iou_hop_roi_nhau_bang_0() -> None:
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    far = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert iou(box, far)[0] == 0.0


def test_iou_dem_ca_hai_dau_pixel() -> None:
    """CỐ Ý: diện tích là ``(x2 - x1 + 1)``, không phải ``(x2 - x1)``.

    Hai hộp 10x10 chồng nhau đúng một nửa: nếu bỏ ``+1`` thì IoU = 1/3; với ``+1`` nó
    lệch khỏi 1/3. Test này neo lại lựa chọn đó để không ai "sửa" nó trong im lặng.
    """
    box = np.array([0, 0, 9, 9], dtype=np.float32)
    half = np.array([[5, 0, 14, 9]], dtype=np.float32)
    assert iou(box, half)[0] != 1 / 3
    assert 0.2 < iou(box, half)[0] < 0.4


def test_suppress_bo_hop_trung_giu_hop_diem_cao() -> None:
    dets = np.array(
        [
            [0, 0, 10, 10, 0.9],  # tốt nhất
            [1, 1, 11, 11, 0.8],  # trùng gần hết -> bỏ
            [50, 50, 60, 60, 0.7],  # rời hẳn -> giữ
        ],
        dtype=np.float32,
    )
    assert suppress(dets, 0.3) == [0, 2]


def test_suppress_ton_trong_nms_top_k() -> None:
    dets = np.array(
        [[i * 100, 0, i * 100 + 10, 10, 0.9 - i * 0.01] for i in range(10)], dtype=np.float32
    )
    assert len(suppress(dets, 0.3, nms_top_k=3)) == 3


def test_multiclass_gan_dung_chi_so_lop() -> None:
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([[0.9, 0.1], [0.1, 0.8]], dtype=np.float32)  # (C=2, N=2)

    dets = multiclass(boxes, scores, score_threshold=0.5)

    assert dets.shape == (2, 6)
    assert dets[0][-1] == 0  # hộp đầu thuộc lớp 0
    assert dets[1][-1] == 1  # hộp sau thuộc lớp 1


def test_multiclass_khong_co_gi_vuot_nguong_tra_mang_rong_dung_hinh_dang() -> None:
    """Nơi gọi lập chỉ mục ``dets[:, :-2]`` nên hình dạng phải đúng kể cả khi rỗng."""
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([[0.01]], dtype=np.float32)

    dets = multiclass(boxes, scores, score_threshold=0.5)

    assert dets.shape == (0, 6)
    assert len(dets) == 0
