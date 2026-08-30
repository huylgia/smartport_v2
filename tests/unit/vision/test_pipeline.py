"""Kiểm đường ống mã container mà không cần GPU, Triton hay model thật.

Hai hàm suy luận được thay bằng hàm giả, nên test chạy được ở CI. Thứ được kiểm ở đây là
**hình dạng luồng dữ liệu và các cửa lọc** — chỗ dễ sai nhất khi port, và là chỗ mà lỗi
sẽ biểu hiện thành "hệ thống chạy nhưng không đọc được gì".
"""

from __future__ import annotations

import numpy as np

from internal.pkg.nptypes import Array
from internal.pkg.vision import textcrop
from internal.pkg.vision.ccode_pipeline import CCodePipeline, RoiParams, Stats
from tests.unit.vision.conftest import CHAR_DICT, bitmap_with_box, make_logits, noisy_image

# ---------------------------------------------------------------- đường ống


class _FakeInfer:
    """Model giả: detector trả bitmap dựng sẵn, recognizer trả cùng một chuỗi."""

    def __init__(self, bitmap: Array, sequence: list[int]) -> None:
        self.bitmap = bitmap
        self.sequence = sequence
        self.rec_batch_sizes: list[int] = []

    def det(self, tensor: Array) -> Array:
        assert tensor.ndim == 4, "detector phải nhận 4 chiều"
        return self.bitmap[np.newaxis, np.newaxis]

    def rec(self, tensor: Array) -> Array:
        # Hợp đồng mới: UINT8 NHWC. Model *_folded tự Cast và Transpose bên trong.
        assert tensor.shape[1:] == (64, 256, 3), f"mong NHWC, nhận {tensor.shape}"
        assert tensor.dtype == np.uint8
        self.rec_batch_sizes.append(tensor.shape[0])
        return np.stack([make_logits(self.sequence)] * tensor.shape[0])


def test_pipeline_doc_duoc_va_gom_crop_thanh_mot_lan_goi() -> None:
    bitmap = bitmap_with_box((200, 300), (40, 50, 140, 90))
    bitmap += bitmap_with_box((200, 300), (180, 120, 280, 160))
    fake = _FakeInfer(bitmap, [11, 12, 13])

    pipeline = CCodePipeline(
        vertical=False, char_dict=CHAR_DICT, det_infer=fake.det, rec_infer=fake.rec
    )
    results, stats = pipeline.run(
        noisy_image((200, 300)),
        RoiParams(det_size=(200, 300), sharpness_min=0.0, score_threshold=0.0),
    )

    assert [r.text for r in results] == ["ABC", "ABC"]
    assert stats == Stats(detected=2, after_top_k=2, after_sharpness=2, recognized=2)
    # ⭐ Điểm chính: MỘT lời gọi rec cho cả hai crop, không phải hai — xem DN-007.
    assert fake.rec_batch_sizes == [2]


def test_pipeline_cong_net_loai_bo_crop_nhoe() -> None:
    """Ngưỡng nét cao ⇒ không crop nào qua ⇒ KHÔNG gọi recognizer."""
    bitmap = bitmap_with_box((200, 300), (40, 50, 140, 90))
    fake = _FakeInfer(bitmap, [11])

    pipeline = CCodePipeline(
        vertical=False, char_dict=CHAR_DICT, det_infer=fake.det, rec_infer=fake.rec
    )
    results, stats = pipeline.run(
        np.zeros((200, 300, 3), np.uint8),
        RoiParams(det_size=(200, 300), sharpness_min=1e9),
    )

    assert results == []
    assert stats.after_sharpness == 0
    assert fake.rec_batch_sizes == [], "không có crop nào thì đừng gọi model"


def test_pipeline_top_k_gioi_han_so_crop() -> None:
    bitmap = np.zeros((200, 400), np.float32)
    for i in range(4):
        bitmap += bitmap_with_box((200, 400), (10 + i * 90, 20, 70 + i * 90, 60 - i * 5))
    fake = _FakeInfer(bitmap, [11])

    pipeline = CCodePipeline(
        vertical=False, char_dict=CHAR_DICT, det_infer=fake.det, rec_infer=fake.rec
    )
    _, stats = pipeline.run(
        noisy_image((200, 400)),
        RoiParams(det_size=(200, 400), sharpness_min=0.0, score_threshold=0.0, top_k=2),
    )

    assert stats.detected == 4
    assert stats.after_top_k == 2
    assert fake.rec_batch_sizes == [2]


def test_pipeline_anh_rong_khong_no() -> None:
    fake = _FakeInfer(np.zeros((10, 10), np.float32), [11])
    pipeline = CCodePipeline(
        vertical=False, char_dict=CHAR_DICT, det_infer=fake.det, rec_infer=fake.rec
    )

    assert pipeline.run(np.zeros((0, 0, 3), np.uint8), RoiParams()) == ([], Stats())


def test_pipeline_khong_phat_hien_gi_thi_khong_goi_rec() -> None:
    fake = _FakeInfer(np.zeros((200, 300), np.float32), [11])
    pipeline = CCodePipeline(
        vertical=False, char_dict=CHAR_DICT, det_infer=fake.det, rec_infer=fake.rec
    )

    results, stats = pipeline.run(noisy_image((200, 300)), RoiParams(det_size=(200, 300)))

    assert results == []
    assert stats.detected == 0
    assert fake.rec_batch_sizes == []


def test_pipeline_bo_ket_qua_ocr_rong() -> None:
    """Điểm OCR dưới ngưỡng ⇒ chuỗi rỗng ⇒ không được lọt vào kết quả."""
    bitmap = bitmap_with_box((200, 300), (40, 50, 140, 90))
    fake = _FakeInfer(bitmap, [11, 12])

    pipeline = CCodePipeline(
        vertical=False, char_dict=CHAR_DICT, det_infer=fake.det, rec_infer=fake.rec
    )
    results, stats = pipeline.run(
        noisy_image((200, 300)),
        RoiParams(det_size=(200, 300), sharpness_min=0.0, score_threshold=0.99),
    )

    assert results == []
    assert stats.after_sharpness == 1
    assert stats.recognized == 0


def test_roi_params_tu_dict_thieu_khoa_dung_mac_dinh() -> None:
    got = RoiParams.from_mapping({"expand_ratio": [1.1, 1.7], "det_size": [576, 608]})

    assert got.expand_ratio == (1.1, 1.7)
    assert got.det_size == (576, 608)
    assert got.box_threshold == 0.2
    assert got.top_k == textcrop.DEFAULT_TOP_K


def _capture_det_input(*, vertical: bool) -> Array:
    """Chạy đường ống một lần, giữ lại đúng tensor đã gửi cho detector."""
    seen: list[Array] = []

    def det(tensor: Array) -> Array:
        seen.append(tensor.copy())
        return np.zeros((1, 1, 100, 100), np.float32)

    CCodePipeline(
        vertical=vertical,
        char_dict=CHAR_DICT,
        det_infer=det,
        rec_infer=lambda t: t,
    ).run(np.full((100, 100, 3), 255, np.uint8), RoiParams(det_size=(100, 100)))
    return seen[0]


def test_gui_pixel_tho_khong_dung_vao_gia_tri() -> None:
    """Model tự chuẩn hoá ⇒ Python KHÔNG được đụng vào giá trị pixel.

    Gửi nhầm dữ liệu đã chuẩn hoá vào model đã gấp thì nó vẫn chạy, vẫn trả về chuỗi, chỉ
    là chuỗi rác. Test này là chốt chặn duy nhất ở tầng đơn vị.
    """
    for vertical in (False, True):
        tensor = _capture_det_input(vertical=vertical)
        assert tensor.max() == 255, "phải là pixel thô [0,255]"
        assert tensor.dtype == np.uint8, "không được đổi kiểu — model tự Cast"
        assert tensor.shape == (1, 100, 100, 3), "NHWC — model tự Transpose"


def test_hai_huong_gui_cung_mot_tensor() -> None:
    """Khác biệt ngang/dọc nay nằm TRONG hai model, không còn ở Python."""
    assert np.array_equal(_capture_det_input(vertical=False), _capture_det_input(vertical=True))
