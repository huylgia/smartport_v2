"""Đường ống PicoDet: tensor hoá, NMS, đưa toạ độ về ảnh gốc."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from internal.pkg.nptypes import Array
from internal.pkg.vision.pico import INPUT_SIZE, PicoParams, crop, detect, to_tensor

LABELS = {0: "head", 1: "container"}


def fake_infer(boxes: Array, scores: Array) -> Callable[[Array], tuple[Array, Array]]:
    """Bộ suy luận giả trả đúng cặp tensor mà model thật trả."""

    def _infer(_tensor: Array) -> tuple[Array, Array]:
        return boxes, scores

    return _infer


def one_box(
    box: tuple[float, float, float, float], score: float, cls: int, n_cls: int = 2
) -> tuple[Array, Array]:
    """Một hộp duy nhất vượt ngưỡng, phần còn lại điểm 0."""
    boxes = np.zeros((4, 4), dtype=np.float32)
    boxes[0] = box
    scores = np.zeros((n_cls, 4), dtype=np.float32)
    scores[cls, 0] = score
    return boxes, scores


# ---------------------------------------------------------------- tensor hoá


def test_tensor_is_bgr_float32_nchw_at_the_network_size() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor = to_tensor(image)
    assert tensor.shape == (1, 3, *INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_pixels_stay_in_0_255_because_the_graph_does_the_scaling() -> None:
    """DN-012: phép chia 255 nằm trong đồ thị ONNX. Chia thêm ở đây cho ra hộp rác mà
    không có exception nào để lần ra."""
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert to_tensor(image).max() > 1.0


def test_channel_order_is_preserved_not_swapped() -> None:
    """Ảnh BGR thuần lam ⇒ kênh 0 sáng, kênh 2 tối. Đảo kênh cũng đã ở trong đồ thị."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    tensor = to_tensor(image)
    assert tensor[0, 0].mean() > 250
    assert tensor[0, 2].mean() < 5


def test_an_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="rỗng"):
        to_tensor(np.zeros((0, 10, 3), dtype=np.uint8))


def test_cubic_and_linear_really_do_differ() -> None:
    """Đối chứng cho lựa chọn nội suy: nếu hai phép cho kết quả giống hệt thì việc chốt
    ``INTER_CUBIC`` là vô nghĩa, và test nào khẳng định điều ngược lại đang nói dối."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    cubic = to_tensor(image)
    linear = cv2.resize(image, INPUT_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    linear = linear.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    assert not np.allclose(cubic, linear)


# ---------------------------------------------------------------- phát hiện


def test_boxes_come_back_in_original_image_coordinates() -> None:
    """Model làm việc ở 416x416; hộp phải quy về khung hình thật."""
    # Hộp phủ nửa trái trên của khung mạng ⇒ nửa trái trên của ảnh gốc.
    boxes, scores = one_box((0, 0, 208, 208), 0.9, cls=0)
    image = np.zeros((832, 1664, 3), dtype=np.uint8)

    found = detect(image, fake_infer(boxes, scores), LABELS)

    assert len(found) == 1
    assert found[0].label == "head"
    assert found[0].box == (0, 0, 832, 416)


def test_a_class_missing_from_the_mapping_is_dropped() -> None:
    """``truckhead_pico`` trả 2 lớp nhưng chỉ lớp 0 có nghĩa; lớp 1 là rác lúc huấn luyện."""
    boxes, scores = one_box((10, 10, 100, 100), 0.9, cls=1)
    image = np.zeros((416, 416, 3), dtype=np.uint8)

    assert detect(image, fake_infer(boxes, scores), {0: "head"}) == []
    assert len(detect(image, fake_infer(boxes, scores), {1: "container"})) == 1


def test_nothing_above_threshold_gives_an_empty_list() -> None:
    boxes, scores = one_box((10, 10, 100, 100), 0.1, cls=0)
    image = np.zeros((416, 416, 3), dtype=np.uint8)
    assert detect(image, fake_infer(boxes, scores), LABELS) == []


def test_the_threshold_is_honoured_from_params() -> None:
    boxes, scores = one_box((10, 10, 100, 100), 0.5, cls=0)
    image = np.zeros((416, 416, 3), dtype=np.uint8)
    assert detect(image, fake_infer(boxes, scores), LABELS, PicoParams(score_threshold=0.6)) == []
    assert (
        len(detect(image, fake_infer(boxes, scores), LABELS, PicoParams(score_threshold=0.4))) == 1
    )


def test_negative_coordinates_are_clamped_to_zero() -> None:
    boxes, scores = one_box((-50, -30, 100, 100), 0.9, cls=0)
    image = np.zeros((416, 416, 3), dtype=np.uint8)

    found = detect(image, fake_infer(boxes, scores), LABELS)

    assert found[0].box[:2] == (0, 0)


def test_a_box_past_the_right_edge_is_left_alone() -> None:
    """⚠️ GIỮ NGUYÊN hành vi v1: chỉ kẹp cận dưới.

    Kẹp cả hai đầu hợp lý hơn, nhưng toạ độ này đi thẳng vào phép gán lane của ``CRANE01``
    nên đổi nó là đổi kết quả nghiệp vụ — để dành cho một thay đổi riêng có đo lại.
    """
    boxes, scores = one_box((10, 10, 500, 500), 0.9, cls=0)
    image = np.zeros((416, 416, 3), dtype=np.uint8)

    found = detect(image, fake_infer(boxes, scores), LABELS)

    assert found[0].box[2] > 416


# ---------------------------------------------------------------- cắt ảnh


def test_crop_takes_the_region_named_by_the_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:60, 30:90] = 255
    assert crop(image, (30, 20, 90, 60)).shape == (40, 60, 3)
    assert bool((crop(image, (30, 20, 90, 60)) == 255).all())


def test_a_box_past_the_edge_yields_a_shorter_crop_not_an_error() -> None:
    """numpy tự cắt bớt. Đây là lý do không kẹp cận trên trong ``detect`` mà vẫn an toàn."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop(image, (50, 50, 500, 500)).shape == (50, 50, 3)


def test_a_box_wholly_outside_gives_an_empty_array() -> None:
    """Nơi gọi phải kiểm ``.size``; ``TCodeModel`` dựa vào đúng điều này để giữ chỗ."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop(image, (200, 200, 300, 300)).size == 0
