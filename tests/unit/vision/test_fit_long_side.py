"""Suy kích thước detector từ vùng, thay cho việc dò tay từng vùng."""

from __future__ import annotations

import pytest

from internal.pkg.vision.preprocess import DET_LONG_SIDE, fit_long_side


def test_it_preserves_the_region_aspect_ratio() -> None:
    """Đây là quy luật v1 tuân theo mà không viết ra: đo lại 20 vùng của v1 thấy tỉ lệ
    ``input_size`` khớp tỉ lệ vùng, lệch trung vị **2,0 %** — toàn bộ là do làm tròn 32."""
    for h, w in ((581, 610), (605, 1020), (873, 785), (296, 363)):
        out_h, out_w = fit_long_side(h, w)
        assert abs((out_h / out_w) - (h / w)) / (h / w) < 0.05


def test_the_long_side_lands_on_the_target() -> None:
    for h, w in ((400, 800), (800, 400), (640, 640)):
        assert max(fit_long_side(h, w, 640)) == 640


def test_every_side_is_a_multiple_of_32() -> None:
    """Detector DB hạ mẫu 5 lần; kích thước lẻ khiến nó tự đệm và bản đồ xác suất lệch."""
    for h, w in ((581, 610), (137, 999), (33, 31), (1520, 2688)):
        out = fit_long_side(h, w)
        assert out[0] % 32 == 0 and out[1] % 32 == 0, (h, w, out)


def test_a_sliver_never_collapses_to_zero() -> None:
    """Làm tròn xuống một cạnh rất mỏng sẽ cho 0, và detector nhận tensor rỗng."""
    assert min(fit_long_side(5, 2000)) >= 32


def test_it_reproduces_two_of_v1s_hand_tuned_sizes() -> None:
    """⚠️ Đây là đối chứng: nếu công thức KHÔNG khớp chỗ nào của v1 thì nó chỉ là một con
    số bịa. Camera 1514 có hai vùng mà v1 chỉnh tay ra đúng thứ công thức tính được.
    """
    assert fit_long_side(605, 1020, 640) == (384, 640)
    assert fit_long_side(873, 785, 640) == (640, 576)


def test_a_degenerate_region_is_rejected() -> None:
    with pytest.raises(ValueError, match="phải dương"):
        fit_long_side(0, 100)
    with pytest.raises(ValueError, match="long_side"):
        fit_long_side(100, 100, 0)


def test_the_default_sits_inside_the_measured_plateau() -> None:
    """Đo trên 4 mẫu có đồng thuận: 576-832 cho 3-4/4, còn <=544 tụt về 1/4. Mặc định phải
    nằm trên sàn đó, không phải ở rìa."""
    assert 576 <= DET_LONG_SIDE <= 832
