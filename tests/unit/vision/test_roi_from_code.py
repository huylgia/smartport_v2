"""Suy vùng crop từ 4 điểm của mã container."""

from __future__ import annotations

import pytest

from internal.pkg.vision.preprocess import CODE_CONTEXT, fit_long_side, roi_from_code

CODE = [(1000.0, 700.0), (1174.0, 700.0), (1174.0, 728.0), (1000.0, 728.0)]
"""Mã 174x28 px — kích thước trung vị đo được trên mẫu GC03."""


def test_it_lands_on_the_measured_operating_point() -> None:
    """⚠️ Đây là phép neo. Đo trên cấu hình đang chạy ở cảng: vùng 776x722 cho một mã
    174x28, và detector nhận 640x592 với mã chiếm ~20 % bề rộng.

    Công thức phải cho ra cùng vùng đó, nếu không nó chỉ là một con số bịa.
    """
    x1, y1, x2, y2 = roi_from_code(CODE, 2688, 1520)
    w, h = round((x2 - x1) * 2688), round((y2 - y1) * 1520)

    assert (w, h) == (870, 870)
    assert fit_long_side(h, w) == (640, 640)


def test_the_region_is_square_not_the_code_aspect() -> None:
    """Mã dẹt 6:1 nhưng vùng gần vuông: xe dịch lên xuống giữa các chuyến nhiều hơn dịch
    ngang, nên phần chừa dọc phải tính theo cạnh DÀI của mã."""
    x1, y1, x2, y2 = roi_from_code(CODE, 2688, 1520)
    assert (x2 - x1) * 2688 == pytest.approx((y2 - y1) * 1520, rel=0.01)


def test_it_stays_centred_on_the_code() -> None:
    x1, y1, x2, y2 = roi_from_code(CODE, 2688, 1520)
    assert (x1 + x2) / 2 * 2688 == pytest.approx(1087.0, abs=1.0)
    assert (y1 + y2) / 2 * 1520 == pytest.approx(714.0, abs=1.0)


def test_a_code_near_the_edge_gives_a_clipped_region() -> None:
    """Không có gì để lấy ngoài mép ảnh, nên vùng lệch tâm là đúng — không phải lỗi."""
    edge = [(10.0, 10.0), (184.0, 10.0), (184.0, 38.0), (10.0, 38.0)]

    x1, y1, x2, y2 = roi_from_code(edge, 2688, 1520)

    assert (x1, y1) == (0.0, 0.0)
    assert 0.0 < x2 <= 1.0 and 0.0 < y2 <= 1.0


def test_the_context_factor_scales_the_region() -> None:
    small = roi_from_code(CODE, 2688, 1520, context=2.0)
    big = roi_from_code(CODE, 2688, 1520, context=10.0)
    assert (big[2] - big[0]) > (small[2] - small[0]) * 4


def test_a_degenerate_quad_is_rejected() -> None:
    """4 điểm trùng nhau cho vùng rỗng, và detector nhận tensor rỗng."""
    with pytest.raises(ValueError, match="suy biến"):
        roi_from_code([(5.0, 5.0)] * 4, 2688, 1520)


def test_it_needs_exactly_four_points() -> None:
    with pytest.raises(ValueError, match="4 điểm"):
        roi_from_code(CODE[:3], 2688, 1520)


def test_the_default_matches_both_independent_measurements() -> None:
    """Tính ngược 20 vùng của v1 cho 4,89x; điểm vận hành cho 5,0x. Mặc định phải nằm giữa
    hai con số đó, không phải một giá trị chọn cho tiện."""
    assert 4.5 <= CODE_CONTEXT <= 6.0
