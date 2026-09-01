from __future__ import annotations

import pytest

from common.enum import Direction, Lane
from internal.pkg.geometry import (
    Anchor,
    LaneZones,
    PolygonZone,
    anchor_points,
    denormalize,
    stop_side,
)

FRAME = (1280, 720)

# Ba lane hình thang, không chồng lấn — dạng gần với phối cảnh thật của camera nhìn chéo.
# Toạ độ TƯƠNG ĐỐI [0..1]: y = 0 / 1/3 / 2/3 / 1 tương ứng 0 / 240 / 480 / 720 px.
LANE_CFG: dict[str, list[list[float]]] = {
    "1": [[0.0, 0.0], [1.0, 0.0], [0.9375, 1 / 3], [0.0625, 1 / 3]],
    "2": [[0.0625, 1 / 3], [0.9375, 1 / 3], [0.8984, 2 / 3], [0.1016, 2 / 3]],
    "3": [[0.1016, 2 / 3], [0.8984, 2 / 3], [0.8594, 1.0], [0.1406, 1.0]],
}


def square(x0: float, y0: float, size: float) -> list[list[float]]:
    """Ô vuông theo PIXEL — dùng cho PolygonZone.from_points (vốn nhận pixel)."""
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]]


def nsquare(x0: float, y0: float, size: float) -> list[list[float]]:
    """Ô vuông theo toạ độ TƯƠNG ĐỐI — dùng cho LaneZones.from_config."""
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]]


# ---------------------------------------------------------------- denormalize


def test_denormalize_scales_to_pixels() -> None:
    assert denormalize([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]], FRAME) == [
        (0.0, 0.0),
        (1280.0, 720.0),
        (640.0, 180.0),
    ]


@pytest.mark.parametrize(
    "bad", [[[0.0, 0.0], [1.5, 0.0], [0.0, 1.0]], [[-0.1, 0.0], [1.0, 0.0], [0.0, 1.0]]]
)
def test_denormalize_rejects_out_of_range(bad: list[list[float]]) -> None:
    """Nguyên nhân hay gặp nhất: dán toạ độ pixel vào một trường mong đợi [0..1]."""
    with pytest.raises(ValueError, match=r"ngoài \[0, 1\]"):
        denormalize(bad, FRAME)


def test_denormalize_error_mentions_pixel_mistake() -> None:
    with pytest.raises(ValueError, match=r"chia cho kích thước khung"):
        denormalize([[0, 0], [1280, 0], [640, 720]], FRAME)


def test_denormalize_rejects_bad_frame_size() -> None:
    with pytest.raises(ValueError, match="frame_size phải dương"):
        denormalize([[0, 0], [1, 0], [0, 1]], (0, 720))


def test_denormalize_rejects_malformed_point() -> None:
    with pytest.raises(ValueError, match="đúng 2 toạ độ"):
        denormalize([[0, 0], [1, 0, 5], [0, 1]], FRAME)


def test_same_config_at_two_resolutions_is_geometrically_equivalent() -> None:
    """Đây là lý do chuẩn hoá: cùng một config chạy được ở mọi độ phân giải.

    Toạ độ tuyệt đối trói config vào một độ phân giải, nên đổi độ phân giải xử lý là phải
    hiệu chỉnh lại toàn bộ vùng — chính là thứ toạ độ tương đối tránh được.
    """
    at_720 = LaneZones.from_config(LANE_CFG, frame_size=(1280, 720))
    at_4mp = LaneZones.from_config(LANE_CFG, frame_size=(2688, 1520))

    # Cùng một điểm tương đối phải cho cùng một lane ở cả hai độ phân giải.
    for rel_x, rel_y in [(0.5, 0.16), (0.5, 0.5), (0.5, 0.83)]:
        assert at_720.lane_at((rel_x * 1280, rel_y * 720)) is at_4mp.lane_at(
            (rel_x * 2688, rel_y * 1520)
        )


# ---------------------------------------------------------------- anchor_points


def test_anchor_center() -> None:
    assert anchor_points((0, 0, 10, 20), Anchor.CENTER) == [(5.0, 10.0)]


def test_anchor_bottom_is_bottom_centre() -> None:
    assert anchor_points((0, 0, 10, 20), Anchor.BOTTOM) == [(5.0, 20.0)]


def test_anchor_both_returns_two_points() -> None:
    pts = anchor_points((0, 0, 10, 20), Anchor.CENTER | Anchor.BOTTOM)
    assert pts == [(5.0, 10.0), (5.0, 20.0)]


def test_anchor_empty_flag_rejected() -> None:
    with pytest.raises(ValueError, match="ít nhất một"):
        anchor_points((0, 0, 1, 1), Anchor(0))


# ---------------------------------------------------------------- PolygonZone


def test_zone_contains_point_inside_and_outside() -> None:
    z = PolygonZone.from_points(square(0, 0, 100))
    assert z.contains((50, 50))
    assert not z.contains((150, 50))


def test_zone_needs_three_vertices() -> None:
    with pytest.raises(ValueError, match="ít nhất 3 đỉnh"):
        PolygonZone.from_points([[0, 0], [1, 1]])


def test_zone_rejects_zero_area() -> None:
    """Ba điểm thẳng hàng — nhập sai config thường gặp."""
    with pytest.raises(ValueError, match=r"diện tích 0"):
        PolygonZone.from_points([[0, 0], [10, 0], [20, 0]])


BOWTIE = [[0, 0], [100, 100], [100, 0], [0, 100]]
"""Hình nơ tự cắt (pixel) — gồm hai tam giác, mỗi tam giác diện tích 2500."""

NBOWTIE = [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
"""Cùng hình nơ nhưng toạ độ tương đối, cho LaneZones.from_config."""


def test_invalid_polygon_is_rejected_by_default() -> None:
    """Fail-fast, KHÔNG tự "sửa" bằng buffer(0) — xem PolygonZone.from_points."""
    with pytest.raises(ValueError, match=r"không hợp lệ"):
        PolygonZone.from_points(BOWTIE)


def test_sanitize_is_lossy_which_is_why_it_is_opt_in() -> None:
    """Bằng chứng cho quyết định fail-fast: buffer(0) VỨT MẤT một nửa hình nơ.

    Hai tam giác 2500 + 2500, sau khi sửa chỉ còn một Polygon diện tích 2500.
    Với vùng lane thì đó là mất nửa làn xe mà không ai biết.
    """
    z = PolygonZone.from_points(BOWTIE, sanitize=True)
    assert z.was_sanitized
    assert z.area == pytest.approx(2500.0)  # không phải 5000
    assert len(z.polygons) == 1


def test_valid_polygon_is_not_flagged_as_sanitized() -> None:
    assert not PolygonZone.from_points(square(0, 0, 10)).was_sanitized


def test_zone_area() -> None:
    assert PolygonZone.from_points(square(0, 0, 10)).area == pytest.approx(100.0)


def test_zone_overlaps() -> None:
    a = PolygonZone.from_points(square(0, 0, 100))
    assert a.overlaps(PolygonZone.from_points(square(50, 50, 100)))
    assert not a.overlaps(PolygonZone.from_points(square(200, 200, 100)))


def test_touching_zones_do_not_count_as_overlapping() -> None:
    """Hai lane dùng chung cạnh là bình thường — không được coi là chồng lấn."""
    a = PolygonZone.from_points(square(0, 0, 100))
    b = PolygonZone.from_points(square(100, 0, 100))
    assert not a.overlaps(b)


def test_contains_bbox_with_both_anchors() -> None:
    z = PolygonZone.from_points(square(0, 0, 100))
    assert z.contains_bbox((40, 40, 60, 60), Anchor.CENTER | Anchor.BOTTOM)
    # Đáy thò ra ngoài ⇒ yêu cầu cả hai điểm thì trượt, chỉ tâm thì đạt.
    assert not z.contains_bbox((40, 40, 60, 140), Anchor.CENTER | Anchor.BOTTOM)
    assert z.contains_bbox((40, 10, 60, 110), Anchor.CENTER)


# ---------------------------------------------------------------- LaneZones


@pytest.fixture
def lanes() -> LaneZones:
    return LaneZones.from_config(LANE_CFG, frame_size=FRAME)


def test_lane_at_resolves_each_lane(lanes: LaneZones) -> None:
    assert lanes.lane_at((640, 120)) is Lane.ONE
    assert lanes.lane_at((640, 360)) is Lane.TWO
    assert lanes.lane_at((640, 600)) is Lane.THREE


def test_point_outside_every_lane_returns_none(lanes: LaneZones) -> None:
    """Đây là điều cách dùng ĐƯỜNG không diễn đạt được: mọi điểm đều rơi vào một dải.

    Xe chạy ngang ngoài khu vực làm hàng phải cho None, không phải lane 1 hay lane 3.
    """
    assert lanes.lane_at((5, 700)) is None
    assert lanes.lane_at((640, -50)) is None
    assert lanes.lane_at((2000, 360)) is None


def test_lane_for_bbox_uses_centre_by_default(lanes: LaneZones) -> None:
    """Mặc định CENTER: camera cẩu nhìn gần thẳng xuống, tâm bbox là điểm đại diện tốt."""
    assert lanes.lane_for_bbox((600, 320, 680, 400)) is Lane.TWO


def test_lane_for_bbox_returns_none_when_anchors_disagree(lanes: LaneZones) -> None:
    """bbox vắt qua hai lane ⇒ không quyết được, trả None thay vì đoán.

    Lane 1 kết thúc ở y=240 tại x=640. bbox dưới có tâm y=220 (lane 1) và đáy y=290 (lane 2).
    """
    straddling = (600.0, 150.0, 680.0, 290.0)
    assert lanes.lane_at((640, 220)) is Lane.ONE
    assert lanes.lane_at((640, 290)) is Lane.TWO
    assert lanes.lane_for_bbox(straddling, Anchor.CENTER | Anchor.BOTTOM) is None
    assert lanes.lane_for_bbox(straddling, Anchor.CENTER) is Lane.ONE


def test_no_overlap_in_a_well_formed_config(lanes: LaneZones) -> None:
    assert lanes.overlapping_lanes() == []


def test_overlapping_lanes_are_detected() -> None:
    """Kiểu lỗi MỚI mà đa giác tạo ra còn đường thì không thể có — phải bắt lúc load config."""
    bad = LaneZones.from_config(
        {"1": nsquare(0.0, 0.0, 0.4), "2": nsquare(0.2, 0.2, 0.4)}, frame_size=FRAME
    )
    assert bad.overlapping_lanes() == [(Lane.ONE, Lane.TWO)]


def test_sanitized_lanes_are_reported_for_logging() -> None:
    """internal/pkg thuần, không tự log — trả cờ để nơi gọi log."""
    z = LaneZones.from_config({"1": NBOWTIE}, frame_size=FRAME, sanitize=True)
    assert z.sanitized_lanes() == [Lane.ONE]


def test_from_config_rejects_invalid_polygon_by_default() -> None:
    with pytest.raises(ValueError, match=r"lane '1'"):
        LaneZones.from_config({"1": NBOWTIE}, frame_size=FRAME)


def test_from_config_rejects_unknown_lane_key() -> None:
    with pytest.raises(ValueError, match="lane '9' không hợp lệ"):
        LaneZones.from_config({"9": nsquare(0.0, 0.0, 0.1)}, frame_size=FRAME)


def test_from_config_error_names_the_offending_lane() -> None:
    with pytest.raises(ValueError, match="lane '2'"):
        LaneZones.from_config(
            {"1": nsquare(0.0, 0.0, 0.1), "2": [[0, 0], [1, 1]]}, frame_size=FRAME
        )


def test_mapping_protocol(lanes: LaneZones) -> None:
    assert len(lanes) == 3
    assert Lane.ONE in lanes
    assert lanes[Lane.ONE].area > 0


def test_partial_lane_config_is_allowed() -> None:
    """Cẩu chỉ khai 2 lane vẫn hợp lệ — num_lane khác nhau giữa các cẩu."""
    two = LaneZones.from_config(
        {"1": nsquare(0.0, 0.0, 0.4), "2": nsquare(0.0, 0.5, 0.4)}, frame_size=FRAME
    )
    assert len(two) == 2
    assert Lane.THREE not in two


# ---------------------------------------------------------------- mép xe dừng


@pytest.mark.parametrize(
    ("center_x", "expected"),
    [
        (0.0, Direction.RIGHT_TO_LEFT),
        (0.20, Direction.RIGHT_TO_LEFT),
        (0.35, Direction.RIGHT_TO_LEFT),  # đúng biên: vẫn hợp lệ
        (0.36, None),
        (0.50, None),  # giữa ảnh — xe kẹt hoặc dừng chờ, KHÔNG phải vị trí làm hàng
        (0.64, None),
        (0.65, Direction.LEFT_TO_RIGHT),
        (1.0, Direction.LEFT_TO_RIGHT),
    ],
)
def test_stop_side_classifies_by_edge(center_x: float, expected: Direction | None) -> None:
    assert stop_side(center_x, stop_band=0.35) == expected


def test_stop_side_rejects_the_middle_even_when_the_bbox_is_frozen() -> None:
    """⚠️ Điểm chính của cổng này: "bbox đứng yên" KHÔNG đồng nghĩa "xe vào đúng vị trí".

    Xe kẹt hoặc dừng chờ giữa khung hình cũng đứng yên đủ ``stable_duration``. Không có cổng
    vị trí thì nó mở cổng OCR cho một lane không có xe nào ở đúng chỗ — và 5 camera ccode
    chạy DB detection + SVTR recognition cho một khung hình chẳng có gì.
    """
    assert stop_side(0.5, stop_band=0.35) is None
    assert stop_side(0.5, stop_band=0.49) is None


def test_stop_band_widens_the_valid_region() -> None:
    """``stop_band`` khai theo từng camera vì bố trí mỗi cẩu mỗi khác."""
    assert stop_side(0.4, stop_band=0.35) is None
    assert stop_side(0.4, stop_band=0.45) is Direction.RIGHT_TO_LEFT


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.7, -0.1])
def test_stop_band_outside_zero_to_half_is_rejected(bad: float) -> None:
    """``>= 0.5`` làm hai dải chồng nhau ⇒ mọi vị trí đều "hợp lệ", tức cổng vô hiệu."""
    with pytest.raises(ValueError, match="stop_band"):
        stop_side(0.5, stop_band=bad)
