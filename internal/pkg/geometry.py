"""Vùng đa giác và phép kiểm tra chứa điểm.

**Vùng là đa giác, không phải "phía nào của một đường".** Cách chia lane bằng hai đường
phân chia rồi xét dấu là cám dỗ thường trực vì nó chỉ cần bốn số. Nó hỏng ở bốn chỗ:

1. **Chỉ tạo được các dải song song.** Camera nhìn chéo từ trên xuống nên lane trong ảnh
   bị méo phối cảnh — hình thang, không phải dải song song.
2. **Mọi điểm trong ảnh đều thuộc một lane nào đó.** Không diễn đạt được "nằm ngoài khu
   vực làm hàng". Một xe chạy ngang phía xa vẫn bị gán vào lane 1 hoặc lane 3.
3. **Ranh giới ngầm định.** Lane 2 là "khoảng giữa hai đường" — không viết ra được. Thêm
   lane thứ 4 là phải suy lại toàn bộ.
4. **Phải lật dấu theo camera.** Hai camera nhìn cùng một khu vực từ hai hướng cần hai
   phép xét dấu **ngược nhau**, tức hai bản code cùng hình dạng — và một cờ đảo dấu mà ai
   đó sẽ quên. Với đa giác, mỗi camera chỉ khai vùng của nó.

**Toạ độ trong config là tương đối ``[0..1]``, không phải pixel.** Toạ độ tuyệt đối trói
config vào một độ phân giải xử lý cụ thể (camera 10 bị downscale về ``720p`` — xem
``HARDWARE_BUDGET.md`` §2.3), nên đổi độ phân giải là phải hiệu chỉnh lại toàn bộ vùng.
Việc quy đổi sang pixel xảy ra **đúng một lần** ở :meth:`LaneZones.from_config`, nơi biết
độ phân giải; sau đó mọi thứ ở runtime đều là pixel.

Module này **thuần**: không I/O, không log, không đọc config. Việc chuẩn hoá trả về cờ
:attr:`PolygonZone.was_sanitized` để nơi gọi tự ghi log.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import TYPE_CHECKING

from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.prepared import PreparedGeometry, prep

if TYPE_CHECKING:
    from common.enum import Lane

__all__ = [
    "Anchor",
    "LaneZones",
    "PolygonZone",
    "anchor_points",
    "denormalize",
]

XY = tuple[float, float]
BBoxXYXY = tuple[float, float, float, float]
PointSeq = Sequence[Sequence[float]]


def denormalize(points: PointSeq, frame_size: tuple[int, int]) -> list[XY]:
    """Quy đổi toạ độ tương đối ``[0..1]`` sang pixel.

    Args:
        points: Đỉnh dạng ``[[x, y], ...]`` với ``x``, ``y`` trong ``[0, 1]``.
        frame_size: ``(width, height)`` tính bằng pixel của **khung mà model nhìn thấy**.

    Raises:
        ValueError: nếu một toạ độ nằm ngoài ``[0, 1]``, hoặc ``frame_size`` không dương.
            Thông báo lỗi nêu luôn nguyên nhân hay gặp nhất là dán nhầm toạ độ pixel.
    """
    w, h = frame_size
    if w <= 0 or h <= 0:
        raise ValueError(f"frame_size phải dương, nhận {frame_size}")

    out: list[XY] = []
    for i, pt in enumerate(points):
        if len(pt) != 2:
            raise ValueError(f"đỉnh {i} phải có đúng 2 toạ độ, nhận {list(pt)}")
        x, y = float(pt[0]), float(pt[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"đỉnh {i} = ({x}, {y}) nằm ngoài [0, 1]. Config của v2 dùng toạ độ "
                f"**tương đối**, không phải pixel — nếu đây là toạ độ pixel thì chia cho "
                f"kích thước khung ({w}x{h})."
            )
        out.append((x * w, y * h))
    return out


class Anchor(Flag):
    """Điểm nào của bbox phải nằm trong vùng. Kết hợp bằng ``|``.

    Yêu cầu cả hai nghĩa là **cả hai** điểm đều phải nằm trong — chặt hơn, dùng khi muốn
    chắc chắn vật thể thực sự ở trong vùng chứ không phải ló nửa người vào.
    """

    CENTER = auto()
    """Tâm bbox — mặc định, và đủ dùng khi camera nhìn từ trên xuống."""

    BOTTOM = auto()
    """Tâm cạnh đáy bbox — điểm chạm đất khi camera nhìn ngang."""


def anchor_points(bbox: BBoxXYXY, anchor: Anchor) -> list[XY]:
    """Các điểm mốc của một bbox theo :class:`Anchor` đã chọn."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    points: list[XY] = []
    if anchor & Anchor.CENTER:
        points.append((cx, (y1 + y2) / 2))
    if anchor & Anchor.BOTTOM:
        points.append((cx, y2))
    if not points:
        raise ValueError("anchor phải chứa ít nhất một trong CENTER hoặc BOTTOM")
    return points


@dataclass(frozen=True, slots=True)
class PolygonZone:
    """Một vùng đa giác đã chuẩn hoá, cache sẵn prepared geometry.

    Dùng :meth:`from_points` để tạo — hàm đó lo phần chuẩn hoá.
    """

    polygons: tuple[Polygon, ...]
    """Một vùng logic có thể tách thành nhiều đa giác sau khi chuẩn hoá (đa giác tự cắt
    hình zigzag sẽ thành MultiPolygon). Mọi mảnh đều thuộc cùng một vùng."""

    was_sanitized: bool = False
    """``True`` nếu đa giác gốc không hợp lệ và đã bị ``buffer(0)`` sửa.

    ⚠️ Chỉ có thể là ``True`` khi nơi gọi truyền ``sanitize=True``. Mặc định là **từ chối**,
    vì việc sửa có thể **mất dữ liệu** — xem :meth:`from_points`."""

    _prepared: tuple[PreparedGeometry, ...] = field(default_factory=tuple, repr=False)

    @classmethod
    def from_points(cls, points: PointSeq, *, sanitize: bool = False) -> PolygonZone:
        """Dựng vùng từ danh sách đỉnh.

        **Mặc định từ chối đa giác không hợp lệ** (tự cắt), thay vì gọi ``buffer(0)`` để
        "sửa" rồi chạy tiếp. Phép sửa đó **có thể mất dữ liệu mà không lộ ra**.

        Ví dụ đã đo: hình nơ ``[[0,0],[100,100],[100,0],[0,100]]`` gồm hai tam giác, mỗi
        tam giác diện tích 2500. ``buffer(0)`` trả về **một** Polygon diện tích 2500 —
        **mất hẳn một nửa**. Với vùng lane thì đó là mất nửa làn xe, và không ai biết.

        Vì vùng lane là config do người vẽ và sửa được, cách đúng là **fail-fast** để người
        vận hành vẽ lại, chứ không phải im lặng chạy tiếp với nửa vùng.

        Args:
            points: Danh sách đỉnh ``[[x, y], ...]``, tối thiểu 3.
            sanitize: Đặt ``True`` để sửa bằng ``buffer(0)`` thay vì ném lỗi. Chỉ dùng khi
                chạy tiếp với vùng méo còn hơn là dừng hẳn.

        Raises:
            ValueError: nếu ít hơn 3 đỉnh, diện tích 0, hoặc đa giác không hợp lệ mà
                ``sanitize=False``.
        """
        if len(points) < 3:
            raise ValueError(f"đa giác cần ít nhất 3 đỉnh, nhận {len(points)}")

        poly = Polygon(points)
        sanitized = False

        if poly.is_valid:
            if poly.area == 0:
                raise ValueError(f"đa giác có diện tích 0: {list(points)}")
            parts: tuple[Polygon, ...] = (poly,)
        else:
            # Cả hai kiểu hỏng đều cho shoelace area = 0, nên không phân biệt được bằng
            # `.area`. Dùng buffer(0): đỉnh thẳng hàng cho hình RỖNG, còn hình tự cắt
            # (nơ) cho hình khác rỗng. Đã đo bằng shapely 2.1.
            fixed = poly.buffer(0)
            if fixed.is_empty:
                raise ValueError(f"đa giác có diện tích 0 (các đỉnh thẳng hàng?): {list(points)}")
            if not sanitize:
                raise ValueError(
                    f"đa giác không hợp lệ (thường là các cạnh tự cắt nhau): "
                    f"{list(points)}. Sửa lại toạ độ trong config; truyền sanitize=True "
                    f"nếu chấp nhận để buffer(0) sửa, nhưng phép sửa đó có thể làm mất "
                    f"một phần vùng."
                )
            sanitized = True
            parts = tuple(fixed.geoms) if isinstance(fixed, MultiPolygon) else (fixed,)

        return cls(
            polygons=parts,
            was_sanitized=sanitized,
            _prepared=tuple(prep(p) for p in parts),
        )

    def contains(self, point: XY) -> bool:
        """Điểm có nằm trong vùng không (bất kỳ mảnh nào)."""
        p = Point(point)
        return any(pre.contains(p) for pre in self._prepared)

    def contains_bbox(self, bbox: BBoxXYXY, anchor: Anchor = Anchor.CENTER) -> bool:
        """Mọi điểm mốc của bbox đều nằm trong vùng."""
        return all(self.contains(pt) for pt in anchor_points(bbox, anchor))

    @property
    def area(self) -> float:
        return float(sum(p.area for p in self.polygons))

    def overlaps(self, other: PolygonZone) -> bool:
        """Có chồng lấn với vùng khác không — dùng để validate config."""
        return any(
            a.intersects(b) and a.intersection(b).area > 0
            for a in self.polygons
            for b in other.polygons
        )


class LaneZones:
    """Ánh xạ lane → vùng, trả về lane chứa một điểm.

    Thay ``lane_position_config`` (hai đường) cũ. Điểm không nằm trong lane nào trả về
    ``None`` — điều mà cách dùng đường **không** diễn đạt được, vì với hai đường thì mọi
    điểm trong ảnh đều rơi vào một dải nào đó.
    """

    def __init__(self, zones: dict[Lane, PolygonZone]) -> None:
        self._zones = dict(zones)

    @classmethod
    def from_config(
        cls,
        raw: Mapping[str, PointSeq],
        *,
        frame_size: tuple[int, int],
        sanitize: bool = False,
    ) -> LaneZones:
        """Dựng từ config dạng ``{"1": [[x,y], ...], "2": [...]}``, toạ độ **tương đối**.

        Đây là chỗ **duy nhất** quy đổi tương đối → pixel. Sau khi dựng xong, mọi truy vấn
        (:meth:`lane_at`, :meth:`lane_for_bbox`) đều nhận toạ độ **pixel**, cùng hệ với bbox
        do model trả về.

        Args:
            raw: Ánh xạ lane → đỉnh đa giác, toạ độ trong ``[0, 1]``.
            frame_size: ``(width, height)`` của khung mà model nhìn thấy.
            sanitize: Xem :meth:`PolygonZone.from_points`.

        Raises:
            ValueError: nếu khoá lane không hợp lệ, toạ độ ngoài ``[0, 1]``, hoặc một đa
                giác không dựng được. Thông báo luôn nêu rõ lane nào gây lỗi.
        """
        from common.enum import Lane as _Lane

        zones: dict[_Lane, PolygonZone] = {}
        for key, points in raw.items():
            try:
                lane = _Lane(key)
            except ValueError:
                valid = ", ".join(sorted(m.value for m in _Lane))
                raise ValueError(f"lane {key!r} không hợp lệ; hợp lệ: {valid}") from None
            try:
                pixels = denormalize(points, frame_size)
                zones[lane] = PolygonZone.from_points(pixels, sanitize=sanitize)
            except ValueError as exc:
                raise ValueError(f"lane {key!r}: {exc}") from None
        return cls(zones)

    def lane_at(self, point: XY) -> Lane | None:
        """Lane chứa điểm, hoặc ``None`` nếu nằm ngoài mọi lane.

        Nếu config có vùng chồng lấn thì kết quả là lane đầu tiên theo thứ tự khai báo —
        nhưng chồng lấn nên bị chặn từ lúc load config, xem :meth:`overlapping_lanes`.
        """
        p = Point(point)
        for lane, zone in self._zones.items():
            if any(pre.contains(p) for pre in zone._prepared):
                return lane
        return None

    def lane_for_bbox(self, bbox: BBoxXYXY, anchor: Anchor = Anchor.CENTER) -> Lane | None:
        """Lane chứa **mọi** điểm mốc của bbox.

        Mặc định ``CENTER``: camera cẩu nhìn gần như thẳng xuống nên tâm bbox đầu kéo là
        điểm đại diện tốt. Camera nhìn ngang thì truyền ``BOTTOM``.
        """
        points = anchor_points(bbox, anchor)
        candidates = {self.lane_at(pt) for pt in points}
        if len(candidates) == 1:
            return candidates.pop()
        return None

    def overlapping_lanes(self) -> list[tuple[Lane, Lane]]:
        """Các cặp lane bị chồng lấn.

        Đây là kiểu lỗi **mới** mà đa giác tạo ra còn đường thì không thể có: hai lane có
        thể vẽ đè lên nhau. Phải kiểm ở bước validate config, không để lộ ra lúc chạy.
        """
        items = list(self._zones.items())
        return [
            (a_lane, b_lane)
            for i, (a_lane, a) in enumerate(items)
            for b_lane, b in items[i + 1 :]
            if a.overlaps(b)
        ]

    def sanitized_lanes(self) -> list[Lane]:
        """Các lane có đa giác phải sửa lúc dựng — nơi gọi nên log cảnh báo."""
        return [lane for lane, zone in self._zones.items() if zone.was_sanitized]

    def __len__(self) -> int:
        return len(self._zones)

    def __contains__(self, lane: Lane) -> bool:
        return lane in self._zones

    def __getitem__(self, lane: Lane) -> PolygonZone:
        return self._zones[lane]
