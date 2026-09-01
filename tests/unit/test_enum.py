from __future__ import annotations

import json

import pytest

from common.enum import (
    CameraRole,
    ContainerDim,
    ContainerPosition,
    Direction,
    IxCd,
    Lane,
    SignalKind,
    StrEnum,
)

ALL_ENUMS: list[type[StrEnum]] = [
    CameraRole,
    ContainerDim,
    ContainerPosition,
    Direction,
    IxCd,
    Lane,
    SignalKind,
]


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_str_returns_value_not_repr(enum_cls: type[StrEnum]) -> None:
    """Cái bẫy: `class F(str, Enum)` trần cho f"{F.A}" ra "F.A" thay vì "a".

    Lỗi đó chỉ lộ ra khi message đã nằm trên Kafka, nên chốt bằng test.
    """
    for member in enum_cls:
        assert str(member) == member.value
        assert f"{member}" == member.value


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_is_json_serialisable(enum_cls: type[StrEnum]) -> None:
    for member in enum_cls:
        assert json.loads(json.dumps(member)) == member.value


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_compares_equal_to_plain_string(enum_cls: type[StrEnum]) -> None:
    """Rule đọc config dạng chuỗi thô; so sánh phải hoạt động không cần ép kiểu."""
    for member in enum_cls:
        assert member == member.value


def test_camera_roles_cover_every_camera_in_gc03() -> None:
    assert {r.value for r in CameraRole} == {
        "ccode",
        "tcode",
        "crane",
        "bottom",
        "evidence_only",
    }


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (CameraRole.CCODE, True),
        (CameraRole.TCODE, True),
        (CameraRole.CRANE, True),
        (CameraRole.BOTTOM, False),
        (CameraRole.EVIDENCE_ONLY, False),
    ],
)
def test_runs_model_decides_which_cameras_get_decoded(role: CameraRole, expected: bool) -> None:
    """Chỉ 3/5 vai trò cần decode ⇒ 8 luồng thay vì 10. Xem HARDWARE_BUDGET §2.3."""
    assert role.runs_model is expected


def test_runs_model_is_defined_for_every_role() -> None:
    for role in CameraRole:
        assert isinstance(role.runs_model, bool)


def test_lane_values_match_v1_strings() -> None:
    """Chuỗi "1"/"2"/"3" là hợp đồng với dashboard — đổi là đổi API đối ngoại."""
    assert [lane.value for lane in Lane] == ["1", "2", "3"]


def test_ix_cd_matches_oracle_catos_codes() -> None:
    assert IxCd.IMPORT.value == "I"
    assert IxCd.EXPORT.value == "X"


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (ContainerPosition.FT40, ""),
        (ContainerPosition.FT20_1, "F"),
        (ContainerPosition.FT20_2, "A"),
    ],
)
def test_chassis_code_matches_v1_lookup_table(position: ContainerPosition, expected: str) -> None:
    """Mã gửi dashboard: 40feet → "", 20feet-1 → "F", 20feet-2 → "A"."""
    assert position.chassis_code == expected


def test_every_position_carries_its_chassis_code() -> None:
    """Thêm vị trí mới mà quên mã dashboard sẽ nổ ở đây, không phải lúc gửi lên e-port."""
    for position in ContainerPosition:
        assert isinstance(position.chassis_code, str)


def test_chassis_code_is_not_an_enum_member() -> None:
    """Annotation-only ⇒ không được biến thành thành viên enum."""
    assert "chassis_code" not in ContainerPosition.__members__


@pytest.mark.parametrize(
    ("cont_dim", "slot_index", "expected"),
    [
        # 40 ft chiếm cả rơ-moóc ⇒ chỉ một slot, slot_index bị bỏ qua
        (ContainerDim.FT40, 0, ContainerPosition.FT40),
        (ContainerDim.FT40, 1, ContainerPosition.FT40),
        (ContainerDim.FT40, 99, ContainerPosition.FT40),
        # 20 ft: slot 0 = gần đầu xe nhất
        (ContainerDim.FT20, 0, ContainerPosition.FT20_1),
        (ContainerDim.FT20, 1, ContainerPosition.FT20_2),
    ],
)
def test_for_slot_maps_ordinal_to_position(
    cont_dim: ContainerDim, slot_index: int, expected: ContainerPosition
) -> None:
    """slot_index là hạng khi sắp xếp theo khoảng cách tới đầu xe. Xem DESIGN_NOTES DN-001."""
    assert ContainerPosition.for_slot(cont_dim, slot_index) is expected


@pytest.mark.parametrize("slot_index", [2, 3, -1])
def test_for_slot_rejects_impossible_20ft_slot(slot_index: int) -> None:
    """Một rơ-moóc chở tối đa hai container 20 ft."""
    with pytest.raises(ValueError, match="slot_index"):
        ContainerPosition.for_slot(ContainerDim.FT20, slot_index)


def test_container_dim_values_match_v1() -> None:
    assert {d.value for d in ContainerDim} == {"20feet", "40feet"}


def test_truck_position_is_gone() -> None:
    """Không có TruckPosition: slot container xác định bằng khoảng cách container ↔ đầu
    xe, không bằng dải dọc mà đầu xe rơi vào. Xem DESIGN_NOTES DN-001."""
    import common.enum as enum_module

    assert not hasattr(enum_module, "TruckPosition")


def test_direction_is_a_closed_enum() -> None:
    """Chiều là enum đóng, không phải chỉ số 1/2 kèm tag chuỗi."""
    assert {d.value for d in Direction} == {"RIGHT_TO_LEFT", "LEFT_TO_RIGHT"}


def test_signal_kinds_are_unique() -> None:
    values = [k.value for k in SignalKind]
    assert len(values) == len(set(values))


def test_signal_kinds_cover_all_eight_rules() -> None:
    """Composition spec chỉ được tham chiếu các kind có ở đây — gõ sai sẽ khiến
    một trường im lặng không bao giờ được điền.

    Hai kind **cố ý không có**:
    - "chassis_position": suy 1:1 từ cont_position (xem ContainerPosition.chassis_code)
    - "truck_position": slot container xác định bằng khoảng cách, xem DN-001
    """
    assert {k.value for k in SignalKind} == {
        "lane_active",
        "truck_stable",
        "crane_op",
        "cont_dim",
        "cont_position",
        "container_no",
        "truck_no",
        "bottom_ready",
    }
