from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from common.enum import (
    CameraRole,
    ContainerDim,
    ContainerPosition,
    Direction,
    IxCd,
    Lane,
    SignalKind,
)
from common.message import (
    SCHEMA_VERSION,
    BBox,
    ContainerSlot,
    ControlAction,
    ControlMessage,
    Detection,
    EventMessage,
    EvidenceJob,
    EvidenceJobMessage,
    EvidenceKind,
    ManifestEntry,
    ManifestMessage,
    OcrResult,
    PerceptionMessage,
    Signal,
    Topic,
    decode,
    encode,
    model_for_topic,
    perception_topic,
)

TS = 1_756_312_837.4


def _perception(**over: object) -> PerceptionMessage:
    kw: dict[str, object] = {
        "crane_id": "GC03",
        "camera_code": "GC03_113_160_225_15_1508",
        "role": CameraRole.CCODE,
        "frame_id": 300,
        "start_ts": 1_756_312_827.4,
        "fps": 30.0,
        "frame_ts": TS,
    }
    kw.update(over)
    return PerceptionMessage(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------- Topic


def test_every_topic_has_a_model() -> None:
    """Thêm topic mới mà quên map sẽ nổ ở đây, không phải lúc chạy."""
    for topic in Topic:
        assert model_for_topic(topic) is not None


def test_perception_topic_covers_every_model_role() -> None:
    for role in CameraRole:
        if role.runs_model:
            assert perception_topic(role).value.startswith("craneops.perception.")


@pytest.mark.parametrize("role", [CameraRole.BOTTOM, CameraRole.EVIDENCE_ONLY])
def test_perception_topic_rejects_roles_without_a_model(role: CameraRole) -> None:
    """Camera chỉ ghi hình không sinh message perception nào — hỏi topic của nó là lỗi logic."""
    with pytest.raises(ValueError, match="không chạy model"):
        perception_topic(role)


def test_topic_names_are_stable() -> None:
    """Đổi tên topic là thay đổi phá vỡ tương thích — chốt lại bằng test."""
    assert Topic.SIGNALS.value == "craneops.signals"
    assert Topic.MANIFEST.value == "craneops.manifest"
    assert Topic.EVIDENCE_FAST.value == "craneops.evidence.fast"
    assert Topic.EVIDENCE_SLOW.value == "craneops.evidence.slow"


# ---------------------------------------------------------------- BBox


def test_bbox_geometry() -> None:
    b = BBox(x1=505, y1=81, x2=1115, y2=662)  # ROI cam 1 thật của GC03
    assert b.width == 610
    assert b.height == 581
    assert b.area == 610 * 581
    assert b.center == (810.0, 371.5)


def test_bbox_rejects_inverted() -> None:
    """Hộp lật ngược không làm gì nổ ở hạ nguồn — chỉ cho diện tích âm và IoU vô nghĩa."""
    with pytest.raises(ValidationError, match="lật ngược"):
        BBox(x1=100, y1=0, x2=50, y2=10)
    with pytest.raises(ValidationError, match="lật ngược"):
        BBox(x1=0, y1=100, x2=10, y2=50)


def test_bbox_allows_degenerate() -> None:
    """Hộp rộng/cao bằng 0 hợp lệ — detector đôi khi trả về vậy."""
    assert BBox(x1=5, y1=5, x2=5, y2=5).area == 0


def test_bbox_from_xyxy_roundtrip() -> None:
    xyxy = (1.0, 2.0, 3.0, 4.0)
    assert BBox.from_xyxy(xyxy).as_tuple() == xyxy


# ---------------------------------------------------------------- schema_version


def test_schema_version_defaults() -> None:
    assert _perception().schema_version == SCHEMA_VERSION


def test_same_major_is_accepted() -> None:
    """Thêm trường tuỳ chọn ⇒ tăng minor ⇒ vẫn phải đọc được."""
    assert _perception(schema_version="1.7").schema_version == "1.7"


def test_different_major_is_rejected() -> None:
    with pytest.raises(ValidationError, match="lệch phiên bản"):
        _perception(schema_version="2.0")


# ---------------------------------------------------------------- forward compat


def test_unknown_fields_are_ignored_not_rejected() -> None:
    """Quy tắc 4: nâng cấp lệch pha không được giết consumer cũ.

    ds_app cập nhật trước ruled và thêm trường mới ⇒ ruled phải bỏ qua, không chết.
    """
    payload = _perception().model_dump()
    payload["truong_moi_tu_phien_ban_sau"] = 123
    msg = decode(Topic.PERCEPTION_CCODE, json.dumps(payload).encode())
    assert isinstance(msg, PerceptionMessage)


def test_missing_required_field_is_rejected() -> None:
    payload = _perception().model_dump()
    del payload["frame_ts"]
    with pytest.raises(ValidationError):
        decode(Topic.PERCEPTION_CCODE, json.dumps(payload).encode())


def test_wrong_type_is_rejected() -> None:
    payload = _perception().model_dump()
    payload["camera_code"] = ""
    with pytest.raises(ValidationError):
        decode(Topic.PERCEPTION_CCODE, json.dumps(payload).encode())


# ---------------------------------------------------------------- encode/decode


def test_encode_decode_roundtrip() -> None:
    msg = _perception(
        detections=[
            Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), class_name="container", confidence=0.93)
        ],
        ocr=[
            OcrResult(
                roi_index=0,
                shape="vertical",
                lane=Lane.ONE,
                cont_dim=ContainerDim.FT40,
                bbox=BBox(x1=612, y1=190, x2=704, y2=540),
                text="MSKU1234567",
                confidence=0.97,
            )
        ],
    )
    back = decode(Topic.PERCEPTION_CCODE, encode(msg))
    assert back == msg


def test_decode_rejects_broken_json() -> None:
    with pytest.raises(ValueError, match="không phải JSON hợp lệ"):
        decode(Topic.SIGNALS, b"{ khong phai json")


def test_decode_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="phải là object JSON"):
        decode(Topic.SIGNALS, b"[1, 2, 3]")


def test_encoded_payload_uses_enum_values_not_names() -> None:
    """Cái bẫy đã chốt ở test_enum: "ccode" chứ không phải "CameraRole.CCODE"."""
    raw = json.loads(encode(_perception()))
    assert raw["role"] == "ccode"


# ---------------------------------------------------------------- PerceptionMessage


def test_ocr_only_allowed_for_ccode_role() -> None:
    ocr = OcrResult(
        roi_index=0,
        shape="horizontal",
        lane=Lane.TWO,
        cont_dim=ContainerDim.FT20,
        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        text="X",
        confidence=0.5,
    )
    with pytest.raises(ValidationError, match="không được mang kết quả OCR"):
        _perception(role=CameraRole.CRANE, ocr=[ocr])


def test_confidence_must_be_a_probability() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), class_name="c", confidence=bad)


def test_camera_code_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        _perception(camera_code="")


def test_fps_is_source_fps_not_effective() -> None:
    """Nguồn thật là 30 fps; decimate xuống 5 fps là việc của drop-frame-interval."""
    assert _perception().fps == 30.0


# ---------------------------------------------------------------- Signal


def test_signal_defaults_to_right_direction() -> None:
    s = Signal(
        rule_code="CCODE01",
        crane_id="GC03",
        camera_code="GC03_113_160_225_15_1508",
        lane=Lane.ONE,
        kind=SignalKind.CONTAINER_NO,
        frame_ts=TS,
    )
    assert s.direction is Direction.RIGHT_TO_LEFT
    assert s.confidence == 1.0


def test_signal_kind_must_be_a_known_enum() -> None:
    """Composition spec tham chiếu kind; chuỗi tự do sẽ khiến một trường im lặng rỗng."""
    with pytest.raises(ValidationError):
        Signal(
            rule_code="X",
            crane_id="GC03",
            camera_code="GC03_113_160_225_15_1508",
            lane=Lane.ONE,
            kind="go_sai_ten",  # type: ignore[arg-type]
            frame_ts=TS,
        )


# ---------------------------------------------------------------- Manifest


def test_empty_manifest_means_no_ship_not_failure() -> None:
    """Bến trống là trạng thái vận hành bình thường, KHÔNG phải lỗi."""
    m = ManifestMessage(crane_id="GC03", berth_no="TS03", synced_at=TS)
    assert m.is_empty


def test_non_empty_manifest() -> None:
    m = ManifestMessage(
        crane_id="GC03",
        berth_no="TS03",
        synced_at=TS,
        containers=[
            ManifestEntry(container_no="MSKU1234567", ix_cd=IxCd.IMPORT, cont_dim=ContainerDim.FT40)
        ],
    )
    assert not m.is_empty
    assert m.containers[0].ix_cd is IxCd.IMPORT


# ---------------------------------------------------------------- Evidence


def test_evidence_message_requires_at_least_one_job() -> None:
    with pytest.raises(ValidationError):
        EvidenceJobMessage(event_id="e1", crane_id="GC03", lane=Lane.ONE, anchor_ts=TS, jobs=[])


def test_evidence_slow_lane_carries_delay() -> None:
    """Clip phải chờ 20-40 s sau thao tác cẩu; ảnh thì không."""
    m = EvidenceJobMessage(
        event_id="e1",
        crane_id="GC03",
        lane=Lane.ONE,
        anchor_ts=TS,
        delay=40.0,
        jobs=[
            EvidenceJob(
                kind=EvidenceKind.CLIP,
                camera_code="GC03_113_160_225_15_1516",
                from_ts=TS - 35.0,
                to_ts=TS + 10.0,
            )
        ],
    )
    assert m.delay == 40.0


# ---------------------------------------------------------------- Event


def test_event_slots_replace_v1_duplicated_fields() -> None:
    """Thay containerNo/containerNo2, ixCd/ixCd_2, shortVideo/shortVideo2..."""
    ev = EventMessage(
        event_id="GC03-1756312837-1",
        crane_id="GC03",
        lane=Lane.ONE,
        direction=Direction.RIGHT_TO_LEFT,
        anchor_ts=TS,
        truck_no="45",
        slots=[
            ContainerSlot(
                container_no="MSKU1234567",
                ix_cd=IxCd.IMPORT,
                cont_position=ContainerPosition.FT20_1,
            ),
            ContainerSlot(
                container_no="TGHU7654321",
                ix_cd=IxCd.IMPORT,
                cont_position=ContainerPosition.FT20_2,
            ),
        ],
    )
    assert len(ev.slots) == 2
    assert ev.slots[1].chassis_code == "A"


def test_slot_chassis_code_is_derived_not_stored() -> None:
    """Không có trường chassis_position trên wire — suy từ cont_position."""
    assert "chassis_position" not in ContainerSlot.model_fields
    assert ContainerSlot(cont_position=ContainerPosition.FT40).chassis_code == ""
    assert ContainerSlot(cont_position=ContainerPosition.FT20_1).chassis_code == "F"
    assert ContainerSlot(cont_position=ContainerPosition.FT20_2).chassis_code == "A"


def test_slot_without_position_has_empty_chassis_code() -> None:
    """Rỗng, không phải None: dashboard đọc chuỗi rỗng là "không áp dụng"."""
    assert ContainerSlot().chassis_code == ""


def test_event_rejects_more_than_two_slots() -> None:
    """Twin-lift tối đa 2 container."""
    with pytest.raises(ValidationError):
        EventMessage(
            event_id="e",
            crane_id="GC03",
            lane=Lane.ONE,
            direction=Direction.RIGHT_TO_LEFT,
            anchor_ts=TS,
            slots=[ContainerSlot(), ContainerSlot(), ContainerSlot()],
        )


def test_event_with_no_slots_is_valid() -> None:
    """Sự kiện chưa nhận dạng được mã vẫn phải dựng được — để triage."""
    ev = EventMessage(
        event_id="e",
        crane_id="GC03",
        lane=Lane.ONE,
        direction=Direction.LEFT_TO_RIGHT,
        anchor_ts=TS,
    )
    assert ev.slots == []


# ---------------------------------------------------------------- Control


def test_reload_rule_requires_rule_code() -> None:
    with pytest.raises(ValidationError, match="bắt buộc có rule_code"):
        ControlMessage(crane_id="GC03", action=ControlAction.RELOAD_RULE, issued_at=TS)


def test_reload_rule_with_code_is_valid() -> None:
    m = ControlMessage(
        crane_id="GC03", action=ControlAction.RELOAD_RULE, rule_code="CCODE01", issued_at=TS
    )
    assert m.rule_code == "CCODE01"


@pytest.mark.parametrize(
    "action", [ControlAction.PAUSE, ControlAction.RESUME, ControlAction.RELOAD_CONFIG]
)
def test_other_actions_do_not_need_rule_code(action: ControlAction) -> None:
    assert ControlMessage(crane_id="GC03", action=action, issued_at=TS).rule_code is None


# ---------------------------------------------------------------- immutability


def test_messages_are_frozen() -> None:
    msg = _perception()
    with pytest.raises(ValidationError):
        msg.camera_code = "KHAC"


# ---------------------------------------------------------------- cửa sổ bằng chứng


def test_a_clip_job_is_self_contained() -> None:
    """Khoảng là **tuyệt đối**, nên đưa một job cho worker không phải kèm mốc neo.

    Bản trước mang độ lệch so với ``anchor_ts``: mỗi consumer phải tự cộng, và mỗi phép
    cộng là một chỗ có thể nhầm dấu — nhầm dấu cho ra một clip trông bình thường, chỉ lệch
    chỗ. Quy đổi giờ làm MỘT lần, ở orchestrator, nơi nó đọc config.
    """
    job = EvidenceJob(
        kind=EvidenceKind.CLIP,
        camera_code="GC03_1_2_3_4_1508",
        from_ts=1788283524.0,
        to_ts=1788283559.0,
    )
    assert job.span == (1788283524.0, 1788283559.0)


def test_an_image_job_has_no_span_because_it_uses_the_anchor() -> None:
    """``image`` chụp MỘT khoảnh khắc, và khoảnh khắc đó là ``anchor_ts`` của message."""
    job = EvidenceJob(kind=EvidenceKind.IMAGE, camera_code="GC03_1_2_3_4_1508")
    assert job.from_ts is None and job.to_ts is None
    with pytest.raises(ValueError, match="dùng anchor_ts"):
        _ = job.span


def test_an_image_job_with_a_span_is_rejected() -> None:
    """Một khoảng ở đây nghĩa là ai đó tưởng nó cắt video, và sẽ ngạc nhiên khi nhận ảnh."""
    with pytest.raises(ValueError, match="không nhận khoảng"):
        EvidenceJob(
            kind=EvidenceKind.IMAGE,
            camera_code="GC03_1_2_3_4_1508",
            from_ts=1.0,
            to_ts=2.0,
        )


def test_a_clip_job_without_a_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="phải có from_ts và to_ts"):
        EvidenceJob(kind=EvidenceKind.CLIP, camera_code="GC03_1_2_3_4_1508")


def test_an_inverted_span_is_rejected() -> None:
    """Bắt lúc dựng message, không phải lúc cắt clip."""
    with pytest.raises(ValueError, match="lật ngược hoặc rỗng"):
        EvidenceJob(
            kind=EvidenceKind.CLIP,
            camera_code="GC03_1_2_3_4_1508",
            from_ts=1788283559.0,
            to_ts=1788283524.0,
        )


def test_jobs_in_one_message_may_span_different_ranges() -> None:
    """Một sự kiện, nhiều camera, cửa sổ khác nhau — camera đáy lùi xa hơn nhiều."""
    anchor = 1788283544.0
    ccode = EvidenceJob(
        kind=EvidenceKind.CLIP,
        camera_code="GC03_1_2_3_4_1508",
        from_ts=anchor - 20,
        to_ts=anchor + 15,
    )
    bottom = EvidenceJob(
        kind=EvidenceKind.CLIP,
        camera_code="GC03_1_2_3_4_1516",
        from_ts=anchor - 35,
        to_ts=anchor + 10,
    )
    assert ccode.span == (anchor - 20, anchor + 15)
    assert bottom.span == (anchor - 35, anchor + 10)


def test_mosaic_still_requires_a_grid() -> None:
    with pytest.raises(ValueError, match="mosaic phải có grid"):
        EvidenceJob(
            kind=EvidenceKind.MOSAIC,
            camera_code="GC03_1_2_3_4_1516",
            from_ts=1.0,
            to_ts=2.0,
        )
