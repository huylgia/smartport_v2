"""Producer bus: không chặn nơi gọi, và phần mất mát phải đếm được."""

from __future__ import annotations

from typing import Any

import pytest

from common.enum import CameraRole, Lane
from common.message import BBox, Detection, PerceptionMessage, Topic, perception_topic
from gateway.contract.bus import BusProducer


class FakeFuture:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self._cb: Any = None
        self._eb: Any = None

    def add_callback(self, fn: Any) -> None:
        if not self._fail:
            fn(object())

    def add_errback(self, fn: Any) -> None:
        if self._fail:
            fn(RuntimeError("broker từ chối"))


class FakeProducer:
    """Producer Kafka giả. ``raise_on_send`` mô phỏng bộ đệm đầy."""

    def __init__(self, *, fail: bool = False, raise_on_send: bool = False) -> None:
        self.sent: list[tuple[str, bytes, bytes | None]] = []
        self.warmed: list[str] = []
        self.flushed = False
        self._fail = fail
        self._raise = raise_on_send

    def send(self, topic: str, value: bytes, key: bytes | None = None) -> FakeFuture:
        if self._raise:
            raise BufferError("bộ đệm đầy")
        self.sent.append((topic, value, key))
        return FakeFuture(self._fail)

    def partitions_for(self, topic: str) -> set[int]:
        self.warmed.append(topic)
        return {0}

    def flush(self, timeout: float = 10.0) -> None:
        self.flushed = True

    def close(self, timeout: float = 10.0) -> None:
        pass


def a_message() -> PerceptionMessage:
    return PerceptionMessage(
        crane_id="GC03",
        camera_code="GC03_1_2_3_4_1517",
        role=CameraRole.CRANE,
        frame_id=9,
        start_ts=1788283530.0,
        fps=30.0,
        frame_ts=1788283530.3,
        detections=[
            Detection(
                class_name="head",
                confidence=0.94,
                bbox=BBox.from_xyxy((1.0, 2.0, 3.0, 4.0)),
            )
        ],
    )


def producer(**kw: Any) -> tuple[BusProducer, FakeProducer]:
    fake = FakeProducer(**kw)
    bus = BusProducer("x:9092", producer_factory=lambda: fake)
    return bus, fake


# ---------------------------------------------------------------- topic


def test_the_topic_comes_from_the_role_not_the_class() -> None:
    """⚠️ ``PerceptionMessage.topic`` là ClassVar cố định ở ``ccode``; topic thật phụ thuộc
    ROLE. Lấy từ message sẽ đẩy mọi khung crane lên topic ccode, và consumer ccode sẽ đọc
    được message hợp lệ — chỉ là của camera khác."""
    bus, fake = producer()
    bus.start()

    bus.publish(a_message(), topic=perception_topic(CameraRole.CRANE))

    assert fake.sent[0][0] == Topic.PERCEPTION_CRANE.value
    assert fake.sent[0][0] != PerceptionMessage.topic.value


def test_other_messages_use_their_own_topic() -> None:
    from common.message import EvidenceJob, EvidenceJobMessage, EvidenceKind

    bus, fake = producer()
    bus.start()
    message = EvidenceJobMessage(
        event_id="e1",
        crane_id="GC03",
        lane=Lane.ONE,
        anchor_ts=1788283544.0,
        jobs=[EvidenceJob(kind=EvidenceKind.IMAGE, camera_code="GC03_1_2_3_4_1517")],
    )

    bus.publish(message)

    assert fake.sent[0][0] == Topic.EVIDENCE_FAST.value


# ---------------------------------------------------------------- khoá


def test_the_key_pins_a_camera_to_one_partition() -> None:
    """Cùng khoá ⇒ cùng phân vùng ⇒ giữ thứ tự. Message của một camera tới consumer sai
    thứ tự sẽ phá mọi phép đếm chuỗi liên tiếp (``min_streak``)."""
    bus, fake = producer()
    bus.start()

    bus.publish(a_message(), key="GC03_1_2_3_4_1517", topic=Topic.PERCEPTION_CRANE)

    assert fake.sent[0][2] == b"GC03_1_2_3_4_1517"


# ---------------------------------------------------------------- nạp sẵn


def test_start_warms_metadata_for_the_topics_it_will_use() -> None:
    """⚠️ Hồi quy đo được: không nạp sẵn thì **2 message đầu mất** ở mỗi lần chạy, trong
    lúc client còn hỏi metadata cluster — ``max_block_ms`` thấp biến chờ thành lỗi."""
    bus, fake = producer()

    bus.start([Topic.PERCEPTION_CRANE, Topic.PERCEPTION_TCODE])

    assert fake.warmed == [Topic.PERCEPTION_CRANE.value, Topic.PERCEPTION_TCODE.value]


# ---------------------------------------------------------------- đếm


def test_a_send_that_raises_is_counted_and_swallowed() -> None:
    """Nơi gọi là thread suy luận; một message mất không đáng để mất luôn nhánh xử lý."""
    bus, _ = producer(raise_on_send=True)
    bus.start()

    assert bus.publish(a_message(), topic=Topic.PERCEPTION_CRANE) is False
    assert bus.stats.snapshot()["dropped"] == 1
    assert not bus.stats.clean


def test_acked_counts_only_what_the_broker_confirmed() -> None:
    """v1 bắn fire-and-forget và không bao giờ biết kết quả. ``queued`` vs ``acked`` là
    chỗ duy nhất phân biệt được "đã xếp" với "đã tới broker"."""
    bus, _ = producer()
    bus.start()

    bus.publish(a_message(), topic=Topic.PERCEPTION_CRANE)

    stats = bus.stats.snapshot()
    assert (stats["queued"], stats["acked"], stats["in_flight"]) == (1, 1, 0)
    assert bus.stats.clean


def test_a_broker_error_is_counted_separately_from_a_drop() -> None:
    bus, _ = producer(fail=True)
    bus.start()

    bus.publish(a_message(), topic=Topic.PERCEPTION_CRANE)

    stats = bus.stats.snapshot()
    assert (stats["queued"], stats["acked"], stats["failed"]) == (1, 0, 1)
    assert not bus.stats.clean


def test_publishing_before_start_is_a_loud_error() -> None:
    """Im lặng bỏ ở đây sẽ giống hệt "broker chết" trong bảng thống kê."""
    bus, _ = producer()
    with pytest.raises(RuntimeError, match="chưa start"):
        bus.publish(a_message(), topic=Topic.PERCEPTION_CRANE)


# ---------------------------------------------------------------- payload


def test_the_payload_round_trips_through_the_contract() -> None:
    """Đối chứng: nếu payload không đọc lại được thì mọi test trên đây chỉ đang đếm byte."""
    bus, fake = producer()
    bus.start()
    original = a_message()

    bus.publish(original, topic=Topic.PERCEPTION_CRANE)

    back = PerceptionMessage.model_validate_json(fake.sent[0][1])
    assert back == original
    assert back.detections[0].bbox.as_tuple() == (1.0, 2.0, 3.0, 4.0)
