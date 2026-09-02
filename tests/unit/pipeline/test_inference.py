"""Hàng đợi suy luận: không bao giờ chặn probe, và phần mất mát phải đếm được."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from common.enum import CameraRole
from ds_app.src.pipeline.inference import (
    BLS_FOR_ROLE,
    FrameJob,
    InferenceClient,
    output_names,
    to_detections,
)


def a_job(code: str = "GC03_1_2_3_4_1517", role: CameraRole = CameraRole.CRANE) -> FrameJob:
    return FrameJob(
        camera_code=code,
        role=role,
        frame_id=1,
        frame_ts=1788283539.0,
        start_ts=1788283530.0,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class FakeClient:
    """Client Triton giả. ``block`` giữ worker lại để test được lúc hàng đợi đầy."""

    def __init__(self, block: threading.Event | None = None) -> None:
        self.calls = 0
        self._block = block

    def infer(self, *_a: Any, **_k: Any) -> Any:
        self.calls += 1
        if self._block is not None:
            self._block.wait(timeout=5)
        raise AssertionError("test này không nên tới đây")


# ---------------------------------------------------------------- bảng model


def test_only_roles_with_a_model_are_listed() -> None:
    """``bottom``/``evidence_only`` không decode nên không bao giờ có khung để gửi."""
    assert set(BLS_FOR_ROLE) == {CameraRole.CRANE, CameraRole.TCODE}
    assert CameraRole.BOTTOM not in BLS_FOR_ROLE
    assert CameraRole.EVIDENCE_ONLY not in BLS_FOR_ROLE


def test_every_listed_role_actually_runs_a_model() -> None:
    """Đối chứng: bảng này không được chứa role mà pipeline sẽ không decode."""
    assert all(role.runs_model for role in BLS_FOR_ROLE)


# ---------------------------------------------------------------- hàng đợi


def test_submit_never_blocks_when_the_queue_is_full() -> None:
    """Probe chạy trên thread đẩy của GStreamer. Chặn nó là chặn cả batch — và với
    ``leaky`` ở hàng đợi phía trên, hệ quả là MẤT KHUNG chứ không phải chậm."""
    client = InferenceClient("x", lambda *_: None, workers=0, queue_size=2)

    started = time.perf_counter()
    accepted = [client.submit(a_job()) for _ in range(5)]
    elapsed = time.perf_counter() - started

    assert accepted == [True, True, False, False, False]
    assert elapsed < 0.05, f"submit đã chặn {elapsed:.3f}s"


def test_dropped_frames_are_counted_not_swallowed() -> None:
    """Bỏ khung im lặng là cách hỏng tệ nhất: hệ thống trông vẫn chạy, chỉ kém dần."""
    client = InferenceClient("x", lambda *_: None, workers=0, queue_size=1)
    for _ in range(4):
        client.submit(a_job())

    stats = client.stats.snapshot()
    assert stats["submitted"] == 1
    assert stats["dropped"] == 3
    assert not client.stats.clean


def test_a_clean_run_reports_clean() -> None:
    client = InferenceClient("x", lambda *_: None, workers=0, queue_size=4)
    assert client.submit(a_job())
    assert client.stats.clean


# ---------------------------------------------------------------- worker


def test_a_failing_call_does_not_kill_the_worker() -> None:
    """Các camera dùng chung worker; một khung hỏng không được kéo theo phần còn lại."""
    done = threading.Event()
    seen: list[int] = []

    class Flaky:
        n = 0

        def infer(self, *_a: Any, **_k: Any) -> Any:
            Flaky.n += 1
            seen.append(Flaky.n)
            if Flaky.n <= 2:
                raise RuntimeError("Triton đang bực")
            done.set()
            raise RuntimeError("vẫn bực, nhưng worker phải còn sống")

    client = InferenceClient("x", lambda *_: None, workers=1, client_factory=lambda _u: Flaky())
    client.start()
    try:
        for _ in range(3):
            client.submit(a_job())
        assert done.wait(timeout=5), "worker chết sau lần lỗi đầu"
    finally:
        client.stop()

    assert len(seen) == 3
    assert client.stats.snapshot()["failed"] == 3


def test_stop_returns_even_when_the_queue_is_full() -> None:
    """⚠️ Hồi quy: bản đầu dừng bằng "viên thuốc độc" ``put_nowait(None)``.

    Hàng đợi đầy đúng lúc dừng thì ``put_nowait`` ném ``Full``, worker chờ mãi một tín
    hiệu không bao giờ tới, và ``stop()`` hết giờ. Giờ worker dùng ``get(timeout=...)``.
    """
    release = threading.Event()
    client = InferenceClient(
        "x",
        lambda *_: None,
        workers=1,
        queue_size=1,
        client_factory=lambda _u: FakeClient(block=release),
    )
    client.start()
    for _ in range(4):
        client.submit(a_job())

    started = time.perf_counter()
    release.set()
    client.stop(timeout=3)
    elapsed = time.perf_counter() - started

    assert elapsed < 3, f"stop() mất {elapsed:.2f}s — worker không thoát"


# ---------------------------------------------------------------- đọc kết quả


def test_a_tcode_result_carries_the_head_code_as_a_flat_attr() -> None:
    """``Detection.attrs`` phẳng vì DeepStream gắn kết quả SGIE lên chính object meta."""
    out = to_detections(
        {
            "labels": np.array([b"head"], dtype=object),
            "scores": np.array([0.94], dtype=np.float32),
            "boxes": np.array([[10, 20, 30, 40]], dtype=np.int32),
            "codes": np.array([5], dtype=np.int32),
            "code_scores": np.array([0.86], dtype=np.float32),
        },
        CameraRole.TCODE,
    )

    assert len(out) == 1
    assert out[0].class_name == "head"
    assert out[0].confidence == pytest.approx(0.94)
    assert out[0].bbox.as_tuple() == (10, 20, 30, 40)
    assert out[0].attrs == {"headcode_05": pytest.approx(0.86)}


def test_an_unread_head_code_produces_no_attr_at_all() -> None:
    """``codes == -1`` nghĩa là không đọc được. Gửi nó đi sẽ thành ``headcode_-1``, và
    rule sẽ nhận nó như một số xe thật."""
    out = to_detections(
        {
            "labels": np.array([b"head"], dtype=object),
            "scores": np.array([0.94], dtype=np.float32),
            "boxes": np.array([[0, 0, 1, 1]], dtype=np.int32),
            "codes": np.array([-1], dtype=np.int32),
            "code_scores": np.array([0.0], dtype=np.float32),
        },
        CameraRole.TCODE,
    )

    assert out[0].attrs == {}


def test_a_crane_result_has_no_code_fields() -> None:
    out = to_detections(
        {
            "labels": np.array([b"container"], dtype=object),
            "scores": np.array([0.7], dtype=np.float32),
            "boxes": np.array([[1, 2, 3, 4]], dtype=np.int32),
        },
        CameraRole.CRANE,
    )

    assert out[0].class_name == "container"
    assert out[0].attrs == {}


def test_only_tcode_asks_for_the_classifier_outputs() -> None:
    """Hỏi ``codes`` ở ``craneops_crane`` là hỏi một output không có — Triton từ chối cả
    request, nên đây không phải chuyện thừa vài byte."""
    assert output_names(CameraRole.CRANE) == ["labels", "scores", "boxes"]
    assert output_names(CameraRole.TCODE) == [
        "labels",
        "scores",
        "boxes",
        "codes",
        "code_scores",
    ]


def test_an_empty_result_is_a_valid_answer() -> None:
    """Khung không có gì là chuyện thường; nó phải ra danh sách rỗng, không phải lỗi."""
    empty = to_detections(
        {
            "labels": np.array([], dtype=object),
            "scores": np.array([], dtype=np.float32),
            "boxes": np.empty((0, 4), dtype=np.int32),
        },
        CameraRole.CRANE,
    )
    assert empty == []
