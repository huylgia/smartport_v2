"""Gọi Triton **ngoài** luồng streaming của GStreamer.

Probe chạy trên thread đẩy dữ liệu của GStreamer. Mọi mili-giây nó tiêu là mili-giây
``nvstreammux`` không đẩy được batch kế, và với ``leaky`` đặt ở hàng đợi phía trên thì hậu
quả không phải "chậm" mà là **mất khung**, im lặng. Một lời gọi Triton mất ~18 ms
(HARDWARE_BUDGET §6.1) nên nó tuyệt đối không được nằm trong probe.

Ở đây dùng **thread**, không phải process, vì phần đắt là I/O mạng và nó nhả GIL. Đo được:
truyền chiếm ~16,6 ms trong 18,5 ms, và phần CPU giữ GIL còn lại nhỏ hơn nhiều so với ngân
sách 137 ms/khung ở tải mục tiêu. Process riêng sẽ phải pickle khung 12,26 MB qua ống —
đắt hơn chính thứ nó định tránh.

Hàng đợi **có chặn và xả bản mới**: khi Triton chậm hơn nguồn thì lựa chọn duy nhất là bỏ
khung, và bỏ khung mới giữ được thứ tự thời gian của phần đang xử lý. Phần bỏ được **đếm**
(:attr:`InferenceClient.dropped`) để nó không im lặng.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from common.enum import CameraRole
from common.message import BBox, Detection

if TYPE_CHECKING:
    from internal.pkg.nptypes import Image

__all__ = [
    "BLS_FOR_ROLE",
    "FrameJob",
    "InferenceClient",
    "InferenceStats",
    "output_names",
    "to_detections",
]

BLS_FOR_ROLE: dict[CameraRole, str] = {
    CameraRole.CRANE: "craneops_crane",
    CameraRole.TCODE: "craneops_tcode",
}
"""Model BLS phục vụ từng role. ``ccode`` sẽ vào đây ở Phase 3b.

``bottom`` và ``evidence_only`` KHÔNG có mặt, và đó là chủ ý: chúng không decode nên không
bao giờ có khung để gửi. Xem ``CameraRole.runs_model``."""


@dataclass(frozen=True, slots=True)
class FrameJob:
    """Một khung chờ suy luận. Ảnh đã được **chép** ra khỏi buffer GStreamer."""

    camera_code: str
    role: CameraRole
    frame_id: int
    frame_ts: float
    source_fps: float
    """fps của **camera này**, không phải của cẩu. Hai camera cùng cẩu có thể chạy khác
    nhịp — đo được 18 vs 30 trên GC03 — và gửi số của cẩu ra ngoài là nói dối trên dây."""

    start_ts: float
    """Gốc trục thời gian của camera. ``PerceptionMessage`` mang cả hai để nơi nhận kiểm
    được ``frame_ts == start_ts + frame_id / fps`` thay vì phải tin."""

    image: Image
    segment_hint: str | None = None


@dataclass
class InferenceStats:
    """Đếm để phần mất mát không im lặng.

    ``dropped`` khác 0 nghĩa là Triton không theo kịp nguồn — hoặc tăng ``instance_group``
    của model BLS, hoặc giảm ``model_fps`` của camera. Đừng chỉ nới hàng đợi: hàng đợi sâu
    hơn chỉ đổi mất-khung thành dữ liệu-cũ.
    """

    submitted: int = 0
    completed: int = 0
    dropped: int = 0
    failed: int = 0
    """Lời gọi Triton hỏng."""

    sink_failed: int = 0
    """Callback ``on_result`` ném. Khác ``failed``: suy luận đã xong, chỗ hỏng là nơi nhận."""

    last_error: str = ""
    """Thông báo lỗi ĐẦU TIÊN gặp phải. Một con số đếm nói rằng có hỏng; nó không nói hỏng
    ở đâu, và không có dòng này thì người vận hành chỉ còn cách đoán."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def note_error(self, exc: BaseException) -> None:
        """Ghi lỗi ĐẦU TIÊN, giữ nguyên các lỗi sau — cái đầu thường là nguyên nhân."""
        with self._lock:
            if not self.last_error:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "submitted": self.submitted,
                "completed": self.completed,
                "dropped": self.dropped,
                "failed": self.failed,
                "sink_failed": self.sink_failed,
            }

    @property
    def clean(self) -> bool:
        with self._lock:
            return self.dropped == 0 and self.failed == 0 and self.sink_failed == 0


class InferenceClient:
    """Nhận khung từ probe, gọi BLS tương ứng, giao kết quả cho ``on_result``.

    ``on_result`` chạy trên thread worker, **không** trên thread GStreamer — nơi nhận phải
    tự lo an toàn luồng. Đó là chủ ý: đưa kết quả ngược về thread streaming sẽ dựng lại
    đúng chỗ nghẽn mà lớp này sinh ra để tránh.
    """

    def __init__(
        self,
        url: str,
        on_result: Callable[[FrameJob, list[Detection]], None],
        *,
        workers: int = 2,
        queue_size: int = 8,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._url = url
        self._on_result = on_result
        # Nông có chủ ý. Sâu hơn không cứu được thông lượng — nó chỉ để khung cũ dần đi
        # trong hàng đợi rồi mới được xử lý, và một phán quyết lane dựa trên khung 2 giây
        # trước còn tệ hơn không có phán quyết nào.
        self._queue: queue.Queue[FrameJob | None] = queue.Queue(maxsize=queue_size)
        self._workers = workers
        self._client_factory = client_factory or self._default_client
        self.stats = InferenceStats()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    @staticmethod
    def _default_client(url: str) -> Any:
        import tritonclient.grpc as grpcclient

        return grpcclient.InferenceServerClient(url=url)

    def start(self) -> None:
        for i in range(self._workers):
            t = threading.Thread(target=self._run, name=f"infer-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def submit(self, job: FrameJob) -> bool:
        """Xếp một khung. Trả ``False`` nếu đã bỏ vì hàng đợi đầy.

        **Không bao giờ chặn** — nó được gọi từ probe.
        """
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.stats.bump("dropped")
            return False
        self.stats.bump("submitted")
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)

    # -- thread worker -------------------------------------------------------

    def _run(self) -> None:
        client = self._client_factory(self._url)
        while not self._stop.is_set():
            try:
                # Timeout chứ KHÔNG phải viên thuốc độc: hàng đợi có thể đang đầy đúng
                # lúc dừng, và khi đó `put_nowait(None)` ném Full — worker sẽ chờ mãi một
                # tín hiệu không bao giờ tới, rồi `join` hết giờ.
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                results = self._infer(client, job)
            except Exception as exc:
                # Một khung hỏng không được giết worker: camera khác dùng chung nó.
                self.stats.note_error(exc)
                self.stats.bump("failed")
                continue
            self.stats.bump("completed")

            try:
                self._on_result(job, results)
            except Exception as exc:
                # ⚠️ Callback nằm TRONG try, và đó không phải cẩn thận thừa: bản đầu để nó
                # ngoài, nên một lỗi ở nơi nhận (đo được: biến `bus` bị che, gọi `publish`
                # lên GStreamer Bus) giết luôn thread worker. Triệu chứng là `completed`
                # vẫn tăng tới lúc thread cuối chết rồi mọi thứ dừng hẳn — không exception
                # nào ra tới đâu, vì thread chết lặng lẽ.
                #
                # Đếm riêng: "Triton hỏng" và "nơi nhận hỏng" là hai vấn đề khác nhau và
                # cần hai cách sửa khác nhau.
                self.stats.note_error(exc)
                self.stats.bump("sink_failed")

    def _infer(self, client: Any, job: FrameJob) -> list[Detection]:
        import numpy as np
        import tritonclient.grpc as grpcclient

        names = output_names(job.role)
        inp = grpcclient.InferInput("image", list(job.image.shape), "UINT8")
        inp.set_data_from_numpy(np.ascontiguousarray(job.image))
        response = client.infer(
            BLS_FOR_ROLE[job.role],
            [inp],
            outputs=[grpcclient.InferRequestedOutput(n) for n in names],
        )
        return to_detections({n: response.as_numpy(n) for n in names}, job.role)


def output_names(role: CameraRole) -> list[str]:
    """Tensor cần hỏi ở model BLS của role này.

    Hỏi ``codes`` ở ``craneops_crane`` là hỏi một output không tồn tại — Triton từ chối cả
    request, nên danh sách phải đúng theo role.
    """
    names = ["labels", "scores", "boxes"]
    if role is CameraRole.TCODE:
        names += ["codes", "code_scores"]
    return names


def to_detections(out: dict[str, Any], role: CameraRole) -> list[Detection]:
    """Tensor của BLS → :class:`~common.message.Detection` đã hợp lệ.

    Trả model chứ không trả ``dict``: validator của ``BBox`` (chặn hộp lật ngược) và của
    ``Confidence`` chạy **ngay tại đây**, tức bên trong ``try`` của worker. Một hộp hỏng
    làm mất một khung, không phải mất cả nhánh.

    Hàm thuần, tách khỏi lời gọi mạng: đây là chỗ có logic thật, và nó phải test được mà
    không cần Triton lẫn ``tritonclient``.
    """
    found: list[Detection] = []
    for i in range(len(out["labels"])):
        attrs: dict[str, float] = {}
        if role is CameraRole.TCODE:
            code = int(out["codes"][i])
            # -1 = không đọc được (hộp nằm ngoài ảnh). Bỏ khoá thay vì gửi một số âm:
            # `attrs` là ánh xạ tên->điểm, và một mục "headcode_-1" sẽ đi thẳng vào rule
            # như thể nó là một số xe thật.
            if code >= 0:
                attrs[f"headcode_{code:02d}"] = float(out["code_scores"][i])
        found.append(
            Detection(
                class_name=out["labels"][i].decode("utf-8"),
                confidence=float(out["scores"][i]),
                bbox=BBox.from_xyxy(tuple(float(v) for v in out["boxes"][i])),
                attrs=attrs,
            )
        )
    return found
