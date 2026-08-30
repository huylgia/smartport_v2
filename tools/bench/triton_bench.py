"""Đo hiệu năng Triton: model thô, đường ống BLS, và mức gom batch thực tế.

Nửa còn lại của tiêu chí thoát Phase 2; nửa kia là độ chính xác
(``tools/golden/accuracy.py``). Ba câu hỏi cần trả lời bằng số:

1. **Có gánh nổi tải thật không?** Tải mục tiêu suy từ ``assets/config.yaml`` cũ:
   5 camera ccode, 20 ROI khai báo, mỗi lane hoạt động chạy khoảng một nửa số ROI song
   song, ở ~5 fps sau decimate ⇒ **~50 request/giây**, gấp đôi khi hai lane cùng chạy.

2. **``dynamic_batching`` có thật sự gom không?** Câu hỏi này KHÔNG trả lời được bằng
   throughput — một hệ thống gom batch tốt và một hệ thống chạy batch=1 nhưng nhanh có
   thể cho cùng con số. Phải đọc thẳng metrics của Triton:
   ``nv_inference_count / nv_inference_exec_count`` = số mẫu trung bình mỗi lần chạy GPU.
   Đây cũng là phép kiểm chứng cho ``docs/DESIGN_NOTES.md`` DN-009.

3. **Batch có đáng không?** Quét batch 1→32 trên recognizer để thấy đường cong thực tế,
   thay vì tin vào giả định "gom batch thì nhanh hơn".

Chạy::

    uv run --with "tritonclient[grpc]" python -m tools.bench.triton_bench --all
    uv run --with "tritonclient[grpc]" python -m tools.bench.triton_bench --pipeline --rps 50

**Máy dùng chung**: mặc định thời lượng ngắn và mức đồng thời thấp. Tăng có ý thức.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

from internal.pkg.nptypes import Image

ClientFactory: TypeAlias = "Callable[[], Any]"
"""Hàm tạo client Triton. Mỗi luồng đo tự tạo một client riêng — dùng chung một client
qua nhiều luồng sẽ biến hàng đợi của chính client thành nút thắt và ta sẽ đo nhầm nó
thay vì đo Triton."""

ASSETS = Path(os.environ.get("CRANEOPS_ASSETS", "/ssd1/huylg/dnp_project/smartport/assets"))
"""Kho model và ảnh mẫu. Cùng biến môi trường với ``tools/export_models.py`` — nếu chỉ một
trong hai đọc env thì đặt ``CRANEOPS_ASSETS`` sẽ cho một lần chạy nửa vời."""

# ROI thật của GC03 camera 1, lấy nguyên từ assets/config.yaml. Dùng dữ liệu thật vì số
# hộp phát hiện được quyết định kích thước batch gửi sang recognizer — ảnh ngẫu nhiên sẽ
# cho ra 0 hộp và biến phép đo thành vô nghĩa.
CAM01_ROIS = (
    "V0_1_0_505_81_1115_662_576_608_1.0_1.1_0.95",
    "H1_1_0_505_81_1115_662_640_672_1.1_1.7_0.95",
    "V0_1_1_0_161_560_683_512_576_1.0_1.1_0.95",
    "H1_1_1_0_138_535_720_704_640_1.1_1.7_0.95",
)

SAMPLE = ASSETS / "samples/QC3/Cam01/DRYU2874604-1731336343-01.jpg"


# ---------------------------------------------------------------- kết quả


@dataclass
class Latencies:
    """Gom độ trễ rồi tính phân vị. Trung bình che mất đuôi — đuôi mới là thứ làm nghẽn."""

    values: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, seconds: float) -> None:
        with self._lock:
            self.values.append(seconds * 1000)

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {}
        ordered = sorted(self.values)
        return {
            "p50": statistics.median(ordered),
            "p95": ordered[int(len(ordered) * 0.95)],
            "p99": ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)],
            "max": ordered[-1],
        }


def _fmt(name: str, count: int, seconds: float, lat: Latencies, extra: str = "") -> str:
    s = lat.summary()
    rps = count / seconds if seconds else 0.0
    return (
        f"  {name:<34} {rps:8.1f} req/s   "
        f"p50 {s.get('p50', 0):6.1f}  p95 {s.get('p95', 0):6.1f}  "
        f"p99 {s.get('p99', 0):6.1f} ms{extra}"
    )


# ---------------------------------------------------------------- metrics


def read_metrics(url: str) -> dict[str, dict[str, float]]:
    """Đọc ``/metrics`` của Triton, gom theo model.

    Ba con số cần: ``request_count`` (số lời gọi), ``inference_count`` (tổng số mẫu),
    ``exec_count`` (số lần thực thi trên GPU). Tỉ số ``inference/exec`` chính là kích
    thước batch trung bình — thứ duy nhất chứng minh được ``dynamic_batching`` có tác
    dụng hay không.
    """
    import urllib.request

    wanted = {
        "nv_inference_request_success": "requests",
        "nv_inference_count": "inferences",
        "nv_inference_exec_count": "executions",
    }
    out: dict[str, dict[str, float]] = {}
    with urllib.request.urlopen(url, timeout=5) as fh:  # noqa: S310
        for raw in fh.read().decode().splitlines():
            if raw.startswith("#") or "{" not in raw:
                continue
            metric = raw.split("{", 1)[0]
            key = wanted.get(metric)
            if key is None:
                continue
            labels, value = raw.split("{", 1)[1].split("}", 1)
            tags = dict(part.split("=", 1) for part in labels.split(",") if "=" in part)
            name = tags.get("model", "").strip('"')
            out.setdefault(name, {})[key] = float(value.strip())
    return out


Metrics: TypeAlias = "dict[str, dict[str, float]]"


def batch_report(before: Metrics, after: Metrics, models: list[str]) -> None:
    print("\n  mức gom batch thực tế (từ metrics của Triton)")
    print(f"  {'model':<30}{'lời gọi':>9}{'mẫu':>9}{'lần chạy GPU':>14}{'batch TB':>11}")
    for name in models:
        b, a = before.get(name, {}), after.get(name, {})
        req = a.get("requests", 0) - b.get("requests", 0)
        inf = a.get("inferences", 0) - b.get("inferences", 0)
        ex = a.get("executions", 0) - b.get("executions", 0)
        if not ex:
            continue
        print(f"  {name:<30}{req:>9.0f}{inf:>9.0f}{ex:>14.0f}{inf / ex:>11.2f}")


# ---------------------------------------------------------------- đo model thô

# Kích thước thay cho chiều động khi đo. Lấy đúng ``optShapes`` trong
# ``tools/export_models.py`` để đo ở điểm engine được tối ưu.
DYNAMIC_DIMS = {
    "craneops_ccode_det_h": (640, 672),
    "craneops_ccode_det_v": (512, 576),
}

_CONFIG_URL = "http://localhost:19200"
"""Gốc HTTP của Triton, dùng để đọc config model. ``main()`` ghi đè theo ``--http``."""


def describe_model(model: str) -> tuple[str, tuple[int, ...], str, str]:
    """``(tên input, dims, kiểu dữ liệu, tên output)`` — ĐỌC TỪ TRITON, không viết tay.

    Hợp đồng đầu vào của model đã đổi hai lần trong dự án này (gấp chuẩn hoá vào đồ thị,
    rồi chuyển sang UINT8 NHWC — DN-011, DN-012). Mỗi lần đổi, một bảng shape viết tay lại
    lỗi thời trong im lặng: phép đo hoặc nổ, hoặc tệ hơn là đo nhầm thứ. Đọc thẳng từ
    ``/v2/models/<tên>/config`` thì không thể lệch.
    """
    import json
    import urllib.request

    url = f"{_CONFIG_URL}/v2/models/{model}/config"
    with urllib.request.urlopen(url, timeout=5) as fh:  # noqa: S310
        config = json.load(fh)

    inp = config["input"][0]
    dims = tuple(
        d if d > 0 else DYNAMIC_DIMS.get(model, (640, 640))[i % 2]
        for i, d in enumerate(int(x) for x in inp["dims"])
    )
    return inp["name"], dims, inp["data_type"].removeprefix("TYPE_"), config["output"][0]["name"]


def bench_model(
    client_factory: ClientFactory,
    model: str,
    *,
    batches: tuple[int, ...],
    concurrency: int,
    duration: float,
) -> None:
    """Quét kích thước batch trên một model, dùng tensor giả.

    Tensor giả là ĐỦ ở đây: ta đo thông lượng số học, không đo chất lượng. Nội dung ảnh
    chỉ quan trọng khi đo đường ống (số hộp phát hiện được thay đổi theo ảnh).
    """
    import tritonclient.grpc as grpc

    input_name, shape, dtype, output_name = describe_model(model)
    print(f"\n{model}   [{dtype} {shape}]")
    rng = np.random.default_rng(0)
    for batch in batches:
        data = (
            rng.integers(0, 256, (batch, *shape), dtype=np.uint8)
            if dtype == "UINT8"
            else rng.random((batch, *shape), dtype=np.float32)
        )
        count, elapsed, lat = _drive(
            client_factory,
            concurrency,
            duration,
            _model_call(grpc, model, input_name, output_name, data, dtype),
        )
        print(
            _fmt(
                f"batch={batch:<3} luồng={concurrency}",
                count,
                elapsed,
                lat,
                f"   {count * batch / elapsed:8.0f} mẫu/s",
            )
        )


def _model_call(
    grpc: Any,
    model: str,
    input_name: str,
    output_name: str,
    data: Any,
    dtype: str,
) -> Callable[[Any], object]:
    """Đóng gói một lời gọi model. Trả về hàm để KHÔNG bắt biến vòng lặp qua closure."""

    def call(cli: Any) -> object:
        inp = grpc.InferInput(input_name, data.shape, dtype)
        inp.set_data_from_numpy(data)
        return cli.infer(model, [inp], outputs=[grpc.InferRequestedOutput(output_name)])

    return call


def _drive(
    client_factory: ClientFactory,
    concurrency: int,
    duration: float,
    call: Callable[[Any], object],
) -> tuple[int, float, Latencies]:
    """Chạy ``call`` liên tục trên N luồng trong ``duration`` giây, thu độ trễ.

    Tách riêng để hai phép đo (model thô và đường ống) dùng chung cách đếm — hai vòng đo
    khác nhau thì hai bộ số không so được với nhau.
    """
    lat = Latencies()
    stop = time.monotonic() + duration
    counter = _Counter()

    def worker() -> None:
        cli = client_factory()
        while time.monotonic() < stop:
            t0 = time.perf_counter()
            call(cli)
            lat.add(time.perf_counter() - t0)
            counter.bump()

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(worker)
    return counter.value, time.monotonic() - started, lat


class _Counter:
    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def bump(self) -> None:
        with self._lock:
            self.value += 1


# ---------------------------------------------------------------- đo đường ống


def load_rois() -> list[tuple[str, Image, bytes]]:
    """Chuẩn bị sẵn ROI đã cắt + tham số JSON, để vòng đo không tốn thời gian tiền xử lý."""
    # cv2.imread trả None khi không đọc được, dù stub khai là ndarray.
    image: Image | None = cv2.imread(str(SAMPLE))
    if image is None:
        raise FileNotFoundError(f"thiếu ảnh mẫu: {SAMPLE}")

    out = []
    for token in CAM01_ROIS:
        kind, rest = token[0], token[1:].split("_")
        _, _lane, _dim, x1, y1, x2, y2, in_h, in_w, ex_w, ex_h, thr = rest
        roi = np.ascontiguousarray(image[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1])
        params = json.dumps(
            {
                "det_size": [int(in_h), int(in_w)],
                "expand_ratio": [float(ex_w), float(ex_h)],
                "box_threshold": 0.2,
                "score_threshold": float(thr),
            }
        ).encode()
        model = "craneops_ccode_v" if kind == "V" else "craneops_ccode_h"
        out.append((model, roi, params))
    return out


def bench_pipeline(
    client_factory: ClientFactory,
    *,
    concurrency: int,
    duration: float,
    target_rps: float | None,
) -> tuple[int, float, Latencies]:
    """Đo đường ống BLS end-to-end bằng ROI thật.

    ``target_rps`` mô phỏng tải thật: gửi đúng nhịp đó thay vì đập hết sức. Đây mới là
    chế độ giống production — camera phát khung theo nhịp cố định, không phải càng nhanh
    càng tốt. Chạy không có ``target_rps`` để tìm trần.
    """
    import tritonclient.grpc as grpc

    rois = load_rois()
    lat = Latencies()
    count = 0
    errors = 0
    count_lock = threading.Lock()
    stop = time.monotonic() + duration
    interval = concurrency / target_rps if target_rps else 0.0

    def worker(slot: int) -> None:
        nonlocal count, errors
        cli = client_factory()
        i = slot
        while True:
            now = time.monotonic()
            if now >= stop:
                return
            model, roi, params = rois[i % len(rois)]
            i += 1
            ins = [
                grpc.InferInput("image", roi.shape, "UINT8"),
                grpc.InferInput("params", [1], "BYTES"),
            ]
            ins[0].set_data_from_numpy(roi)
            ins[1].set_data_from_numpy(np.array([params], dtype=object))
            t0 = time.perf_counter()
            try:
                cli.infer(model, ins, outputs=[grpc.InferRequestedOutput("texts")])
                lat.add(time.perf_counter() - t0)
                with count_lock:
                    count += 1
            except Exception:
                with count_lock:
                    errors += 1
            if interval:
                # Ngủ bù phần còn lại của nhịp; nếu đã trễ thì đi tiếp ngay.
                slack = interval - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for slot in range(concurrency):
            pool.submit(worker, slot)
    elapsed = time.monotonic() - started

    if errors:
        print(f"  ⚠️  {errors} lời gọi lỗi")
    return count, elapsed, lat


# ---------------------------------------------------------------- CLI


def _run(cmd: list[str]) -> str | None:
    import subprocess

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)  # noqa: S603
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def gpu_memory_mib(container: str | None) -> float:
    """VRAM của ĐÚNG tiến trình Triton này.

    Hai cái bẫy, cả hai đều dẫn tới kết luận sai về ngân sách VRAM:

    * ``--query-gpu=memory.used`` cho tổng của cả GPU. Máy này dùng chung nên phần lớn
      con số đó là của người khác.
    * Lọc theo tên tiến trình ``tritonserver`` cũng sai: trên máy này có **nhiều** Triton
      của các dự án khác nhau cùng chạy.

    Nên phải khoá theo **PID** của container, lấy qua ``docker inspect``.
    """
    if not container:
        return float("nan")
    pid = _run(["docker", "inspect", "-f", "{{.State.Pid}}", container])
    if not pid or not pid.isdigit():
        return float("nan")

    listing = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if listing is None:
        return float("nan")

    total = 0.0
    for line in listing.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == pid:
            total += float(parts[1])
    return total if total else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="localhost:19201", help="gRPC của Triton")
    ap.add_argument("--metrics", default="http://localhost:19202/metrics")
    ap.add_argument("--http", default="http://localhost:19200", help="HTTP của Triton (đọc config)")
    ap.add_argument("--models", action="store_true", help="quét batch trên model thô")
    ap.add_argument("--pipeline", action="store_true", help="đo đường ống BLS")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--duration", type=float, default=10.0, help="giây mỗi phép đo")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--vram-container",
        default="craneops-triton-1",
        help="tên container Triton — VRAM phải đo theo PID, xem gpu_memory_mib",
    )
    ap.add_argument(
        "--rps",
        type=float,
        default=None,
        help="phát theo nhịp này thay vì đập hết sức (mô phỏng tải thật)",
    )
    args = ap.parse_args(argv)

    if not (args.models or args.pipeline or args.all):
        args.all = True

    global _CONFIG_URL
    _CONFIG_URL = args.http

    import tritonclient.grpc as grpc

    def client_factory() -> object:
        return grpc.InferenceServerClient(url=args.url)

    print(f"Triton  : {args.url}")
    print(f"VRAM lúc rảnh: {gpu_memory_mib(args.vram_container):.0f} MiB\n")

    if args.models or args.all:
        print("=" * 78)
        print("MODEL THÔ — quét kích thước batch")
        print("=" * 78)
        for model, batches in (
            ("craneops_ccode_det_h", (1, 2)),
            ("craneops_ccode_det_v", (1, 2)),
            ("craneops_ccode_rec_h", (1, 4, 8, 16, 32)),
            ("craneops_ccode_rec_v", (1, 8, 32)),
            ("craneops_truckitems_pico", (1, 2, 4)),
            ("craneops_truckhead_pico", (1, 2, 4)),
            ("craneops_headcode_cls", (1, 4, 16)),
        ):
            bench_model(
                client_factory,
                model,
                batches=batches,
                concurrency=args.concurrency,
                duration=args.duration,
            )

    if args.pipeline or args.all:
        print()
        print("=" * 78)
        print("ĐƯỜNG ỐNG BLS — ROI thật, det + hậu xử lý + rec + CTC")
        print("=" * 78)
        tracked = [
            "craneops_ccode_h",
            "craneops_ccode_v",
            "craneops_ccode_det_h",
            "craneops_ccode_det_v",
            "craneops_ccode_rec_h",
            "craneops_ccode_rec_v",
        ]
        levels = [args.concurrency] if args.rps else [1, 4, 8, 16]
        for level in levels:
            before = read_metrics(args.metrics)
            count, elapsed, lat = bench_pipeline(
                client_factory,
                concurrency=level,
                duration=args.duration,
                target_rps=args.rps,
            )
            after = read_metrics(args.metrics)
            label = f"luồng={level}" + (f" nhịp={args.rps:.0f}/s" if args.rps else "")
            print(
                _fmt(
                    label,
                    count,
                    elapsed,
                    lat,
                    f"   VRAM {gpu_memory_mib(args.vram_container):.0f} MiB",
                )
            )
            if level == levels[-1]:
                batch_report(before, after, tracked)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
