"""Chạy **nhánh model**: camera RTSP → gộp batch theo role → Triton → in kết quả.

Đây là bước kiểm chứng đầu-cuối của Phase 3a. Nó trả lời ba câu mà test đơn vị không trả
lời được, vì cả ba chỉ sai khi chạy thật:

1. ``pyds.get_nvds_buf_surface()`` có map được khung ra numpy đúng màu không.
2. Trục thời gian suy-từ-PTS có khớp giữa nhánh ghi và nhánh model không.
3. Ở nhịp thật, hàng đợi suy luận có bỏ khung nào không.

Chưa đẩy Kafka — phần đó là bước kế. Ở đây kết quả in ra màn hình để đối chiếu bằng mắt.

⚠️ **Phải cấp GPU bằng ``--gpus all``, không phải ``device=N``.** Pin GPU cấp CUDA compute
nhưng không cấp node V4L2 của NVDEC, và khi đó pipeline đứng ở ``PREROLLING`` không một
thông báo lỗi nào. Xem ``docs/DESIGN_NOTES.md`` DN-014.

Chạy::

    craneops-ds detect --role crane --duration 60
    craneops-ds detect --role tcode --duration 60 --triton host:19001
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.config import load_crane  # noqa: E402
from common.enum import CameraRole  # noqa: E402
from common.message import Detection, PerceptionMessage, perception_topic  # noqa: E402
from ds_app.src.pipeline.elements import SOURCE_QUEUE, apply_props, link_pads, make  # noqa: E402
from ds_app.src.pipeline.inference import (  # noqa: E402
    BLS_FOR_ROLE,
    FrameJob,
    InferenceClient,
)
from ds_app.src.pipeline.model import ModelBranch, roles_with_cameras  # noqa: E402
from ds_app.src.pipeline.sources import make_source_bin, source_bin_name  # noqa: E402
from ds_app.src.pipeline.timesync import TimeSync  # noqa: E402
from gateway.contract.bus import BusProducer  # noqa: E402


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", default="configs/cranes/GC03.yaml")
    ap.add_argument(
        "--role",
        default="all",
        help="crane | tcode | all — role nào có camera chạy model",
    )
    ap.add_argument("--triton", default="localhost:19001", help="gRPC của Triton")
    ap.add_argument(
        "--bus",
        default="",
        help="bootstrap Kafka; để trống thì KHÔNG publish (chỉ in ra màn hình)",
    )
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--quiet", action="store_true", help="chỉ in tổng kết, không in từng khung")
    return ap.parse_args()


def main() -> int:
    args = _args()
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    crane = load_crane(args.config)
    by_role = roles_with_cameras(crane)
    if args.role != "all":
        try:
            want = CameraRole(args.role)
        except ValueError:
            print(  # noqa: T201
                f"role không tồn tại: {args.role}; nhận {', '.join(r.value for r in CameraRole)}",
                file=sys.stderr,
            )
            return 2
        if want not in by_role:
            # Nói rõ VÌ SAO, vì có hai lý do rất khác nhau: role không có camera nào, hay
            # role chưa có model. Bản trước chỉ báo "không khớp" và người dùng phải đoán.
            reason = (
                "chưa có model BLS (Phase 3b)"
                if want not in BLS_FOR_ROLE
                else "không có camera nào decode"
            )
            print(f"role {want.value}: {reason}", file=sys.stderr)  # noqa: T201
            return 2
        by_role = {want: by_role[want]}
    if not by_role:
        print(f"không role nào chạy model khớp --role {args.role}", file=sys.stderr)  # noqa: T201
        return 2

    total_cams = sum(len(c) for c in by_role.values())
    print(  # noqa: T201
        f"cẩu {crane.crane_id} · {len(by_role)} role · {total_cams} camera · Triton {args.triton}\n"
        + "\n".join(
            f"  {role.value:<8}{cam.key:<14}{cam.code:<26}{cam.effective_fps:.1f} fps"
            for role, cams in by_role.items()
            for cam in cams
        ),
        flush=True,
    )

    Gst.init(None)
    pipeline = Gst.Pipeline.new("detect")
    sync = TimeSync()

    seen: dict[str, int] = {}
    lock = threading.Lock()

    def on_result(job: FrameJob, found: list[Detection]) -> None:
        with lock:
            seen[job.camera_code] = seen.get(job.camera_code, 0) + 1
            n = seen[job.camera_code]

        if bus is not None:
            # Dựng bằng pydantic nên một trường sai là lỗi TẠI ĐÂY, không phải rác nằm
            # trên topic tới lúc consumer nổ.
            message = PerceptionMessage(
                crane_id=crane.crane_id,
                camera_code=job.camera_code,
                role=job.role,
                frame_id=job.frame_id,
                start_ts=job.start_ts,
                fps=crane.source_fps,
                frame_ts=job.frame_ts,
                segment_hint=job.segment_hint,
                detections=found,
            )
            # Khoá = camera_code: cùng khoá ⇒ cùng phân vùng ⇒ giữ thứ tự. Message của
            # một camera tới consumer sai thứ tự sẽ làm hỏng mọi phép đếm chuỗi liên tiếp.
            bus.publish(message, key=job.camera_code, topic=perception_topic(job.role))
        if args.quiet:
            return
        what = ", ".join(
            # Kích thước hộp có mặt vì nó quyết định chất lượng crop đưa vào classifier:
            # crop huấn luyện có tỉ lệ rộng/cao ~1,0, nên một hộp dẹt bị ép về 224x224
            # vuông sẽ méo khác hẳn lúc huấn luyện.
            f"{d.class_name}@{d.confidence:.2f}"
            + f" {d.bbox.x2 - d.bbox.x1:.0f}x{d.bbox.y2 - d.bbox.y1:.0f}"
            + f"(ar {(d.bbox.x2 - d.bbox.x1) / max(1.0, d.bbox.y2 - d.bbox.y1):.2f})"
            + ("".join(f" [{k}={v:.2f}]" for k, v in d.attrs.items()))
            for d in found
        )
        print(  # noqa: T201
            f"[{job.camera_code}] #{job.frame_id:<6} t={job.frame_ts:.3f}  "
            f"{len(found)} vật  {what or '(không có)'}   (khung thứ {n})",
            flush=True,
        )

    bus = BusProducer(args.bus, client_id="ds_app") if args.bus else None
    if bus is not None:
        # Nạp sẵn metadata cho ĐÚNG các topic sắp dùng: không có bước này thì hai message
        # đầu mất trong lúc client hỏi cluster (xem BusProducer.start).
        bus.start(perception_topic(role) for role in by_role)

    client = InferenceClient(args.triton, on_result)
    client.start()

    branches = []
    index = 0
    for role, cams in by_role.items():
        branch = ModelBranch(role, cams, crane, submit=client.submit, time_sync=sync)
        muxer = branch.build(Gst, pipeline)

        for pad_index, cam in enumerate(cams):
            bin_ = make_source_bin(Gst, index, cam)
            pipeline.add(bin_)
            queue = make(Gst, "queue", f"q_{role.value}_{pad_index}")
            apply_props(queue, SOURCE_QUEUE)
            pipeline.add(queue)

            sink_pad = muxer.request_pad_simple(f"sink_{pad_index}")
            if sink_pad is None:
                raise RuntimeError(f"nvstreammux {role.value} không cấp được sink_{pad_index}")
            name = source_bin_name(index)
            link_pads(Gst, bin_.get_static_pad("src"), queue.get_static_pad("sink"), name, "queue")
            link_pads(Gst, queue.get_static_pad("src"), sink_pad, "queue", f"mux_{role.value}")
            branch.attach(pad_index, cam)
            index += 1
        branches.append(branch)

    errors: list[str] = []
    # `gst_bus`, KHÔNG phải `bus`: cái tên đó đã thuộc về producer Kafka ở trên, và gán đè
    # lên nó làm mọi `publish()` gọi lên GStreamer Bus. Đo được — nó không nổ ở đây mà ở
    # tận `flush()` lúc thoát, sau khi đã im lặng bỏ toàn bộ message.
    gst_bus = pipeline.get_bus()
    gst_bus.add_signal_watch()
    gst_bus.connect("message::error", lambda _b, m: errors.append(m.parse_error()[0].message))

    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(int(args.duration), lambda: (loop.quit(), False)[1])
    signal.signal(signal.SIGINT, lambda *_: loop.quit())

    started = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        client.stop()
        if bus is not None:
            # Trước khi đọc thống kê: phần còn trong bộ đệm chưa rời máy, và không flush
            # thì bảng dưới sẽ báo "đã gửi" cho những message chưa từng tới broker.
            bus.flush()
            bus.close()

    elapsed = time.time() - started
    stats = client.stats.snapshot()
    print(  # noqa: T201
        f"\n--- {elapsed:.1f}s ---\n"
        f"  gửi {stats['submitted']}  xong {stats['completed']}  "
        f"BỎ {stats['dropped']}  LỖI {stats['failed']}  LỖI-NHẬN {stats['sink_failed']}"
        + (f"\n  lỗi đầu tiên: {client.stats.last_error}" if client.stats.last_error else "")
    )
    if bus is not None:
        b = bus.stats.snapshot()
        print(  # noqa: T201
            f"  kafka: xếp {b['queued']}  broker ack {b['acked']}  "
            f"BỎ {b['dropped']}  LỖI {b['failed']}  còn bay {b['in_flight']}"
        )
    for cams in by_role.values():
        for cam in cams:
            got = seen.get(cam.code, 0)
            want = cam.effective_fps * elapsed
            print(  # noqa: T201
                f"  {cam.code:<28} {got:4} khung  "
                f"{got / elapsed:5.2f} fps (đặt {cam.effective_fps:.1f})  "
                f"{100 * got / want if want else 0:5.1f} %"
            )
    for message in errors[:5]:
        print(f"  ✗ {message}", file=sys.stderr)  # noqa: T201

    # Bỏ khung hoặc lỗi là **thất bại**, không phải cảnh báo: cả hai đều im lặng lúc chạy
    # thật, và đây là chỗ duy nhất chúng lộ ra.
    bus_ok = bus is None or bus.stats.clean
    return 0 if not errors and client.stats.clean and bus_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
