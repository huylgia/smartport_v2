"""ds_app đầy đủ: ghi hình **và** suy luận, trên cùng một bộ nguồn.

Đây là chế độ chạy thật. ``record.py`` và ``detect.py`` mỗi cái chỉ dựng một nửa, và
chúng tồn tại để chẩn đoán từng nửa riêng — không phải để chạy production.

Ba thứ chỉ đúng khi hai nhánh chạy CÙNG NHAU, nên chỉ ở đây mới kiểm được:

1. **Một ``nvurisrcbin`` cho mỗi camera.** Nhánh ghi cắm vào ``tee_rtsp_pre_decode`` bên
   trong nó (trước decode), nhánh model lấy src pad đã decode. Hai kết nối RTSP cho cùng
   một camera là gấp đôi socket và gấp đôi tải mạng để lấy đúng một luồng.
2. **Một ``TimeSync`` dùng chung.** Đây là lý do module đó tồn tại: hai nhánh đóng dấu
   thời gian bằng hai đồng hồ khác nhau thì cửa sổ cắt clip lệch dần, và không có gì báo.
3. **``segment_hint``.** Chỉ nhánh ghi biết đoạn nào đang chứa một khoảnh khắc, chỉ nhánh
   model gửi message. Tách hai tiến trình thì trường này vĩnh viễn ``None`` và
   ``evidenced`` không biết cắt clip ở đâu.

⚠️ **Phải cấp GPU bằng ``--gpus all``, không phải ``device=N``.** Pin GPU cấp CUDA compute
nhưng không cấp node V4L2 của NVDEC, và khi đó pipeline đứng ở ``PREROLLING`` không một
thông báo lỗi nào. Xem ``docs/DESIGN_NOTES.md`` DN-014.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.config import load_crane  # noqa: E402
from common.message import Detection, PerceptionMessage, perception_topic  # noqa: E402
from ds_app.src.pipeline.inference import FrameJob, InferenceClient  # noqa: E402
from ds_app.src.pipeline.model import ModelBranch, roles_with_cameras  # noqa: E402
from ds_app.src.pipeline.ratecheck import RateCheck  # noqa: E402
from ds_app.src.pipeline.recorder import RecordingBranch  # noqa: E402
from ds_app.src.pipeline.sources import build_sources  # noqa: E402
from ds_app.src.pipeline.sweeper import SweepPolicy, schedule  # noqa: E402
from ds_app.src.pipeline.timesync import TimeSync  # noqa: E402
from gateway.contract.bus import BusProducer  # noqa: E402
from internal.pkg.fragments import FragmentIndex  # noqa: E402


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", default="configs/cranes/GC03.yaml")
    ap.add_argument("--out", default="/rec", help="thư mục ghi đoạn")
    ap.add_argument("--triton", default="triton:8001")
    ap.add_argument("--bus", default="redpanda:9092", help="để trống ⇒ không publish")
    ap.add_argument("--segment-sec", type=float, default=30.0)
    ap.add_argument("--keep-segments", type=int, default=6)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = chạy mãi")
    ap.add_argument("--stats-every", type=float, default=60.0, help="giây; 0 = tắt")
    return ap.parse_args()


def main() -> int:
    args = _args()
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    crane = load_crane(args.config)
    by_role = roles_with_cameras(crane)

    print(  # noqa: T201
        f"cẩu {crane.crane_id} · ghi {len(crane.record_cameras)} camera · "
        f"model {sum(len(c) for c in by_role.values())} camera / {len(by_role)} role\n"
        + "\n".join(
            f"  {c.key:<16}{c.code:<28}"
            + (f"{c.role.value} @ {c.effective_fps:.1f} fps" if c.decodes else "chỉ ghi hình")
            for c in crane.record_cameras
        ),
        flush=True,
    )

    Gst.init(None)
    pipeline = Gst.Pipeline.new("craneops")
    sync, fragments = TimeSync(), FragmentIndex()

    recorder = RecordingBranch(
        args.out, segment_sec=args.segment_sec, time_sync=sync, fragments=fragments
    )

    bus = BusProducer(args.bus, client_id="ds_app") if args.bus else None
    if bus is not None:
        bus.start(perception_topic(role) for role in by_role)

    seen: dict[str, int] = {}
    lock = threading.Lock()

    def on_result(job: FrameJob, found: list[Detection]) -> None:
        with lock:
            seen[job.camera_code] = seen.get(job.camera_code, 0) + 1
        if bus is None:
            return
        message = PerceptionMessage(
            crane_id=crane.crane_id,
            camera_code=job.camera_code,
            role=job.role,
            frame_id=job.frame_id,
            start_ts=job.start_ts,
            fps=job.source_fps,
            frame_ts=job.frame_ts,
            segment_hint=job.segment_hint,
            detections=found,
        )
        # Khoá = camera_code: cùng khoá ⇒ cùng phân vùng ⇒ giữ thứ tự.
        bus.publish(message, key=job.camera_code, topic=perception_topic(job.role))

    client = InferenceClient(args.triton, on_result)
    client.start()

    def segment_hint(camera_code: str, frame_ts: float) -> str | None:
        """Đoạn đang chứa khoảnh khắc này. ``None`` khi chưa mở đoạn nào.

        Không đòi đoạn đã đóng: khung vừa xử lý gần như luôn nằm trong đoạn ĐANG ghi, và
        ``mp4mux`` làm mới ``moov`` mỗi giây nên đoạn đó vẫn đọc được. Đòi đóng sẽ trả
        ``None`` cho gần như mọi message.
        """
        fragment, _confident = fragments.resolve(camera_code, frame_ts)
        return fragment.path if fragment else None

    rates = RateCheck()

    branches = {
        role: ModelBranch(
            role,
            cams,
            crane,
            submit=client.submit,
            time_sync=sync,
            rate_check=rates,
            segment_hint=segment_hint,
        )
        for role, cams in by_role.items()
    }
    muxers = {role: branch.build(Gst, pipeline) for role, branch in branches.items()}

    for role, pads in build_sources(Gst, pipeline, crane, muxers, recorder=recorder).items():
        for pad_index, camera in pads.items():
            branches[role].attach(pad_index, camera)

    errors: list[str] = []
    gst_bus = pipeline.get_bus()
    gst_bus.add_signal_watch()
    gst_bus.connect("message::error", lambda _b, m: errors.append(m.parse_error()[0].message))
    # `splitmuxsink` báo đóng xong đoạn qua message trên bus, không qua signal — đó là tín
    # hiệu duy nhất đáng tin khi `async-finalize` bật.
    gst_bus.connect("message::element", lambda _b, m: recorder.handle_bus_message(m))

    loop = GLib.MainLoop()
    if args.duration > 0:
        GLib.timeout_add_seconds(int(args.duration), lambda: (loop.quit(), False)[1])
    signal.signal(signal.SIGINT, lambda *_: loop.quit())

    if args.keep_segments > 0:
        schedule(GLib, Path(args.out), SweepPolicy(max_files_per_camera=args.keep_segments))

    started = time.time()
    if args.stats_every > 0:
        GLib.timeout_add_seconds(int(args.stats_every), lambda: _tick(started, client, bus, seen))

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        client.stop()
        if bus is not None:
            bus.flush()
            bus.close()

    return _report(time.time() - started, crane, client, bus, sync, recorder, seen, errors, rates)


def _tick(started: float, client: Any, bus: Any, seen: dict[str, int]) -> bool:
    """Nhịp tim: đủ để thấy hệ đang sống và có mất mát không, không nhiều hơn."""
    elapsed = time.time() - started
    s = client.stats.snapshot()
    line = (
        f"[{elapsed:6.0f}s] suy luận {s['completed']}"
        f"  bỏ {s['dropped']}  lỗi {s['failed'] + s['sink_failed']}"
    )
    if bus is not None:
        b = bus.stats.snapshot()
        line += f"  ·  kafka ack {b['acked']}  bỏ {b['dropped']}  lỗi {b['failed']}"
    print(line + f"  ·  {sum(seen.values())} khung", flush=True)  # noqa: T201
    return True


def _report(
    elapsed: float,
    crane: Any,
    client: Any,
    bus: Any,
    sync: TimeSync,
    recorder: RecordingBranch,
    seen: dict[str, int],
    errors: list[str],
    rates: RateCheck,
) -> int:
    s = client.stats.snapshot()
    print(  # noqa: T201
        f"\n--- {elapsed:.1f}s ---\n"
        f"  suy luận: gửi {s['submitted']}  xong {s['completed']}  "
        f"BỎ {s['dropped']}  LỖI {s['failed']}  LỖI-NHẬN {s['sink_failed']}"
        + (f"\n  lỗi đầu tiên: {client.stats.last_error}" if client.stats.last_error else "")
    )
    if bus is not None:
        b = bus.stats.snapshot()
        print(  # noqa: T201
            f"  kafka: xếp {b['queued']}  ack {b['acked']}  "
            f"BỎ {b['dropped']}  LỖI {b['failed']}  còn bay {b['in_flight']}"
        )
    if sync.resets:
        print(  # noqa: T201
            "  ⚠️ neo lại thời gian (PTS lùi): "
            + ", ".join(f"{c} x{n}" for c, n in sorted(sync.resets.items()))
        )
    print(f"  ghi hình: {recorder.loss.report()}")  # noqa: T201
    for code, r in rates.report():
        if r.measured is None:
            continue
        flag = "  ⚠️ KHAI SAI" if r.mismatched else ""
        print(f"  fps nguồn {code}: đo {r.measured:5.2f}  khai {r.declared:g}{flag}")  # noqa: T201

    from ds_app.src.pipeline.inference import BLS_FOR_ROLE

    for cam in crane.record_cameras:
        got = seen.get(cam.code, 0)
        if not cam.decodes:
            print(f"  {cam.code:<28} (chỉ ghi hình)")  # noqa: T201
            continue
        if cam.role not in BLS_FOR_ROLE:
            # Phân biệt "chưa có model" với "có model mà không ra khung nào". Không tách
            # thì bảng báo 0,0 % và trông y hệt một camera đang hỏng.
            print(f"  {cam.code:<28} (role {cam.role.value} chưa có model)")  # noqa: T201
            continue
        want = cam.effective_fps * elapsed
        print(  # noqa: T201
            f"  {cam.code:<28} {got:5} khung  {got / elapsed:5.2f} fps "
            f"(đặt {cam.effective_fps:.1f})  {100 * got / want if want else 0:5.1f} %"
        )
    for message in errors[:5]:
        print(f"  ✗ {message}", file=sys.stderr)  # noqa: T201

    bus_ok = bus is None or bus.stats.clean
    return 0 if not errors and client.stats.clean and bus_ok and recorder.loss.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
