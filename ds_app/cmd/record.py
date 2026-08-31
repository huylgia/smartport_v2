"""Chế độ **chỉ ghi hình** — chạy nhánh ghi passthrough, không suy luận.

Ba việc dùng tới nó:

1. **Kiểm nguồn tại chỗ.** Camera có kết nối được không, ghi ra file đọc được không, tốn
   bao nhiêu đĩa — trả lời được mà không cần Triton hay model.
2. **Xem thật sự có gì trong metadata.** ``--show-meta`` gắn probe đọc ``NvDsFrameMeta`` và
   in ra từng trường, kèm việc khôi phục chỉ số khung và quy đổi sang trục thời gian chung.
3. **Vai trò ``bottom`` / ``evidence_only``** vốn không bao giờ chạy model — với chúng đây
   là chế độ chạy thật, không phải chế độ thử.

⚠️ **Phải cấp GPU bằng ``--gpus all``, không phải ``device=N``.** Pin GPU cấp CUDA compute
nhưng không cấp node V4L2 của NVDEC, và khi đó pipeline đứng ở ``PREROLLING`` **không một
thông báo lỗi nào**. Xem ``docs/DESIGN_NOTES.md`` DN-014.

Chạy::

    make record CAM=1 DUR=60
    make record CAM=1 DUR=60 META=1
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.config import CameraConfig, load_crane  # noqa: E402
from ds_app.src.pipeline.recorder import RecordingBranch  # noqa: E402
from ds_app.src.pipeline.sources import make_source_bin  # noqa: E402
from ds_app.src.pipeline.sweeper import SweepPolicy, schedule  # noqa: E402
from ds_app.src.pipeline.timesync import FragmentIndex, TimeSync  # noqa: E402


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/cranes/GC03.yaml"))
    ap.add_argument("--cam", type=int, required=True, help="id camera trong config")
    ap.add_argument("--out", default="/var/lib/craneops/rec")
    ap.add_argument("--duration", type=int, default=60, help="giây; 0 = chạy mãi")
    ap.add_argument("--segment-sec", type=int, default=10)
    ap.add_argument("--retain-sec", type=int, default=1800, help="0 = không dọn")
    ap.add_argument(
        "--show-meta",
        action="store_true",
        help="in metadata khung của DeepStream (cần pyds)",
    )
    return ap.parse_args()


def _frame_meta_probe(camera: CameraConfig, sync: TimeSync, index: FragmentIndex) -> Any:
    """Probe in ``NvDsFrameMeta`` — để thấy CHÍNH XÁC những gì DeepStream trả về.

    Không có model nào chạy nên ``obj_meta_list`` sẽ rỗng: đây là metadata **khung**, chưa
    phải metadata **vật thể**. Nó cho thấy sẵn có những gì trước khi nhánh model tồn tại.
    """
    import pyds

    seen = [0]

    def _probe(_pad: Any, info: Any, _u: Any) -> Any:
        from gi.repository import Gst

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        node = batch.frame_meta_list
        while node is not None:
            frame = pyds.NvDsFrameMeta.cast(node.data)
            seen[0] += 1
            if seen[0] <= 3 or seen[0] % 50 == 0:
                pts_sec = frame.buf_pts / Gst.SECOND
                base = sync.anchor(camera.code, pts_sec, time.time())
                unix = base.to_unix(pts_sec) if base else float("nan")
                frag, confident = index.resolve(camera.code, unix)
                objects = 0
                obj = frame.obj_meta_list
                while obj is not None:
                    objects += 1
                    obj = obj.next

                print(  # noqa: T201
                    f"\n[meta] khung #{seen[0]}\n"
                    f"  pad_index      {frame.pad_index}   (chỉ số nguồn trong nvstreammux)\n"
                    f"  source_id      {frame.source_id}\n"
                    f"  frame_num      {frame.frame_num}   (không decimate ⇒ là chỉ số THẬT)\n"
                    f"  buf_pts        {frame.buf_pts}  = {pts_sec:.3f}s\n"
                    f"  → unix (trục chung) {unix:.3f}\n"
                    f"  khung          {frame.source_frame_width}x{frame.source_frame_height}\n"
                    f"  num_obj_meta   {objects}   "
                    f"{'(chưa có model ⇒ rỗng, đúng như mong đợi)' if objects == 0 else ''}\n"
                    f"  đoạn chứa nó   {Path(frag.path).name if frag else '—'}"
                    f"   {'(đã đóng)' if confident else '(còn đang ghi)'}",
                    flush=True,
                )
            node = node.next
        return Gst.PadProbeReturn.OK

    return _probe


def main() -> int:
    args = _args()
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    crane = load_crane(args.config)
    camera = crane.camera(args.cam)
    print(  # noqa: T201
        f"cẩu {crane.crane_id} · camera {camera.id} · vai trò {camera.role}\n"
        f"  mã       {camera.code}\n"
        f"  mô tả    {camera.name}\n"
        f"  decode   {'có' if camera.decodes else 'KHÔNG (chỉ ghi hình)'}\n"
        f"  ghi vào  {args.out}/{camera.code}/",
        flush=True,
    )

    Gst.init(None)
    pipeline = Gst.Pipeline.new("record")
    sync, index = TimeSync(), FragmentIndex()

    opened: list[tuple[str, float]] = []

    def _on_fragment(_cam: str, path: str, unix: float) -> None:
        opened.append((path, unix))
        gap = f"  (+{unix - opened[-2][1]:.2f}s)" if len(opened) > 1 else ""
        print(f"[đoạn] {Path(path).name}   mở lúc {unix:.3f}{gap}", flush=True)  # noqa: T201

    recorder = RecordingBranch(args.out, time_sync=sync, fragments=index, on_fragment=_on_fragment)
    src_bin = make_source_bin(Gst, 0, camera)
    pipeline.add(src_bin)
    recorder.attach(Gst, src_bin, camera.code)

    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", False)
    pipeline.add(sink)

    if args.show_meta:
        # nvstreammux là thứ tạo ra NvDsBatchMeta; không có nó thì không có metadata nào.
        mux = Gst.ElementFactory.make("nvstreammux", "mux")
        mux.set_property("batch-size", 1)
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("live-source", 1)
        pipeline.add(mux)
        src_bin.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
        mux.link(sink)
        mux.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, _frame_meta_probe(camera, sync, index), None
        )
    else:
        src_bin.get_static_pad("src").link(sink.get_static_pad("sink"))

    errors: list[str] = []
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::error", lambda _b, m: errors.append(m.parse_error()[0].message))

    if args.retain_sec > 0:
        schedule(
            GLib,
            args.out,
            SweepPolicy(max_age_sec=args.retain_sec, min_age_sec=min(300, args.retain_sec / 2)),
            every_sec=30,
        )

    loop = GLib.MainLoop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, lambda: (loop.quit(), False)[1])
    if args.duration > 0:
        GLib.timeout_add_seconds(args.duration, lambda: (loop.quit(), False)[1])

    pipeline.set_state(Gst.State.PLAYING)
    started = time.time()
    loop.run()
    pipeline.set_state(Gst.State.NULL)
    time.sleep(1)

    files = sorted(Path(args.out, camera.code).glob("*.mp4"))
    total = sum(f.stat().st_size for f in files)
    elapsed = time.time() - started
    print(  # noqa: T201
        f"\n=== {elapsed:.0f}s ===\n"
        f"  đoạn mở      {len(opened)}\n"
        f"  còn trên đĩa {len(files)}  ({total / 1e6:.1f} MB"
        f" ⇒ {total / 1e6 / max(elapsed, 1) * 3600 / 1000:.1f} GB/giờ)\n"
        f"  độ dài đoạn thật (học được)  {index.observed_duration(camera.code):.2f}s\n"
        f"  lỗi          {len(errors)}",
        flush=True,
    )
    for f in files[:10]:
        print(f"    {f.name}  {f.stat().st_size:>9,} byte")  # noqa: T201
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
