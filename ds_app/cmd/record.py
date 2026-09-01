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

    craneops-ds record --cam ccode_front_right --duration 60
    craneops-ds record --cam ccode_front_right --duration 60 --meta

``--cam`` nhận **khoá camera** trong ``configs/cranes/<cẩu>.yaml``, không phải số. Gõ sai
thì thông báo liệt kê mọi khoá đang có.
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
from ds_app.src.pipeline.timesync import TimeSync  # noqa: E402
from internal.pkg.fragments import FragmentIndex  # noqa: E402  # noqa: E402


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/cranes/GC03.yaml"))
    ap.add_argument(
        "--cam",
        required=True,
        help="khoá camera (vd: ccode1), hoặc `all` để ghi MỌI camera — hình dạng production",
    )
    ap.add_argument("--out", default="/var/lib/craneops/rec")
    ap.add_argument("--duration", type=int, default=60, help="giây; 0 = chạy mãi")
    ap.add_argument("--segment-sec", type=float, default=30.0, help="độ dài đoạn, giây")
    ap.add_argument(
        "--keep-segments",
        type=int,
        default=6,
        help="số đoạn giữ lại MỖI camera; 0 = không dọn. 6 đoạn 30 s = 3 phút",
    )
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
    if args.cam == "all":
        cameras = crane.record_cameras
    else:
        try:
            cameras = [crane.camera(args.cam)]
        except KeyError as exc:
            # KeyError của config đã liệt kê khoá hợp lệ; in trần thì dính thêm dấu nháy.
            print(str(exc).strip('"'), file=sys.stderr)  # noqa: T201
            raise SystemExit(2) from None

    if args.show_meta and len(cameras) > 1:
        raise SystemExit("--show-meta chỉ dùng với MỘT camera (nó dựng nvstreammux riêng)")

    print(  # noqa: T201
        f"cẩu {crane.crane_id} · {len(cameras)} camera · đoạn ~{args.segment_sec:g}s\n"
        + "\n".join(
            f"  {c.key:<16}{c.code:<26}{c.desc or '(không có)'}"
            + ("" if c.decodes else "   (chỉ ghi hình)")
            for c in cameras
        ),
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

    recorder = RecordingBranch(
        args.out,
        segment_sec=args.segment_sec,
        time_sync=sync,
        fragments=index,
        on_fragment=_on_fragment,
    )
    # Mỗi camera một nguồn + một nhánh ghi. KHÔNG dùng nvstreammux: ghi hình tách ở tầng
    # bitstream trước decode (DN-014), nên nhánh này không cần decode lẫn gộp batch — đó
    # chính là lý do 10 camera ghi cùng lúc vẫn 0 NVENC và 0 NVDEC.
    src_bins = []
    for i, cam in enumerate(cameras):
        b = make_source_bin(Gst, i, cam)
        pipeline.add(b)
        recorder.attach(Gst, b, cam.code)
        src_bins.append(b)

    if args.show_meta:
        # nvstreammux là thứ tạo ra NvDsBatchMeta; không có nó thì không có metadata nào.
        # Chế độ này CÓ decode — nó tồn tại để xem metadata, không phải để đo ghi hình.
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        mux = Gst.ElementFactory.make("nvstreammux", "mux")
        mux.set_property("batch-size", 1)
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("live-source", 1)
        pipeline.add(mux)
        src_bins[0].get_static_pad("src").link(mux.request_pad_simple("sink_0"))
        mux.link(sink)
        mux.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, _frame_meta_probe(cameras[0], sync, index), None
        )
    # KHÔNG nối gì vào src pad của nguồn. Đó là cách chế độ ghi hình đạt 0 % NVDEC:
    # nvurisrcbin có decoder bên trong, nhưng pad không nối thì luồng của nó dừng ngay ở
    # buffer đầu (NOT_LINKED) và decoder không chạy. Nhánh ghi tách TRƯỚC decode nên không
    # bị ảnh hưởng.
    #
    # ⚠️ Đo được, không phải suy đoán: nối một `fakesink` vào cho "gọn" làm NVDEC nhảy từ
    # 0 % lên 11,6 % với 10 camera trên RTX 5090 — tức decode cả 10 luồng để vứt đi. Trên
    # RTX 3060 một NVDEC thì đó là chí mạng. Xem HARDWARE_BUDGET §6.3.

    errors: list[str] = []
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::error", lambda _b, m: errors.append(m.parse_error()[0].message))
    # `splitmuxsink` báo đóng xong đoạn qua message trên bus, không qua signal — đó là tín
    # hiệu duy nhất đáng tin khi `async-finalize` bật.
    bus.connect("message::element", lambda _b, m: recorder.handle_bus_message(m))

    if args.keep_segments > 0:
        # Giữ theo SỐ ĐOẠN, không theo tuổi: độ dài đoạn dao động theo GOP, nên đếm tuổi
        # cho ra số đoạn khác nhau mỗi lúc. Sàn `min_age_sec` (mặc định = tầm với của
        # evidenced) vẫn thắng nếu đoạn ngắn bất thường.
        schedule(
            GLib,
            args.out,
            SweepPolicy(max_files_per_camera=args.keep_segments),
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

    elapsed = time.time() - started
    # Chỉ đếm `*.mp4`: đoạn còn đuôi `.part` là chưa chốt. Đoạn cuối luôn còn `.part` vì
    # pipeline dừng trước khi splitmuxsink kịp đóng nó — đó là bình thường, không phải lỗi.
    per_cam = {c.code: sorted(Path(args.out, c.code).glob("*.mp4")) for c in cameras}
    unfinished = {c.code: sorted(Path(args.out, c.code).glob("*.mp4.part")) for c in cameras}
    total = sum(f.stat().st_size for fs in per_cam.values() for f in fs)
    silent = [code for code, fs in per_cam.items() if not fs]

    print(  # noqa: T201
        f"\n=== {elapsed:.0f}s · {len(cameras)} camera ===\n"
        f"  đoạn mở      {len(opened)}\n"
        f"  còn trên đĩa {sum(len(f) for f in per_cam.values())}  ({total / 1e6:.1f} MB"
        f" ⇒ {total / 1e6 / max(elapsed, 1) * 3600 / 1000:.2f} GB/giờ cho cả cẩu)\n"
        f"  lỗi          {len(errors)}",
        flush=True,
    )
    for c in cameras:
        fs = per_cam[c.code]
        size = sum(f.stat().st_size for f in fs)
        # Độ dài đoạn THẬT phải học từ hai mốc mở liên tiếp, không lấy từ config:
        # splitmuxsink chỉ cắt tại keyframe nên nó là bội số của GOP.
        print(  # noqa: T201
            f"    {c.key:<16}{len(fs):>3} đoạn  {size / 1e6:>7.1f} MB"
            f"  {size / 1e6 / max(elapsed, 1) * 3600 / 1000:>5.2f} GB/giờ"
            f"  đoạn thật {index.observed_duration(c.code):.2f}s"
        )
    if silent:
        # Một camera không ra file nào giữa lúc các camera khác vẫn ghi là hỏng RIÊNG nó —
        # loại lỗi mà tổng dung lượng che mất.
        print(f"  ⚠️  {len(silent)} camera KHÔNG ghi được đoạn nào: {silent}")  # noqa: T201

    # Mất dữ liệu ở nhánh ghi KHÔNG tự lộ ra: file vẫn được tạo, vẫn mở được, chỉ thiếu
    # hình ở giữa. Báo ra đây và thoát khác 0 để nó không đi qua im lặng.
    n_part = sum(len(v) for v in unfinished.values())
    if n_part:
        print(f"  chưa chốt    {n_part} đoạn còn đuôi .part (đoạn cuối mỗi camera)")  # noqa: T201

    loss = recorder.loss
    if loss.clean:
        print("  ✅ không mất dữ liệu (0 lần hàng đợi đầy, 0 nghi mất khung I)")  # noqa: T201
    else:
        print("\n  ⚠️  MẤT DỮ LIỆU Ở NHÁNH GHI:")  # noqa: T201
        for line in loss.report():
            print(f"    {line}")  # noqa: T201
    return 1 if errors or silent or not loss.clean else 0


if __name__ == "__main__":
    raise SystemExit(main())
