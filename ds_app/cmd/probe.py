"""Dò nhịp thật của mọi camera, và in ra dòng config sẵn để dán.

Chạy MỘT LẦN lúc lắp đặt một cẩu mới, hoặc khi nghi một camera đã đổi nhịp.

Vì sao cần: ``source_fps`` phải khai trong config và **không tự dò được lúc chạy** —
``nvv4l2decoder.drop-frame-interval`` chỉ đặt được ở NULL/READY (tức trước khi có khung
nào), còn caps của nguồn thì khai ``framerate=0/1`` trên cả 10 camera GC03. Chi tiết ở
``ds_app/src/pipeline/ratecheck.py``.

Cách đo: **ghi hình passthrough rồi đếm khung trong file**. Không decode gì cả, nên nó
đúng cho cả camera chỉ-ghi lẫn camera chạy model, và tốn 0 % NVDEC. Đoạn ghi chứa đúng
bitstream đã tới nên đây là phép đo chính xác nhất có được — chính nó đã bắt camera
``..._1517`` chạy 18 fps trong khi config khai 30.

⚠️ Kết quả **không tự ghi vào config**. Nó in ra để người ta dán vào, vì một camera tạm
thời rớt mạng sẽ đo ra nhịp thấp, và tự sửa config theo đó là chốt vĩnh viễn một con số
sai. Config là artifact review được, không phải trạng thái tự biến đổi.

Chạy::

    craneops-ds probe                # dò mọi camera, ~50 s
    craneops-ds probe --duration 130 # dò kỹ hơn: 3 đoạn mỗi camera
"""

from __future__ import annotations

import argparse
import signal
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.config import CameraConfig, load_crane  # noqa: E402
from ds_app.src.pipeline.recorder import RecordingBranch  # noqa: E402
from ds_app.src.pipeline.sources import make_source_bin  # noqa: E402
from ds_app.src.pipeline.timesync import TimeSync  # noqa: E402
from internal.pkg.fragments import FragmentIndex  # noqa: E402
from internal.pkg.mp4probe import read_mp4_info  # noqa: E402

TOLERANCE = 0.5
"""Chênh bao nhiêu fps thì coi là config khai sai. Khớp ``ratecheck.TOLERANCE``."""


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", default="configs/cranes/GC03.yaml")
    ap.add_argument("--out", default="/rec/_probe", help="thư mục tạm cho đoạn dò")
    ap.add_argument("--segment-sec", type=float, default=20.0)
    ap.add_argument(
        "--duration",
        type=float,
        default=50.0,
        help="giây; phải đủ để ĐÓNG ít nhất một đoạn (≈ 2x segment-sec)",
    )
    return ap.parse_args()


def main() -> int:
    args = _args()
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    crane = load_crane(args.config)
    cameras = crane.record_cameras
    print(  # noqa: T201
        f"dò {len(cameras)} camera của cẩu {crane.crane_id} trong {args.duration:g}s "
        f"(đoạn {args.segment_sec:g}s, ghi passthrough — 0 % NVDEC)\n",
        flush=True,
    )

    Gst.init(None)
    pipeline = Gst.Pipeline.new("probe")
    recorder = RecordingBranch(
        args.out,
        segment_sec=args.segment_sec,
        time_sync=TimeSync(),
        fragments=FragmentIndex(),
    )

    # KHÔNG nối src pad vào đâu cả: đó là cách giữ NVDEC ở 0 % (HARDWARE_BUDGET §6.3).
    for index, camera in enumerate(cameras):
        bin_ = make_source_bin(Gst, index, camera)
        pipeline.add(bin_)
        recorder.attach(Gst, bin_, camera.code)

    errors: list[str] = []
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::error", lambda _b, m: errors.append(m.parse_error()[0].message))
    bus.connect("message::element", lambda _b, m: recorder.handle_bus_message(m))

    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(int(args.duration), lambda: (loop.quit(), False)[1])
    signal.signal(signal.SIGINT, lambda *_: loop.quit())

    started = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    print(f"\nghi {time.time() - started:.0f}s xong, đang đếm khung…\n", flush=True)  # noqa: T201
    return _report(Path(args.out), cameras, errors)


def _measure(folder: Path) -> tuple[float | None, int]:
    """fps của một camera, từ các đoạn **đã đóng**. Trả ``(fps, số đoạn dùng được)``.

    Bỏ đoạn cuối: nó có thể vẫn đang ghi lúc pipeline dừng, và bảng mẫu chưa đầy đủ thì
    đếm ra nhịp thấp giả. Lấy trung vị để một đoạn bị cắt ngắn không kéo lệch kết quả.
    """
    segments = sorted(folder.glob("*.mp4"))[:-1]
    rates = []
    for seg in segments:
        info = read_mp4_info(seg)
        if info is not None and info.duration_sec > 1.0 and info.frames > 0:
            rates.append(info.fps)
    return (statistics.median(rates) if rates else None), len(rates)


def _report(out: Path, cameras: list[CameraConfig], errors: list[str]) -> int:
    rows: list[tuple[CameraConfig, float | None, int]] = []
    for camera in cameras:
        fps, n = _measure(out / camera.code)
        rows.append((camera, fps, n))

    print(f"  {'camera':<28}{'đo được':>10}{'khai':>8}{'đoạn':>7}")  # noqa: T201
    wrong: list[tuple[CameraConfig, float]] = []
    missing = 0
    for camera, fps, n in rows:
        if fps is None:
            print(f"  {camera.code:<28}{'—':>10}{camera.source_fps:>8g}{n:>7}  KHÔNG ĐO ĐƯỢC")  # noqa: T201
            missing += 1
            continue
        off = abs(fps - camera.source_fps) > TOLERANCE
        if off:
            wrong.append((camera, fps))
        print(  # noqa: T201
            f"  {camera.code:<28}{fps:>10.2f}{camera.source_fps:>8g}{n:>7}"
            + ("  ⚠️ KHAI SAI" if off else "")
        )

    if wrong:
        print("\nDán vào configs/cranes/<cẩu>.yaml — thêm `source_fps` vào dòng camera:")  # noqa: T201
        for camera, fps in wrong:
            # Làm tròn về số nguyên khi sát: nhịp camera là số nguyên trong mọi trường hợp
            # đã gặp, và một `source_fps: 17.98` trong config chỉ làm người đọc phân vân.
            value = round(fps) if abs(fps - round(fps)) < 0.2 else round(fps, 1)
            print(f"    {camera.key:<16} source_fps: {value:g}")  # noqa: T201
    elif not missing:
        print("\n✅ mọi camera khớp config")  # noqa: T201

    for message in errors[:5]:
        print(f"  ✗ {message}", file=sys.stderr)  # noqa: T201
    return 0 if not errors and not wrong and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
