"""Kiểm môi trường `ds_app` — chạy cái này TRƯỚC khi nghi ngờ bất cứ thứ gì khác.

Nó phân biệt "image hỏng" với "cấu hình hỏng" với "code hỏng". Ba thứ đó có triệu chứng
giống hệt nhau — pipeline không ra dữ liệu — nhưng cách sửa hoàn toàn khác.

Mỗi mục ở đây tương ứng một cách hỏng đã gặp thật, và cả ba đều **im lặng**: không exception,
không log lỗi, chỉ là không có gì chảy qua pipeline.

    docker compose --env-file build/.env.ds -f build/docker-compose.ds.yml run --rm doctor
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _with_model(crane: Any) -> int:
    """Số camera thật sự có model để gọi. Xem ``BLS_FOR_ROLE``."""
    from ds_app.src.pipeline.model import roles_with_cameras

    return sum(len(c) for c in roles_with_cameras(crane).values())


def main() -> int:
    rows: list[tuple[bool, str, str]] = []

    # --- pyds ------------------------------------------------------------------
    try:
        import pyds

        rows.append((True, "pyds", pyds.__file__))
    except ImportError as exc:
        hint = ""
        if "libcuda" in str(exc):
            # Đúng lỗi gặp lúc build image: không có GPU thì libcuda.so.1 là stub rỗng.
            hint = " — chạy thiếu GPU? cần `--gpus all`"
        rows.append((False, "pyds", f"{exc}{hint}"))

    # --- nvstreammux cũ/mới ----------------------------------------------------
    mux_mode = os.environ.get("USE_NEW_NVSTREAMMUX")
    rows.append(
        (
            mux_mode == "no",
            "USE_NEW_NVSTREAMMUX",
            "no ⇒ mux CŨ (đúng)"
            if mux_mode == "no"
            else f"{mux_mode!r} ⇒ mux MỚI, nó BỎ QUA mọi thuộc tính mux cũ và "
            f"không bao giờ đẩy batch — không báo lỗi gì",
        )
    )

    # --- plugin GStreamer ------------------------------------------------------
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        for name, why in (
            ("nvurisrcbin", "nguồn RTSP kèm tee trước decode"),
            ("nvv4l2decoder", "decode phần cứng — thiếu quyền `video` thì treo PREROLLING"),
            ("nvstreammux", "gộp nhiều nguồn thành batch, và là thứ TẠO RA metadata"),
            # ⚠️ KHÔNG kiểm `nvinferserver`: ds_app không dùng nó. Nhánh model gọi Triton
            # thẳng qua gRPC từ `InferenceClient`, vì `nvinferserver` không dựng được
            # `NvDsObjectMeta` từ đầu ra PicoDet (`DetectionParams.nms` là "reserved, not
            # supported yet"). Kiểm một plugin không ai dùng sẽ khiến người debug đi tìm nó
            # trong pipeline.
            ("nvvideoconvert", "đổi sang RGBA — thiếu thì probe không map được khung ra numpy"),
            ("splitmuxsink", "ghi segment"),
        ):
            rows.append((Gst.ElementFactory.find(name) is not None, name, why))
    except Exception as exc:
        rows.append((False, "gi/Gst", str(exc)))

    # --- config + secret -------------------------------------------------------
    try:
        from common.config import load_crane

        crane = load_crane(REPO / "configs/cranes/GC03.yaml")
        by_role = ", ".join(f"{r.value} {len(g)}" for r, g in crane.cameras.items())
        rows.append(
            (
                True,
                "config + URL camera",
                f"{crane.crane_id}: {len(crane.record_cameras)} camera "
                # Hai con số khác nhau, và trộn chúng là nói dối: `model_cameras` là
                # camera KHAI chạy model, còn `roles_with_cameras` là camera có model
                # HÔM NAY. `ccode` khai 5 camera nhưng BLS của nó thuộc Phase 3b.
                f"({by_role}), {len(crane.model_cameras)} khai chạy model, "
                f"{_with_model(crane)} có model hôm nay",
            )
        )
        cameras = crane.record_cameras
    except Exception as exc:
        rows.append((False, "config + URL camera", str(exc).split("\n")[0]))
        cameras = []

    # --- ánh xạ khoá → camera thật ----------------------------------------------
    # `code` là chuỗi đi xuyên cả hệ: ds_app đặt lên PerceptionMessage, rule tra config
    # theo nó, evidence đặt tên thư mục segment theo nó. Bảng này là cách rẻ nhất để đối
    # chiếu mã với camera THẬT trước khi tin vào bất cứ kết quả nào.
    if cameras:
        print("\n=== camera ===")  # noqa: T201
        for cam in cameras:
            note = "" if cam.decodes else "  (chỉ ghi hình)"
            # In `stream` cạnh `code` vì mã suy từ nó: sai cổng là sai mã, và sai mã thì
            # rule không tìm thấy config của camera đó — im lặng.
            print(  # noqa: T201
                f"  {cam.key:<16}{cam.code:<26}{cam.stream:<40}{cam.desc}{note}"
            )

    # --- thư mục ghi ------------------------------------------------------------
    rec = Path("/rec")
    writable = False
    if rec.is_dir():
        probe = rec / ".doctor"
        try:
            probe.write_text("x")
            probe.unlink()
            writable = True
        except OSError as exc:
            rows.append((False, "/rec ghi được", str(exc)))
    if writable:
        rows.append((True, "/rec ghi được", str(rec)))
    elif not rec.is_dir():
        rows.append((False, "/rec ghi được", "không được mount"))

    width = max(len(name) for _, name, _ in rows)
    print("\n=== ds_app doctor ===")  # noqa: T201
    for ok, name, detail in rows:
        print(f"  {'✅' if ok else '❌'} {name:<{width}}  {detail}")  # noqa: T201

    failed = [name for ok, name, _ in rows if not ok]
    print()  # noqa: T201
    if failed:
        print(f"❌ {len(failed)} mục hỏng: {', '.join(failed)}")  # noqa: T201
        return 1
    print("✅ môi trường sẵn sàng")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
