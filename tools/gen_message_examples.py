"""Sinh ví dụ payload trong ``docs/MESSAGE_CONTRACT.md`` **từ** ``common/message.py``.

Vì sao cần: ví dụ JSON viết tay là phần tài liệu trôi khỏi code nhanh nhất. Nó không có
gì ràng buộc với model pydantic, nên đổi một trường là ví dụ sai — mà sai lặng lẽ, vì
không ai chạy ví dụ trong tài liệu. Sinh ra từ chính ``common/message.py`` thì không lệch
được, và ``make schema`` trong CI biến "quên regenerate" thành lỗi build.

Công cụ này chỉ thay phần giữa hai marker, phần diễn giải viết tay ở ngoài được giữ nguyên.

    make schema          # chạy cùng các bộ sinh khác
    python -m tools.gen_message_examples --check    # CI: fail nếu doc lỗi thời
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.enum import (
    CameraRole,
    ContainerDim,
    ContainerPosition,
    Direction,
    IxCd,
    Lane,
    SignalKind,
)
from common.message import (
    BBox,
    ContainerSlot,
    ControlAction,
    ControlMessage,
    Detection,
    EventMessage,
    EvidenceJob,
    EvidenceJobMessage,
    EvidenceKind,
    ManifestEntry,
    ManifestMessage,
    Message,
    OcrResult,
    PerceptionMessage,
    Signal,
    encode,
)

DOC = Path(__file__).resolve().parent.parent / "docs" / "MESSAGE_CONTRACT.md"
BEGIN = "<!-- BEGIN GENERATED EXAMPLES -->"
END = "<!-- END GENERATED EXAMPLES -->"

TS = 1_756_312_837.4
"""Mốc thời gian cố định để đầu ra ổn định giữa các lần chạy — điều kiện để --check dùng được."""


def samples() -> dict[str, Message]:
    """Một ví dụ đại diện cho mỗi topic, dùng số liệu thật của GC03."""
    return {
        "craneops.perception.*": PerceptionMessage(
            crane_id="GC03",
            camera_code="GC03_113_160_225_15_1508",
            role=CameraRole.CCODE,
            frame_id=300,
            start_ts=TS - 10.0,
            fps=30.0,
            frame_ts=TS,
            segment_hint="/var/lib/craneops/rec/GC03_113_160_225_15_1508/1756312830.mp4",
            detections=[
                Detection(
                    bbox=BBox(x1=505, y1=81, x2=1115, y2=662),
                    class_name="container",
                    confidence=0.93,
                )
            ],
            ocr=[
                OcrResult(
                    roi_index=0,
                    shape="vertical",
                    lane=Lane.ONE,
                    cont_dim=ContainerDim.FT40,
                    bbox=BBox(x1=612, y1=190, x2=704, y2=540),
                    text="MSKU",
                    confidence=0.97,
                )
            ],
        ),
        "craneops.signals": Signal(
            rule_code="CCODE01",
            crane_id="GC03",
            camera_code="GC03_113_160_225_15_1508",
            lane=Lane.ONE,
            direction=Direction.RIGHT,
            kind=SignalKind.CONTAINER_NO,
            frame_ts=TS,
            confidence=0.96,
            payload={"container_no": "MSKU1234567", "iso": "45G1", "streak": 4},
        ),
        "craneops.manifest": ManifestMessage(
            crane_id="GC03",
            berth_no="TS03",
            synced_at=TS,
            vsl_cd="VSL01",
            call_seq="001",
            call_year="2026",
            containers=[
                ManifestEntry(
                    container_no="MSKU1234567",
                    ix_cd=IxCd.IMPORT,
                    cont_dim=ContainerDim.FT40,
                    sztp="45G1",
                )
            ],
        ),
        "craneops.evidence.fast / craneops.evidence.slow": EvidenceJobMessage(
            event_id="GC03-1756312837-1",
            crane_id="GC03",
            lane=Lane.ONE,
            anchor_ts=TS,
            delay=20.0,
            jobs=[
                EvidenceJob(
                    kind=EvidenceKind.CLIP,
                    camera_code="GC03_113_160_225_15_1508",
                    window=(-20.0, 15.0),
                ),
                EvidenceJob(
                    kind=EvidenceKind.MOSAIC,
                    camera_code="GC03_113_160_225_15_1516",
                    window=(-35.0, 10.0),
                    grid=(2, 2),
                    count=3,
                ),
            ],
        ),
        "craneops.events": EventMessage(
            event_id="GC03-1756312837-1",
            crane_id="GC03",
            lane=Lane.ONE,
            direction=Direction.RIGHT,
            anchor_ts=TS,
            berth_no="TS03",
            vsl_cd="VSL01",
            call_seq="001",
            call_year="2026",
            truck_no="45",
            slots=[
                ContainerSlot(
                    container_no="MSKU1234567",
                    ix_cd=IxCd.IMPORT,
                    sztp="45G1",
                    cont_position=ContainerPosition.FT40,
                    container_image="https://eport.../snapshots/...jpg",
                    short_video="https://eport.../videoclips/...mp4",
                    confidence=0.96,
                )
            ],
        ),
        "craneops.control": ControlMessage(
            crane_id="GC03",
            action=ControlAction.RELOAD_RULE,
            rule_code="CCODE01",
            issued_at=TS,
        ),
    }


def render() -> str:
    parts = [BEGIN, "", "*Sinh tự động bởi `tools/gen_message_examples.py` — đừng sửa tay.*", ""]
    for name, msg in samples().items():
        payload = json.dumps(json.loads(encode(msg)), indent=2, ensure_ascii=False)
        parts += [f"### `{name}`", "", "```json", payload, "```", ""]
    parts.append(END)
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi (dùng cho CI)")
    args = ap.parse_args(argv)

    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"❌ {DOC} thiếu marker {BEGIN} … {END}", file=sys.stderr)
        return 2

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = head + render() + tail

    if args.check:
        if updated != text:
            print(
                f"❌ {DOC.name} đã lỗi thời so với common/message.py.\n"
                f"   Chạy: python -m tools.gen_message_examples",
                file=sys.stderr,
            )
            return 1
        print(f"✅ {DOC.name} khớp với common/message.py")
        return 0

    if updated == text:
        print(f"✅ {DOC.name} đã khớp, không đổi gì")
    else:
        DOC.write_text(updated, encoding="utf-8")
        print(f"✅ đã cập nhật {DOC.name} ({len(samples())} topic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
