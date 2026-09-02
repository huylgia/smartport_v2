"""Chuyển vùng OCR từ DSL chuỗi của v1 sang dòng YAML của v2.

v1 khai mỗi vùng bằng một chuỗi nối bằng gạch dưới::

    V0_1_0_505_81_1115_662_576_608_1.0_1.1_0.95
    │  │ │ └─────────┬───────┘ └──┬──┘ └──┬──┘ └┬─┘
    │  │ │       x1,y1,x2,y2   cao,rộng  nới   ngưỡng
    │  │ └─ kích thước: 0=40feet, 1=20feet
    │  └─── làn: 1|2|3
    └────── hình dạng: V0=dọc, H1=ngang

⚠️ **Toạ độ là pixel TUYỆT ĐỐI trong không gian ĐỘ PHÂN GIẢI KHAI, không phải độ phân giải
nguồn.** v1 co giãn mọi khung trước khi xử lý::

    videoscale ! video/x-raw,width={out_width},height={out_height}

Camera 1508 khai ``720p`` nhưng nguồn thật là 2688x1520 (đo 2026-09-02). Chia toạ độ cho
độ phân giải nguồn sẽ làm mọi vùng co về góc trên trái và trỏ vào chỗ không có mã container
nào — mà vẫn chạy, vẫn trả về chuỗi rỗng, không có gì báo.

v2 dùng toạ độ **tương đối [0..1]** nên nó không còn phụ thuộc độ phân giải nào cả — đó
chính là lý do đổi, và là thứ làm vùng sống sót khi camera đổi cấu hình.

``ocr_threshold`` ở cuối chuỗi **không** chuyển sang: đo trên v1 thấy cả 8 vùng dùng chung
0,95, nên nó là tham số của rule ``CCODE01`` chứ không phải thuộc tính của vùng.

Chạy::

    python -m tools.convert_ocr_rois --crane GC03
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

V1_CONFIG = Path("/ssd1/huylg/dnp_project/smartport/assets/config.yaml")

RESOLUTION = {
    "SD": (960, 540),
    "720p": (1280, 720),
    "960p": (1280, 960),
    "1080p": (1920, 1080),
    "3MP": (2048, 1536),
    "4MP": (2688, 1520),
}
"""Chép nguyên từ ``src/utils/__init__.py`` của v1. Không suy từ tên: ``4MP`` là
2688x1520, không phải 2000x2000 hay bất cứ thứ gì tên nó gợi ý."""

SHAPE = {"V0": "vertical", "H1": "horizontal"}
DIM = {"0": "40feet", "1": "20feet"}


@dataclass(frozen=True, slots=True)
class Roi:
    shape: str
    lane: str
    cont_dim: str
    roi: tuple[float, float, float, float]
    input_size: tuple[int, int]
    expand_ratio: tuple[float, float]
    threshold: float


def parse_roi(token: str, width: int, height: int) -> Roi:
    """Một chuỗi DSL → :class:`Roi` với toạ độ đã chuẩn hoá về [0..1]."""
    parts = token.split("_")
    if len(parts) != 12:
        raise ValueError(f"chuỗi vùng phải có 12 trường, nhận {len(parts)}: {token}")
    shape, lane, dim = parts[0], parts[1], parts[2]
    if shape not in SHAPE:
        raise ValueError(f"hình dạng không hợp lệ {shape!r}; nhận {sorted(SHAPE)}")
    if dim not in DIM:
        raise ValueError(f"kích thước không hợp lệ {dim!r}; nhận {sorted(DIM)}")

    x1, y1, x2, y2 = (int(p) for p in parts[3:7])
    # ⚠️ Chia cho độ phân giải KHAI của camera đó, không phải độ phân giải nguồn.
    return Roi(
        shape=SHAPE[shape],
        lane=lane,
        cont_dim=DIM[dim],
        roi=(x1 / width, y1 / height, x2 / width, y2 / height),
        # v1 xếp thứ tự (cao, rộng) — giữ nguyên, `OcrRoi.input_size` cũng vậy.
        input_size=(int(parts[7]), int(parts[8])),
        expand_ratio=(float(parts[9]), float(parts[10])),
        threshold=float(parts[11]),
    )


def _rows(crane_id: str) -> list[tuple[int, str, str, list[Roi]]]:
    import yaml

    data = yaml.safe_load(V1_CONFIG.read_text(encoding="utf-8"))
    if data.get("crane_id") != crane_id:
        raise SystemExit(f"config v1 là của cẩu {data.get('crane_id')!r}, không phải {crane_id!r}")

    out = []
    for entry in data["camera_config"]:
        parts = str(entry).split("|")
        if len(parts) < 9 or not parts[8].strip():
            continue
        cam_id, name, resolution = int(parts[0]), parts[1], parts[2]
        if resolution not in RESOLUTION:
            raise SystemExit(f"camera {cam_id}: độ phân giải lạ {resolution!r}")
        width, height = RESOLUTION[resolution]
        rois = [parse_roi(tok, width, height) for tok in parts[8].split()]
        out.append((cam_id, name, resolution, rois))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--crane", default="GC03")
    args = ap.parse_args(argv)

    rows = _rows(args.crane)
    thresholds = {r.threshold for _, _, _, rois in rows for r in rois}
    total = sum(len(rois) for _, _, _, rois in rows)

    print(f"# {args.crane}: {len(rows)} camera ccode, {total} vùng OCR")
    print(
        f"# ocr_threshold KHÔNG chuyển sang — cả {total} vùng dùng chung "
        f"{sorted(thresholds)}, nên nó là tham số của rule CCODE01."
    )
    for cam_id, name, resolution, rois in rows:
        print(f"\n# camera {cam_id} ({name}) — v1 khai {resolution}")
        print("      ocr_rois:")
        for r in rois:
            roi = ", ".join(f"{v:.4f}" for v in r.roi)
            print(
                f"        - {{shape: {r.shape}, lane: '{r.lane}', cont_dim: {r.cont_dim}, "
                f"roi: [{roi}], input_size: [{r.input_size[0]}, {r.input_size[1]}], "
                f"expand_ratio: [{r.expand_ratio[0]}, {r.expand_ratio[1]}]}}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
