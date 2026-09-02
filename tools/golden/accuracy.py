"""Độ chính xác **từng model** trên engine Triton, so với NHÃN thật.

So với **nhãn**, tức trả lời "model đọc có đúng không" — không phải "kết quả có giống một
bản cài đặt khác không". Hai câu hỏi đó khác nhau: hai bản cài đặt khớp nhau tuyệt đối vẫn
có thể cùng sai. Kết quả mới nhất: ``docs/HARDWARE_BUDGET.md`` §6.2.

Công cụ chỉ dựa vào ``internal/pkg/`` (kiểm chữ số ISO 6346, NMS của PicoDet) và HTTP của
Triton — không cần dựng cả hệ thống để đo một model.

Mỗi model dùng đúng tập dữ liệu nằm cạnh nó trong ``assets/``:

| model | tập | chỉ số |
|---|---|---|
| ``craneops_headcode_cls`` | ``cls-truckHead/samples`` (thư mục = lớp) | top-1 |
| ``craneops_ccode_rec_h`` | ``rec-containerNo/samples/h`` (tên file = chuỗi) | khớp chuỗi |
| ``craneops_ccode_rec_v`` | ``rec-containerNo/samples/v`` | khớp chuỗi |
| ``craneops_truckitems_pico`` | ``det-truckItems/samples`` (labelme) | recall/precision @IoU 0,5 |
| ``craneops_truckhead_pico`` | ``det-truckHead/samples`` (labelme) | recall/precision @IoU 0,5 |
| ``craneops_ccode_det_{h,v}`` | **không có tập** | — |

⚠️ Mọi model ở đây đã gấp tiền xử lý vào đồ thị (DN-012), nên đầu vào là **pixel BGR
thô**. Đưa RGB hay đưa dữ liệu đã chuẩn hoá đều cho kết quả rác mà không báo lỗi.

Chạy::

    make accuracy
    uv run --with "tritonclient[http]" --with opencv-python-headless \\
        python -m tools.golden.accuracy --url localhost:19200
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from internal.pkg.ccode import is_container_code
from internal.pkg.vision.ctc import CtcConfig, load_char_dict
from internal.pkg.vision.ctc import decode as ctc_decode
from internal.pkg.vision.nms import multiclass

ASSETS = Path(os.environ.get("CRANEOPS_ASSETS", "/ssd1/huylg/dnp_project/smartport/assets"))
"""Kho ảnh mẫu có nhãn. Cùng biến môi trường với ``tools/export_models.py``."""
CHAR_DICT = ASSETS / "char_dict" / "char_dict.txt"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def images_in(folder: Path) -> list[Path]:
    """Mọi ảnh trong cây thư mục, đã sắp xếp để phép đo lặp lại được."""
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


IOU_THRESHOLD = 0.5
"""Ngưỡng khớp hộp. 0,5 là quy ước của PASCAL VOC / COCO ``AP50``."""

CLASS_MAPPING = {
    "craneops_truckitems_pico": {0: "head", 1: "container"},  # localize_container_truck.py:46
    "craneops_truckhead_pico": {0: "head"},  # localize_head_truck.py:31
}


@dataclass
class Result:
    model: str
    dataset: str
    n: int = 0
    metric: str = ""
    value: str = ""
    detail: list[str] = field(default_factory=list)
    skipped: str = ""


_INPUT_NAME: dict[str, str] = {}


def input_name(client: Any, model: str) -> str:
    """Tên tensor đầu vào, **hỏi Triton** chứ không đoán theo tên model.

    Bản đầu đoán: ``"input" if model.endswith("_cls") else ("x" if "ccode" in model ...)``.
    Nó đúng cho đúng bảy model hiện có và sẽ sai lặng lẽ ở model thứ tám — hoặc khi đổi
    tên model. Hợp đồng nằm ở ``config.pbtxt``; hỏi thẳng nơi giữ nó.
    """
    if model not in _INPUT_NAME:
        _INPUT_NAME[model] = client.get_model_metadata(model)["inputs"][0]["name"]
    return _INPUT_NAME[model]


def infer(client: Any, model: str, data: Any, output: str, dtype: str) -> Any:
    import tritonclient.http as http

    inp = http.InferInput(input_name(client, model), data.shape, dtype)
    inp.set_data_from_numpy(np.ascontiguousarray(data))
    return client.infer(model, [inp], outputs=[http.InferRequestedOutput(output)]).as_numpy(output)


# ---------------------------------------------------------------- phân loại


def eval_headcode(client: Any) -> Result:
    root = ASSETS / "camera-truckNo" / "cls-truckHead"
    labels = [x for x in (root / "label.txt").read_text().split("\n") if x.strip()]
    files = images_in(root / "samples")
    truth = np.array([p.parent.name for p in files])

    batch = np.stack(
        [
            # BGR THÔ: chuẩn hoá ImageNet và phép đảo kênh đã nằm trong model (DN-012).
            cv2.resize(cv2.imread(str(p)), (224, 224), interpolation=cv2.INTER_LINEAR)
            .astype(np.float32)
            .transpose(2, 0, 1)
            for p in files
        ]
    )
    logits = np.concatenate(
        [
            infer(client, "craneops_headcode_cls", batch[i : i + 16], "head", "FP32")
            for i in range(0, len(batch), 16)
        ]
    )
    predicted = np.array([labels[i] for i in logits.argmax(-1)])
    wrong = [f"{files[i].parent.name}→{predicted[i]}" for i in np.flatnonzero(predicted != truth)]

    return Result(
        model="craneops_headcode_cls",
        dataset="cls-truckHead/samples",
        n=len(files),
        metric="top-1",
        value=f"{(predicted == truth).mean():.1%}",
        detail=wrong[:5],
    )


# ---------------------------------------------------------------- nhận dạng chữ


def eval_rec(client: Any, vertical: bool) -> Result:
    folder = (
        ASSETS / "camera-containerNo" / "rec-containerNo" / "samples" / ("v" if vertical else "h")
    )
    model = f"craneops_ccode_rec_{'v' if vertical else 'h'}"
    files = images_in(folder)
    if not files:
        return Result(model, "/".join(folder.parts[-3:]), skipped="không có ảnh")

    char_dict = load_char_dict(CHAR_DICT)
    hits, wrong, caught = 0, [], 0
    for path in files:
        # UINT8 NHWC, BGR thô — hợp đồng của bản *_folded.
        crop = cv2.resize(cv2.imread(str(path)), (256, 64), interpolation=cv2.INTER_LINEAR)
        logits = infer(
            client, model, crop[np.newaxis, ...], "save_infer_model/scale_0.tmp_0", "UINT8"
        )
        got = ctc_decode(
            np.asarray(logits[0], dtype=np.float32),
            char_dict,
            CtcConfig(character_threshold=0.3, score_threshold=0.0),
        )
        if got.text == path.stem:
            hits += 1
            continue
        # Đọc sai có bị tầng sau bắt không? Mã container hợp lệ phải khớp chữ số kiểm tra
        # ISO 6346. Một chuỗi sai mà VẪN hợp lệ mới là nguy hiểm — nó lọt xuống nghiệp vụ.
        valid = is_container_code(got.text)
        caught += not valid
        wrong.append(
            f"{path.stem} → {got.text!r}@{got.score:.2f}  "
            f"{'ISO 6346 BẮT ĐƯỢC' if not valid else '⚠️ ISO 6346 KHÔNG bắt được'}"
        )

    detail = wrong[:5]
    if wrong:
        detail.append(f"{caught}/{len(wrong)} lỗi bị ISO 6346 chặn trước khi tới nghiệp vụ")

    return Result(
        model=model,
        dataset="/".join(folder.parts[-3:]),
        n=len(files),
        metric="khớp chuỗi",
        value=f"{hits / len(files):.1%}",
        detail=detail,
    )


# ---------------------------------------------------------------- phát hiện vật


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def eval_pico(client: Any, model: str, folder: Path) -> Result:
    mapping = CLASS_MAPPING[model]
    files = [p for p in images_in(folder) if p.with_suffix(".json").exists()]
    matched: dict[str, int] = defaultdict(int)
    truth_count: dict[str, int] = defaultdict(int)
    pred_count: dict[str, int] = defaultdict(int)
    misses: list[str] = []

    for path in files:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        # BGR THÔ float32 NCHW: /255 và đảo kênh đã nằm trong model (DN-012).
        #
        # ⚠️ INTER_CUBIC, phép nội suy mà PicoDet được huấn luyện với (`PicoConfig` của v1,
        # và luật chung ở `vision/preprocess.py`). Bản trước để LINEAR ở đây: recall giống
        # hệt và precision chỉ lệch một hộp, nên bảng số trông vẫn ổn — nhưng **toạ độ hộp
        # lệch tới 17 px**, và CRANE01 gán lane bằng chính điểm mốc của hộp. Đo bằng con số
        # tổng thì không thấy; phải so từng hộp mới thấy.
        resized = cv2.resize(image, (416, 416), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        boxes = infer(client, model, tensor, "tmp_16", "FP32")[0]
        scores = infer(client, model, tensor, "concat_8.tmp_0", "FP32")[0]

        # Ngưỡng lấy đúng từ PicoConfig cũ; phần còn lại theo mặc định trong nms.py.
        dets = multiclass(boxes, scores, nms_threshold=0.3, score_threshold=0.3)
        scale = np.array([416 / width, 416 / height] * 2, dtype=np.float32)
        predictions: list[tuple[str, tuple[float, float, float, float]]] = []
        if len(dets) > 0:
            dets[:, :-2] /= scale
            dets[dets < 0] = 0
            for det in dets:
                name = mapping.get(int(det[-1]))
                if name:
                    predictions.append((name, (det[0], det[1], det[2], det[3])))

        shapes = json.loads(path.with_suffix(".json").read_text())["shapes"]
        truths = []
        for shape in shapes:
            # `head_2` là đầu kéo thứ hai trong khung — vẫn là lớp `head`.
            name = "head" if shape["label"].startswith("head") else shape["label"]
            (x1, y1), (x2, y2) = shape["points"]
            truths.append((name, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))

        for name, _ in truths:
            truth_count[name] += 1
        for name, _ in predictions:
            pred_count[name] += 1

        taken: set[int] = set()
        for name, box in truths:
            best, best_iou = -1, IOU_THRESHOLD
            for i, (pname, pbox) in enumerate(predictions):
                if i in taken or pname != name:
                    continue
                value = iou(box, pbox)
                if value >= best_iou:
                    best, best_iou = i, value
            if best >= 0:
                taken.add(best)
                matched[name] += 1
            elif len(misses) < 5:
                misses.append(f"{path.name}: sót {name}")

    total_truth = sum(truth_count.values())
    total_match = sum(matched.values())
    total_pred = sum(pred_count.values())
    per_class = "  ".join(
        f"{name} {matched[name]}/{truth_count[name]}" for name in sorted(truth_count)
    )
    recall = total_match / total_truth if total_truth else 0.0
    precision = total_match / total_pred if total_pred else 0.0

    return Result(
        model=model,
        dataset="/".join(folder.parts[-2:]),
        n=len(files),
        metric=f"recall/prec @IoU{IOU_THRESHOLD}",
        value=f"{recall:.1%} / {precision:.1%}",
        detail=[per_class, *misses[:3]],
    )


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="localhost:19200")
    args = ap.parse_args(argv)

    import tritonclient.http as http

    client = http.InferenceServerClient(url=args.url)

    results = [
        eval_headcode(client),
        eval_rec(client, vertical=False),
        eval_rec(client, vertical=True),
        eval_pico(
            client, "craneops_truckitems_pico", ASSETS / "camera-crane/det-truckItems/samples"
        ),
        eval_pico(
            client, "craneops_truckhead_pico", ASSETS / "camera-truckNo/det-truckHead/samples"
        ),
        Result(
            model="craneops_ccode_det_h",
            dataset="det-containerNo",
            skipped="KHÔNG CÓ TẬP DỮ LIỆU CÓ NHÃN — không đo được độ chính xác",
        ),
        Result(
            model="craneops_ccode_det_v",
            dataset="det-containerNo",
            skipped="KHÔNG CÓ TẬP DỮ LIỆU CÓ NHÃN — không đo được độ chính xác",
        ),
        Result(
            model="craneops_ccode_{h,v} (BLS)",
            dataset="samples/",
            skipped="chỉ 1 ảnh có nhãn mã container — không đủ để kết luận",
        ),
    ]

    print("\nĐỘ CHÍNH XÁC TỪNG MODEL — engine Triton, so với NHÃN\n")
    print(f"  {'model':<30}{'tập dữ liệu':<34}{'n':>5}  {'chỉ số':<22}kết quả")
    print(f"  {'-' * 104}")
    for r in results:
        if r.skipped:
            print(f"  {r.model:<30}{r.dataset:<34}{'—':>5}  ⚠️  {r.skipped}")
            continue
        print(f"  {r.model:<30}{r.dataset:<34}{r.n:>5}  {r.metric:<22}{r.value}")
        for line in r.detail:
            if line:
                print(f"  {'':<71}{line}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
