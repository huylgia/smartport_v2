"""Xuất model sang Triton model repository.

    .t7 (mã hoá) ──giải mã──► .onnx ──vá batch động──► trtexec --fp16 ──► .plan
                                                            │
                                                            └──► config.pbtxt (sinh từ SPECS)

`config.pbtxt` được **sinh ra** từ bảng :data:`SPECS` chứ không viết tay, để nó không thể
trôi khỏi shape thật của model. Shape trong bảng là **số đo trực tiếp** từ file `.t7` bằng
``onnx.checker``, không phải suy đoán.

Module này **không dựng engine**. Việc đó thuộc về ``triton/modelsvc/main.py``, nơi duy
nhất được phép chạm vào model ở dạng rõ: nó kiểm giấy phép trước, ghi bản rõ vào tmpfs,
tôn trọng cờ :attr:`ModelSpec.fp16`, ghi dấu ``build.json`` và cài ``config.pbtxt``. Bản
đầu của file này từng có đường dựng engine riêng; nó đã chết, đã trôi khỏi ``SPECS``
(hardcode ``--fp16``) và ghi bản rõ ra **đĩa** thay vì tmpfs — nên đã bị xoá.

Cách dùng::

    python -m tools.export_models --check          # chỉ đối chiếu, không ghi
    python -m tools.export_models --emit-config    # sinh lại config.pbtxt

Model phân loại số đầu kéo **không** đi qua đường này — xem :data:`HEADCODE_CLS` và
``docs/DESIGN_NOTES.md`` DN-003.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRITON_REPO = REPO_ROOT / "triton" / "repo" / "craneops"

ASSETS = Path(os.environ.get("CRANEOPS_ASSETS", "/ssd1/huylg/dnp_project/smartport/assets"))
"""Thư mục chứa model .t7 đã mã hoá.

Cấu hình qua ``CRANEOPS_ASSETS`` vì đường dẫn khác nhau giữa host và container: compose
mount nó vào ``/assets``. Mặc định là đường dẫn trên máy dev để chạy tay cho tiện."""

# ---------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class Shape:
    name: str
    dims: tuple[int, ...]
    """Chiều **không kể batch**. Triton suy batch từ ``max_batch_size``."""

    dtype: str = "TYPE_FP32"


@dataclass(frozen=True)
class ModelSpec:
    """Một model trong repository.

    Attributes:
        name: Tên thư mục trong model repository, cũng là tên gọi qua gRPC.
        source: Đường dẫn `.t7` tương đối trong ``assets/`` cũ, hoặc ``None`` nếu
            model không đến từ đường giải mã (xem :data:`HEADCODE_CLS`).
        inputs / outputs: Shape **đã đo** từ ONNX thật.
        max_batch_size: ``0`` nghĩa là model không có chiều batch.
        needs_batch_patch: Model gốc chốt cứng ``batch=1``; phải vá ``dim[0]`` thành động
            trước khi build plan, nếu không Triton không bật được ``dynamic_batching``.
        dynamic_batching: Bật gom batch động. Chỉ bật cho model thực sự được gọi nhiều lần
            rời rạc trong một khung — tức là hai model ``rec``.
        dynamic_shape: Model có chiều H/W động; cần profile min/opt/max cho trtexec.
        fp16: Dựng engine ở FP16. ĐO trước khi bật — xem ``docs/DESIGN_NOTES.md`` DN-008.
    """

    name: str
    source: str | None
    inputs: tuple[Shape, ...]
    outputs: tuple[Shape, ...]
    max_batch_size: int
    needs_batch_patch: bool = False
    dynamic_batching: bool = False
    dynamic_shape: bool = False
    instance_count: int = 1
    fp16: bool = True
    """Dựng engine ở FP16. Tắt cho model mà FP16 làm đổi KẾT QUẢ, không chỉ đổi số lẻ."""
    max_conv_per_mparam: float | None = None
    """Trần số node Conv trên mỗi triệu tham số. Vượt ⇒ nhiều khả năng depthwise
    conv đã bị bung thành hàng nghìn conv một-kênh (lỗi hay gặp của tf2onnx)."""

    forbid_unfused_reparam: bool = False
    """Từ chối model họ RepVGG/MobileOne/FastViT còn ở dạng train-time nhiều nhánh."""

    note: str = ""
    trt_profile: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    """``{tên_input: (min, opt, max)}`` dạng chuỗi shape cho ``trtexec``."""


# Shape dưới đây đo trực tiếp từ file .t7 đã giải mã (onnx.checker pass cả 6).
SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="craneops_truckitems_pico",
        source="camera-crane/det-truckItems/truckItemsDetetion_111124_folded.t7",
        inputs=(Shape("image", (3, 416, 416)),),
        outputs=(Shape("tmp_16", (3598, 4)), Shape("concat_8.tmp_0", (2, 3598))),
        max_batch_size=4,
        needs_batch_patch=True,
        note=(
            "PGIE nhánh crane. 2 lớp: đầu kéo + container. "
            "Bản *_folded: nhận pixel BGR THÔ [0,255] — cả phép chia 255 lẫn phép đảo kênh RGB đã nằm trong đồ thị. DeepStream cấu hình net-scale-factor=1.0, model-color-format=1 (BGR), không offset. Xem DN-012."
        ),
        # Bắt buộc có: vá batch thành động mà không khai profile thì trtexec tự chốt về
        # batch=1 (chỉ cảnh báo, không lỗi) — coi như việc vá batch thành vô nghĩa.
        trt_profile={"image": ("1x3x416x416", "1x3x416x416", "4x3x416x416")},
    ),
    ModelSpec(
        name="craneops_truckhead_pico",
        source="camera-truckNo/det-truckHead/truckHeadDet_261024_folded.t7",
        inputs=(Shape("image", (3, 416, 416)),),
        outputs=(Shape("tmp_16", (3598, 4)), Shape("concat_8.tmp_0", (2, 3598))),
        max_batch_size=4,
        needs_batch_patch=True,
        note=(
            "PGIE nhánh tcode (camera 3, 5). "
            "Bản *_folded: nhận pixel BGR THÔ [0,255] — cả phép chia 255 lẫn phép đảo kênh RGB đã nằm trong đồ thị. DeepStream cấu hình net-scale-factor=1.0, model-color-format=1 (BGR), không offset. Xem DN-012."
        ),
        trt_profile={"image": ("1x3x416x416", "2x3x416x416", "4x3x416x416")},
    ),
    ModelSpec(
        name="craneops_ccode_det_h",
        source="camera-containerNo/det-containerNo/cont_h_1.0_op12_folded.t7",
        inputs=(Shape("x", (-1, -1, 3), dtype="TYPE_UINT8"),),
        outputs=(Shape("sigmoid_10.tmp_0", (1, -1, -1)),),
        max_batch_size=8,
        dynamic_shape=True,
        note=(
            "DB++ phát hiện vùng chữ, mã container NGANG. Vốn đã dynamic shape. "
            "FP32: FP16 làm bitmap xác suất lệch tới 0,198 trong khi ngưỡng quyết định là "
            "0,1-0,2 — đủ để lật một hộp. Xem DN-013. "
            "Bản *_folded: input là **UINT8 NHWC**, pixel BGR THÔ. Chuẩn hoá đã gấp vào đồ "
            "thị. Sinh bằng tools/fold_preprocess.py. Xem DN-011, DN-012."
        ),
        fp16=False,
        trt_profile={"x": ("1x352x480x3", "1x640x672x3", "8x800x992x3")},
    ),
    ModelSpec(
        name="craneops_ccode_det_v",
        source="camera-containerNo/det-containerNo/cont_v_3.3.1_op12_folded.t7",
        inputs=(Shape("x", (-1, -1, 3), dtype="TYPE_UINT8"),),
        outputs=(Shape("save_infer_model/scale_0.tmp_1", (1, -1, -1)),),
        max_batch_size=8,
        dynamic_shape=True,
        note=(
            "DB phát hiện vùng chữ, mã container DỌC. FP32 (xem DN-013). "
            "Bản *_folded: input là **UINT8 NHWC**, pixel BGR THÔ, chuẩn hoá đã gấp vào đồ thị."
        ),
        fp16=False,
        trt_profile={"x": ("1x352x480x3", "1x512x576x3", "8x800x992x3")},
    ),
    ModelSpec(
        name="craneops_ccode_rec_h",
        source="camera-containerNo/rec-containerNo/cont_h_3.3_op11_151124_folded.t7",
        inputs=(Shape("x", (64, 256, 3), dtype="TYPE_UINT8"),),
        outputs=(Shape("save_infer_model/scale_0.tmp_0", (25, 37)),),
        max_batch_size=32,
        needs_batch_patch=True,
        dynamic_batching=True,
        note=(
            "SVTR CTC, mã NGANG. 37 lớp = 1 blank (chỉ số 0) + 36 ký tự 0-9A-Z. "
            "dynamic_batching gom crop của cả 5 camera ccode — nguồn tăng throughput "
            "lớn nhất của dự án. "
            "Bản *_folded: input là **UINT8 NHWC**, pixel BGR THÔ, chuẩn hoá đã gấp vào đồ thị. "
            "FP32, KHÔNG phải FP16: đã đo thấy FP16 làm MẤT ký tự mà điểm tin cậy lại "
            "cao hơn — xem docs/DESIGN_NOTES.md DN-008."
        ),
        fp16=False,
        trt_profile={"x": ("1x64x256x3", "8x64x256x3", "32x64x256x3")},
    ),
    ModelSpec(
        name="craneops_ccode_rec_v",
        source="camera-containerNo/rec-containerNo/cont_v_3.3.1_opset11_051124_folded.t7",
        inputs=(Shape("x", (64, 256, 3), dtype="TYPE_UINT8"),),
        outputs=(Shape("save_infer_model/scale_0.tmp_0", (25, 37)),),
        max_batch_size=32,
        needs_batch_patch=True,
        dynamic_batching=True,
        note=(
            "SVTR CTC, mã DỌC. FP32 (xem DN-008). "
            "Bản *_folded: input là **UINT8 NHWC**, pixel BGR THÔ, chuẩn hoá đã gấp vào đồ thị."
        ),
        fp16=False,
        trt_profile={"x": ("1x64x256x3", "8x64x256x3", "32x64x256x3")},
    ),
)


@dataclass(frozen=True)
class BlsSpec:
    """Một model Python backend (BLS) điều phối det → hậu xử lý → rec → CTC.

    Sinh ra như model TensorRT chứ không viết tay: hai file ``config.pbtxt`` của BLS từng
    là thứ duy nhất viết tay trong repo, và đúng chúng là nơi cấu hình trôi mà không ai
    biết. Nay cả 9 model đều đi qua ``--emit-config`` và ``--check`` bắt được drift.
    """

    name: str
    vertical: bool
    det_model: str
    det_output: str
    rec_model: str
    rec_output: str
    instance_count: int = 3
    """Tiến trình Python **thật** (không phải luồng): hậu xử lý DB nặng CPU và bị GIL chặn.
    Đo được 230 req/s với count=1 so với 610 với count=3 — xem HARDWARE_BUDGET §6.1."""


BLS_SPECS: tuple[BlsSpec, ...] = (
    BlsSpec(
        name="craneops_ccode_h",
        vertical=False,
        det_model="craneops_ccode_det_h",
        det_output="sigmoid_10.tmp_0",
        rec_model="craneops_ccode_rec_h",
        rec_output="save_infer_model/scale_0.tmp_0",
    ),
    BlsSpec(
        name="craneops_ccode_v",
        vertical=True,
        det_model="craneops_ccode_det_v",
        det_output="save_infer_model/scale_0.tmp_1",
        rec_model="craneops_ccode_rec_v",
        rec_output="save_infer_model/scale_0.tmp_0",
    ),
)


HEADCODE_CLS = ModelSpec(
    name="craneops_headcode_cls",
    # ⚠️ KHÔNG phải truckHeadCls_150125.t7. File đó chưa reparameterize và có 3314 node
    # Conv (depthwise bị bung) — không đạt check_health. Bản này sinh bằng
    # tools/export_headcode_cls.py --nchw --fold-preprocess rồi mã hoá lại.
    # Xem docs/DESIGN_NOTES.md DN-003.
    source="camera-truckNo/cls-truckHead/truckHeadCls_reparam_folded.t7",
    # Input là RGB **THÔ [0,255]** — phép chuẩn hoá ImageNet đã gấp vào model, nên
    # DeepStream chỉ cần net-scale-factor=1.0 và không có offset. Xem DN-003.
    inputs=(Shape("input", (3, 224, 224)),),
    outputs=(Shape("head", (54,)),),
    max_batch_size=16,
    dynamic_batching=True,
    max_conv_per_mparam=40.0,
    forbid_unfused_reparam=True,
    note=(
        "FastViT-T8 phân loại số đầu kéo, 54 lớp, NCHW, batch động. "
        "Input BGR THÔ [0,255] — chuẩn hoá ImageNet VÀ phép đảo kênh đều đã gấp vào "
        "model (DN-003 + DN-012). DeepStream: net-scale-factor=1.0, model-color-format=1. "
        "Sinh bằng: tools/export_headcode_cls.py --nchw --fold-preprocess. "
        "KHÔNG dùng truckHeadCls_150125.t7 trong assets/ — chưa hợp nhất, 3314 node Conv. "
        "Xem docs/DESIGN_NOTES.md DN-003."
    ),
    trt_profile={"input": ("1x3x224x224", "4x3x224x224", "16x3x224x224")},
)

ALL_SPECS: tuple[ModelSpec, ...] = (*SPECS, HEADCODE_CLS)

# ---------------------------------------------------------------------------- config.pbtxt


def render_config(spec: ModelSpec) -> str:
    """Sinh nội dung ``config.pbtxt`` cho một model."""
    lines: list[str] = []
    if spec.note:
        for chunk in _wrap(spec.note, 88):
            lines.append(f"# {chunk}")
    lines += [
        "#",
        "# SINH TỰ ĐỘNG bởi tools/export_models.py — đừng sửa tay.",
        "",
        f'name: "{spec.name}"',
        'platform: "tensorrt_plan"',
        f"max_batch_size: {spec.max_batch_size}",
        "",
    ]

    for io_name, shapes in (("input", spec.inputs), ("output", spec.outputs)):
        for sh in shapes:
            dims = ", ".join(str(d) for d in sh.dims)
            lines += [
                f"{io_name} [",
                "  {",
                f'    name: "{sh.name}"',
                f"    data_type: {sh.dtype}",
                f"    dims: [ {dims} ]",
                "  }",
                "]",
                "",
            ]

    if spec.dynamic_batching:
        lines += [
            "# Gom nhiều lời gọi rời rạc thành một batch GPU.",
            "#",
            "# Vì sao 5 ms chứ không phải 0: mỗi camera gửi khoảng 5 khung/giây, tức một",
            "# khung mỗi 200 ms. Cửa sổ 0 (mặc định Triton tự suy khi thiếu config) chỉ gom",
            "# được request ĐÃ nằm sẵn trong hàng đợi, nên lời gọi từ các camera khác nhau",
            "# gần như không bao giờ được gộp. 5 ms là 2,5 % của một chu kỳ khung — không",
            "# đáng kể về độ trễ, nhưng đủ để các nguồn gọi chồng lấn nhau.",
            "#",
            # Nguồn gọi khác nhau ⇒ lợi ích đến từ chỗ khác nhau. Viết chung một đoạn cho
            # cả hai từng làm config của headcode_cls mang lời giải thích về ROI và BLS —
            # những thứ nó không hề dính tới.
            *(
                [
                    "# Phần lớn lợi ích thật ra đến từ chỗ khác: mỗi lời gọi đã mang sẵn",
                    "# TOÀN BỘ crop của một ROI (tới top_k=5 ảnh) thay vì từng ảnh một.",
                ]
                if spec.name.startswith("craneops_ccode_rec")
                else [
                    "# Nguồn gọi ở đây là nvinferserver của DeepStream, không phải BLS: nó",
                    "# đã gom sẵn các object của một khung, nên cửa sổ này chỉ còn việc gộp",
                    "# tiếp giữa các camera cùng vai trò.",
                ]
            ),
            "# Con số 5 ms nên được chỉnh lại sau khi đo thật — xem docs/HARDWARE_BUDGET.md.",
            "dynamic_batching {",
            "  preferred_batch_size: [ 4, 8, 16 ]",
            "  max_queue_delay_microseconds: 5000",
            "}",
            "",
        ]

    if spec.trt_profile:
        lines += _warmup_block(spec)

    lines += [
        "instance_group [",
        "  {",
        f"    count: {spec.instance_count}",
        "    kind: KIND_GPU",
        "  }",
        "]",
        "",
        "version_policy: { specific: { versions: [ 1 ] } }",
    ]
    return "\n".join(lines) + "\n"


def _warmup_block(spec: ModelSpec) -> list[str]:
    """``model_warmup``: chạy một cú giả ngay khi nạp model.

    Lần suy luận đầu sau khi nạp đắt hơn hẳn — TensorRT khởi tạo lười, cấp bộ nhớ và chọn
    kernel ở cú đầu tiên. Đo được: det 10,3 ms so với 2,0 ms ổn định (5,1x), rec 5,8x,
    pico 2,8x. Trong khung 200 ms thì không chết ai, nhưng nó làm mấy khung đầu sau mỗi
    lần khởi động lại chậm bất thường, và Triton chỉ báo READY sau khi warmup xong nên
    ``depends_on`` của compose cũng chính xác hơn.

    Dùng ``zero_data``: mục đích là ép engine khởi tạo, không phải tính ra kết quả đúng.
    Hình dạng lấy từ ``optShapes`` — chính điểm mà engine được tối ưu.
    """
    lines = [
        "# Chạy một cú giả lúc nạp: cú đầu tiên đắt gấp 3-6 lần (xem _warmup_block).",
        "model_warmup [",
        "  {",
        '    name: "khoi_dong"',
        "    batch_size: 1",
    ]
    for shape in spec.inputs:
        opt = spec.trt_profile.get(shape.name)
        dims = [int(d) for d in opt[1].split("x")][1:] if opt else list(shape.dims)
        lines += [
            "    inputs {",
            f'      key: "{shape.name}"',
            "      value {",
            f"        data_type: {shape.dtype}",
            f"        dims: [ {', '.join(str(d) for d in dims)} ]",
            "        zero_data: true",
            "      }",
            "    }",
        ]
    lines += ["  }", "]", ""]
    return lines


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


BLS_MODEL_PY = '''"""Điểm vào Python backend cho mã container {huong}.

SINH TỰ ĐỘNG bởi tools/export_models.py — đừng sửa tay.

Triton bắt buộc lớp phải tên ``TritonPythonModel`` và nằm trong ``model.py``. Toàn bộ nội
dung ở ``triton/bls/ccode.py``; file này chỉ đặt bí danh để hai model ngang/dọc dùng chung
một bản mã.
"""

import os
import sys

_APP_ROOT = os.environ.get("CRANEOPS_APP_ROOT", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from triton.bls.ccode import CCodeModel as TritonPythonModel  # noqa: E402, F401
'''


def render_bls_config(spec: BlsSpec) -> str:
    """Sinh ``config.pbtxt`` cho một model BLS."""
    huong = "DỌC" if spec.vertical else "NGANG"
    nan = "xoay 90°" if spec.vertical else "nắn phối cảnh"
    outputs = [
        ("texts", "TYPE_STRING", "-1"),
        ("scores", "TYPE_FP32", "-1"),
        ("boxes", "TYPE_INT32", "-1, 4"),
        ("quads", "TYPE_INT32", "-1, 4, 2"),
        ("det_scores", "TYPE_FP32", "-1"),
        ("sharpness", "TYPE_FP32", "-1"),
    ]
    lines = [
        f"# Mã container {huong}: det -> hậu xử lý -> {nan} -> cổng nét -> rec -> giải mã CTC.",
        "# Chạy Python backend theo kiểu BLS, KHÔNG phải ensemble: cổng nét loại bớt crop nên",
        "# số tensor ra khác số tensor vào, đồ thị tĩnh không diễn đạt được. Xem DN-007.",
        "#",
        "# SINH TỰ ĐỘNG bởi tools/export_models.py — đừng sửa tay.",
        "",
        f'name: "{spec.name}"',
        'backend: "python"',
        "",
        "# 0 = không gom batch ở tầng này. Ảnh ROI mỗi camera một kích thước nên không xếp",
        "# chồng được; việc gom batch xảy ra ở tầng dưới, trên các crop đã cùng 64x256.",
        "max_batch_size: 0",
        "",
        "input [",
        "  {",
        '    name: "image"',
        "    data_type: TYPE_UINT8",
        "    dims: [ -1, -1, 3 ]",
        "  },",
        "  {",
        '    name: "params"',
        "    data_type: TYPE_STRING",
        "    dims: [ 1 ]",
        "    optional: true",
        "  }",
        "]",
        "",
        "output [",
    ]
    for i, (name, dtype, dims) in enumerate(outputs):
        lines += ["  {", f'    name: "{name}"', f"    data_type: {dtype}", f"    dims: [ {dims} ]"]
        lines.append("  }," if i < len(outputs) - 1 else "  }")
    lines += ["]", ""]

    for key, value in (
        ("det_model", spec.det_model),
        ("det_output", spec.det_output),
        ("rec_model", spec.rec_model),
        ("rec_output", spec.rec_output),
        ("vertical", "true" if spec.vertical else "false"),
    ):
        lines += ["parameters {", f'  key: "{key}"', f'  value: {{ string_value: "{value}" }}', "}"]

    lines += [
        "",
        f"# {spec.instance_count} tiến trình Python thật (không phải luồng) — hậu xử lý DB nặng",
        "# CPU và bị GIL chặn. Đây là tham số quyết định throughput của cả nhánh: đo được",
        "# 230 req/s với count=1 so với 610 với count=3. Xem HARDWARE_BUDGET §6.1.",
        "instance_group [",
        "  {",
        f"    count: {spec.instance_count}",
        "    kind: KIND_CPU",
        "  }",
        "]",
        "",
        "version_policy: { specific: { versions: [ 1 ] } }",
    ]
    return "\n".join(lines) + "\n"


def emit_configs(*, check: bool) -> int:
    """Ghi (hoặc đối chiếu) ``config.pbtxt`` cho mọi model."""
    stale = []
    for spec in ALL_SPECS:
        dest = TRITON_REPO / spec.name / "config.pbtxt"
        want = render_config(spec)
        have = dest.read_text(encoding="utf-8") if dest.exists() else None
        if have == want:
            continue
        if check:
            stale.append(spec.name)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(want, encoding="utf-8")
            print(f"  ✅ {spec.name}/config.pbtxt")

    for bls in BLS_SPECS:
        for dest, want in (
            (TRITON_REPO / bls.name / "config.pbtxt", render_bls_config(bls)),
            (
                TRITON_REPO / bls.name / "1" / "model.py",
                BLS_MODEL_PY.format(huong="DỌC" if bls.vertical else "NGANG"),
            ),
        ):
            have = dest.read_text(encoding="utf-8") if dest.exists() else None
            if have == want:
                continue
            if check:
                stale.append(f"{bls.name}/{dest.name}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(want, encoding="utf-8")
                print(f"  ✅ {bls.name}/{dest.name}")

    if check and stale:
        print(
            f"❌ config.pbtxt lỗi thời: {', '.join(stale)}\n"
            f"   Chạy: python -m tools.export_models --emit-config",
            file=sys.stderr,
        )
        return 1
    print("✅ config.pbtxt khớp SPECS" if check else "✅ đã sinh config.pbtxt")
    return 0


# ---------------------------------------------------------------------------- onnx


def make_batch_dynamic(onnx_bytes: bytes) -> bytes:
    """Đặt ``dim[0]`` của mọi input/output thành động.

    Bốn model (2 pico, 2 rec) chốt cứng ``batch=1`` trong graph. Triton **không** bật được
    ``dynamic_batching`` nếu chiều batch không động — mà gom crop OCR của cả 5 camera ccode
    vào một batch chính là nguồn tăng throughput lớn nhất của dự án.

    Đã kiểm chứng 2026-08-29: cả 4 model vá được — ``shape_inference(strict_mode=True)`` và
    ``checker`` đều pass, và không node ``Reshape``/``Expand`` nào chốt cứng ``batch=1``.

    ⚠️ Graph hợp lệ **không** đồng nghĩa TensorRT dựng đúng engine và cho kết quả giống hệt.
    Bắt buộc parity test batch=1 vs batch=N ở Spike B trước khi tin dùng.
    """
    import onnx
    from onnx import shape_inference

    model = onnx.load_from_string(onnx_bytes)
    for vi in list(model.graph.input) + list(model.graph.output):
        d0 = vi.type.tensor_type.shape.dim[0]
        d0.ClearField("dim_value")
        d0.dim_param = "batch"
    patched = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(patched)
    return patched.SerializeToString()  # type: ignore[no-any-return]


def check_health(onnx_bytes: bytes, spec: ModelSpec) -> list[str]:
    """Bắt các lỗi export khiến model *chạy được nhưng chậm*.

    Cả hai lỗi dưới đây đều **không** làm sai kết quả, nên không có gì báo động — chúng chỉ
    làm inference chậm đi nhiều lần. Đúng loại lỗi lọt được vào production.

    1. **Khối tái tham số hoá chưa hợp nhất.** FastViT / MobileOne / RepVGG huấn luyện với
       nhiều nhánh song song rồi hợp nhất toán học thành một conv khi suy luận. Còn node
       ``Sub`` kèm tên chứa ``REPARAM`` nghĩa là chưa hợp nhất. TensorRT fuse được Conv+BN
       nhưng **không** làm được phép hợp nhất này — nó không biết hai nhánh tương đương.

    2. **Depthwise conv bị bung.** ``tf2onnx`` đôi khi tách một depthwise conv thành từng
       kênh riêng: hàng nghìn ``Conv`` một-kênh cộng ``Reshape`` để ghép lại, thay vì một
       grouped conv. Nhận biết bằng tỉ lệ Conv trên mỗi triệu tham số.

    Returns:
        Danh sách vấn đề; rỗng nghĩa là đạt.
    """
    import numpy as np
    import onnx

    model = onnx.load_from_string(onnx_bytes)
    issues: list[str] = []

    nodes = list(model.graph.node)
    convs = [n for n in nodes if n.op_type == "Conv"]
    n_params = sum(
        int(np.prod(onnx.numpy_helper.to_array(t).shape)) for t in model.graph.initializer
    )
    mparams = n_params / 1e6

    if spec.max_conv_per_mparam is not None and mparams > 0:
        ratio = len(convs) / mparams
        if ratio > spec.max_conv_per_mparam:
            grouped = sum(
                1 for n in convs if next((a.i for a in n.attribute if a.name == "group"), 1) > 1
            )
            issues.append(
                f"depthwise conv nhiều khả năng đã BỊ BUNG: {len(convs)} node Conv cho "
                f"{mparams:.2f} M tham số ({ratio:.0f}/M, trần {spec.max_conv_per_mparam:.0f}/M); "
                f"chỉ {grouped} node có thuộc tính group>1. Xuất lại với opset >= 13 "
                f"và --inputs-as-nchw."
            )

    if spec.forbid_unfused_reparam:
        n_sub = sum(1 for n in nodes if n.op_type == "Sub")
        has_reparam_names = any("REPARAM" in n.name for n in nodes)
        if n_sub and has_reparam_names:
            issues.append(
                f"khối tái tham số hoá CHƯA hợp nhất: {n_sub} node Sub và tên node còn "
                f"chứa 'REPARAM'. Gọi switch_to_deploy() trước khi xuất — xem "
                f"tools/export_headcode_cls.py."
            )

    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="đối chiếu config.pbtxt, không ghi")
    ap.add_argument("--emit-config", action="store_true", help="sinh lại config.pbtxt")
    args = ap.parse_args(argv)

    if args.check:
        return emit_configs(check=True)
    if args.emit_config:
        return emit_configs(check=False)

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
