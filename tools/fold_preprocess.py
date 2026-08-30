"""Gấp tiền xử lý vào chính đồ thị ONNX, để mọi model nhận **pixel BGR thô ``[0,255]``**.

Mục tiêu là một quy tắc duy nhất cho cả bảy model, thay vì mỗi model một kiểu:

    Model nào cũng nhận pixel BGR thô. Mọi phép đổi thang, đổi thứ tự kênh đều nằm
    TRONG model, chạy trên GPU.

Trước khi có công cụ này, mỗi model một luật riêng và luật đó chỉ tồn tại trong đầu người
viết code gọi nó:

| Model | Thang | Thứ tự kênh | mean/std |
|---|---|---|---|
| ccode det_h | ``/255`` | BGR | riêng, std=1 |
| ccode det_v | ``/255`` | BGR | kiểu ImageNet |
| ccode rec | ``/255`` | BGR | 0,5 / 0,5 |
| pico (2 model) | ``/255`` | **RGB** | không có |
| headcode_cls | thô | **RGB** | đã gấp sẵn (DN-003) |

Cho sai một ô trong bảng đó thì model vẫn chạy, vẫn trả kết quả, chỉ là kết quả rác. Đưa
hết vào đồ thị là xoá luôn cả lớp lỗi này.

Các phép được chèn vào đầu đồ thị, theo thứ tự:

    input ─► [Cast float32] ─► [Transpose NHWC→NCHW] ─► [Gather đảo kênh] ─► [Mul A] ─► [Sub B] ─► …

Tất cả đều là phép TensorRT hợp nhất được vào conv đầu tiên lúc dựng engine, nên chi phí
lúc chạy về gần 0. Đo thật: xem ``docs/DESIGN_NOTES.md`` DN-011.

**Không bao giờ ghi đè ``.t7`` gốc.** Luôn sinh file mới với hậu tố ``_folded``.

Chạy::

    python -m tools.fold_preprocess              # sinh + kiểm chứng số học
    python -m tools.fold_preprocess --check      # chỉ kiểm chứng file đã sinh
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from internal.pkg.nptypes import Array
from internal.pkg.security.cipher import decrypt_file, encrypt_bytes
from internal.pkg.vision.preprocess import (
    DET_NORM_HORIZONTAL,
    DET_NORM_VERTICAL,
    REC_NORM,
    Normalization,
)
from tools.export_models import ASSETS

RTOL, ATOL = 1e-3, 1e-3
"""Tiêu chí khớp: ``|cũ - mới| <= ATOL + RTOL·|cũ|`` từng phần tử — ngữ nghĩa
``np.allclose``.

Con số ``1e-3`` chọn theo **độ phân giải quyết định của nghiệp vụ**, không phải chọn cho
vừa số đo. Ở mọi đầu ra, nó nhỏ hơn ngưỡng mà tầng sau dùng để quyết định ít nhất 100 lần:

| Đầu ra | Ngưỡng quyết định của tầng sau | ``1e-3`` nhỏ hơn |
|---|---|---|
| bitmap DB ``[0,1]`` | ``bitmap_threshold=0.1``, ``box_threshold=0.2`` | 100-200x |
| logit SVTR | argmax — chỉ lật khi hai lớp cách nhau < 1e-3 | model đã lưỡng lự sẵn |
| toạ độ bbox PicoDet (0-640 px) | IoU sau NMS | 1e-3 tương đối = 0,6 px |
| điểm PicoDet | ``0.33``-``0.5`` | 330-500x |

Lấy chặt hơn (``1e-4``) thì rơi đúng vào sàn nhiễu float32: đo được 6/7 model nằm trong
khoảng 0,002-1,48 lần ngưỡng đó, tức phép thử không còn phân biệt được lỗi với nhiễu.

Phải có **cả hai vế** ``ATOL`` và ``RTOL``, và đây là chỗ đã sai hai lần trước khi đúng:

* **Chỉ tuyệt đối** thì hỏng với đầu ra thang lớn: PicoDet trả toạ độ bbox tới 623, lệch
  9,2e-04 bị từ chối trong khi đó là 1,5e-06 tương đối — thuần nhiễu float32.
* **Chỉ tương đối** thì hỏng với đầu ra gần 0: điểm số của ``truckhead_pico`` trên ảnh
  không có xe có giá trị lớn nhất 0,054, nên lệch tuyệt đối 5,1e-05 (vô nghĩa về mặt vật
  lý) hoá thành 9,4e-04 tương đối và cũng bị từ chối.

Một model có nhiều đầu ra ở các thang khác nhau thì không có con số đơn lẻ nào đúng cho
tất cả."""

CONTROL_MARGIN = 100.0
"""Phép thử phải có SỨC PHÂN BIỆT: đưa cố ý sai thứ tự kênh vào đồ thị mới thì sai lệch
phải lớn hơn sai lệch hợp lệ ít nhất chừng này lần.

Không có phép thử này thì một đồ thị bỏ qua hẳn đầu vào cũng "khớp" hoàn hảo. Đo thật:
đưa sai kênh cho PicoDet lệch 5,1e+01 so với 9,2e-04 hợp lệ — cách 5 bậc độ lớn."""

IDENTITY = Normalization(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
"""``mean=0, std=1`` ⇒ chỉ còn phép chia 255. Dùng cho PicoDet, vốn không có mean/std."""

NO_SCALE = Normalization(mean=(0.0, 0.0, 0.0), std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0))
"""``A = 1/(255·std) = 1`` ⇒ không đổi thang gì cả. Dùng cho model đã gấp chuẩn hoá từ
trước (``headcode_cls``, DN-003), nay chỉ cần thêm phép đảo kênh."""

PORT_SCENE = "samples/QC3/Cam01/DRYU2874604-1731336343-01.jpg"
"""Ảnh mặc định để kiểm chứng. **Ảnh thật, không phải nhiễu ngẫu nhiên.**

Nhiễu ngẫu nhiên là đầu vào bệnh lý cho detector: nó không thấy vật gì nên mọi điểm số về
gần 0, và khi đó một sai lệch tuyệt đối vô nghĩa cũng thành sai lệch tương đối lớn. Ảnh
thật còn cho đúng phân bố dữ liệu mà model gặp lúc chạy."""


@dataclass(frozen=True)
class FoldTarget:
    """Một model cần gấp tiền xử lý."""

    label: str
    source: str
    """Đường dẫn ``.t7`` **gốc**, tương đối trong ``assets/``."""

    norm: Normalization
    input_shape: tuple[int, ...]
    """``(C, H, W)`` dùng để kiểm chứng số học. KHÔNG kể batch."""

    sample: str = PORT_SCENE
    """Ảnh thật dùng để kiểm chứng, tương đối trong ``assets/``. Nên là ảnh của chính
    camera mà model này chạy trên đó."""

    swap_rb: bool = False
    """Model được huấn luyện trên RGB, nhưng ta muốn nó nhận BGR.

    Chèn ``Gather(indices=[2,1,0], axis=1)`` — TensorRT hợp nhất phép này vào việc sắp
    lại kênh của trọng số conv đầu tiên, nên không tốn gì lúc chạy."""

    uint8_nhwc: bool = False
    """Đổi đầu vào thành ``UINT8`` bố cục ``NHWC``.

    ⚠️ CHỈ dùng cho model mà **Python** gọi trực tiếp (đường BLS của nhánh ccode). Ở đó
    nó bỏ hẳn ``astype(float32)`` và phép sao chép có bước nhảy sang NCHW — hai thứ chiếm
    phần lớn thời gian tiền xử lý còn lại.

    KHÔNG dùng cho model mà DeepStream gọi: ``nvinfer``/``nvinferserver`` tự tiền xử lý
    trên GPU và đưa ra tensor **float**. Đổi sang UINT8 sẽ trói tay cấu hình DeepStream ở
    Phase 3 để đổi lấy một khoản tiết kiệm không tồn tại (đường đó không có Python)."""

    @property
    def dest(self) -> str:
        stem = Path(self.source)
        return str(stem.with_name(f"{stem.stem}_folded{stem.suffix}"))


TARGETS: tuple[FoldTarget, ...] = (
    # --- nhánh ccode: Python (BLS) gọi thẳng ⇒ đổi luôn sang UINT8 NHWC ---------
    FoldTarget(
        label="ccode_det_h",
        source="camera-containerNo/det-containerNo/cont_h_1.0_op12.t7",
        norm=DET_NORM_HORIZONTAL,
        input_shape=(3, 640, 672),
        uint8_nhwc=True,
    ),
    FoldTarget(
        label="ccode_det_v",
        source="camera-containerNo/det-containerNo/cont_v_3.3.1_op12.t7",
        norm=DET_NORM_VERTICAL,
        input_shape=(3, 512, 576),
        uint8_nhwc=True,
    ),
    FoldTarget(
        label="ccode_rec_h",
        source="camera-containerNo/rec-containerNo/cont_h_3.3_op11_151124.t7",
        norm=REC_NORM,
        input_shape=(3, 64, 256),
        uint8_nhwc=True,
    ),
    FoldTarget(
        label="ccode_rec_v",
        source="camera-containerNo/rec-containerNo/cont_v_3.3.1_opset11_051124.t7",
        norm=REC_NORM,
        input_shape=(3, 64, 256),
        uint8_nhwc=True,
    ),
    # --- nhánh crane/tcode: DeepStream gọi ⇒ GIỮ float32 NCHW ------------------
    FoldTarget(
        label="truckitems_pico",
        source="camera-crane/det-truckItems/truckItemsDetetion_111124.t7",
        norm=IDENTITY,
        input_shape=(3, 416, 416),
        swap_rb=True,
        # Chưa có ảnh mẫu của camera 10 (cẩu); dùng tạm một cảnh cảng thật.
        sample="samples/QC5/Cam06/H/20feet_1/vlcsnap-2024-10-18-16h02m14s432.png",
    ),
    FoldTarget(
        label="truckhead_pico",
        source="camera-truckNo/det-truckHead/truckHeadDet_261024.t7",
        norm=IDENTITY,
        input_shape=(3, 416, 416),
        swap_rb=True,
        sample="samples/QC5/Cam05/40feet_2/2700_2_36_camera5.jpg",  # đúng camera đầu kéo
    ),
    FoldTarget(
        label="headcode_cls",
        # Nguồn là bản đã reparameterize + đã gấp chuẩn hoá ImageNet (DN-003), KHÔNG
        # phải truckHeadCls_150125.t7 cũ.
        source="camera-truckNo/cls-truckHead/truckHeadCls_reparam.t7",
        norm=NO_SCALE,
        input_shape=(3, 224, 224),
        swap_rb=True,
        # Ảnh CÓ MÀU NHẤT trong 451 mẫu (|B-R| trung bình 132, so với trung vị 66) — cho
        # phép đối chứng đảo kênh cơ hội tốt nhất để phát hiện sai sót. Kể cả vậy, model
        # này vẫn ít nhạy với thứ tự kênh (bộ phân loại ký tự: hình dạng quyết định, không
        # phải màu), nên phép kiểm chứng thật của nó là bài đo độ chính xác trên cả 451
        # ảnh — xem docs/DESIGN_NOTES.md DN-012.
        sample=(
            "camera-truckNo/cls-truckHead/samples/other/"
            "pico_21102024_7488_1_21102024_032259_camera5_1-head-0.7550.jpg"
        ),
    ),
)


def affine(norm: Normalization) -> tuple[Array, Array]:
    """``(A, B)`` sao cho ``x*A - B ≡ (x/255 - mean)/std``, hình dạng ``(1,3,1,1)``."""
    scale = np.array([1.0 / (255.0 * s) for s in norm.std], dtype=np.float32)
    shift = np.array([m / s for m, s in zip(norm.mean, norm.std, strict=True)], dtype=np.float32)
    return scale.reshape(1, 3, 1, 1), shift.reshape(1, 3, 1, 1)


def _is_noop(scale: Array, shift: Array) -> bool:
    return bool(np.allclose(scale, 1.0) and np.allclose(shift, 0.0))


def fold(model: onnx.ModelProto, target: FoldTarget) -> onnx.ModelProto:
    """Chèn chuỗi tiền xử lý vào đầu đồ thị và khai lại kiểu/bố cục đầu vào."""
    graph = model.graph
    entry = graph.input[0]
    name = entry.name
    internal = f"{name}__pre"

    # Đổi tên TRƯỚC khi thêm node mới: nếu làm sau thì chính các node mới cũng bị đổi
    # theo và đồ thị thành vòng lặp.
    for node in graph.node:
        for i, value in enumerate(node.input):
            if value == name:
                node.input[i] = internal

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    current = name

    def step(node: onnx.NodeProto) -> str:
        nodes.append(node)
        return str(node.output[0])

    if target.uint8_nhwc:
        current = step(helper.make_node("Cast", [current], [f"{name}__f32"], to=TensorProto.FLOAT))
        current = step(
            helper.make_node("Transpose", [current], [f"{name}__nchw"], perm=[0, 3, 1, 2])
        )

    if target.swap_rb:
        idx = numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), f"{name}__rb")
        initializers.append(idx)
        current = step(
            helper.make_node("Gather", [current, idx.name], [f"{name}__swapped"], axis=1)
        )

    scale, shift = affine(target.norm)
    if not _is_noop(scale, shift):
        a = numpy_helper.from_array(scale, f"{name}__scale")
        b = numpy_helper.from_array(shift, f"{name}__shift")
        initializers += [a, b]
        current = step(helper.make_node("Mul", [current, a.name], [f"{name}__scaled"]))
        current = step(helper.make_node("Sub", [current, b.name], [f"{name}__shifted"]))

    if not nodes:
        raise RuntimeError(f"{target.label}: không có phép nào để gấp")

    # Node cuối phải sinh ra đúng tên mà phần còn lại của đồ thị đang đọc.
    nodes[-1].output[0] = internal

    graph.initializer.extend(initializers)
    for node in reversed(nodes):
        graph.node.insert(0, node)  # ONNX yêu cầu thứ tự tô-pô

    if target.uint8_nhwc:
        tensor_type = entry.type.tensor_type
        tensor_type.elem_type = TensorProto.UINT8
        dims = list(tensor_type.shape.dim)
        # NCHW -> NHWC, giữ nguyên chiều động nếu có.
        reordered = [dims[0], dims[2], dims[3], dims[1]]
        del tensor_type.shape.dim[:]
        tensor_type.shape.dim.extend(reordered)

    onnx.checker.check_model(model)
    return model


def _reference_input(target: FoldTarget) -> tuple[Array, Array]:
    """``(đầu vào cho đồ thị CŨ, đầu vào cho đồ thị MỚI)`` từ cùng một ảnh BGR thô."""
    import cv2

    path = ASSETS / target.sample
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"{target.label}: không đọc được ảnh kiểm chứng {path}")

    _, height, width = target.input_shape
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    bgr = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]  # NCHW, BGR

    # Đồ thị cũ chờ đúng thứ tự kênh mà nó được huấn luyện, và đã chuẩn hoá sẵn.
    old = bgr[:, ::-1, :, :].copy() if target.swap_rb else bgr
    mean = np.asarray(target.norm.mean, np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(target.norm.std, np.float32).reshape(1, 3, 1, 1)
    old = (old / 255.0 - mean) / std

    new: Array = bgr.transpose(0, 2, 3, 1).astype(np.uint8) if target.uint8_nhwc else bgr
    return old, new


@dataclass(frozen=True)
class VerifyResult:
    relative: float
    """Sai lệch lớn nhất, tính theo **bội số của ngưỡng cho phép**. ``<= 1`` là đạt."""

    control: float
    """Cùng thước đo, nhưng khi cố ý đưa SAI thứ tự kênh — đo sức phân biệt của phép thử."""

    @property
    def ok(self) -> bool:
        return self.relative <= 1.0 and self.control > self.control_floor

    @property
    def control_floor(self) -> float:
        """Đối chứng phải vượt mức này thì phép thử mới có ý nghĩa.

        Sàn ``1.0`` là bắt buộc và từng bị thiếu: khi ``relative ≈ 0`` thì
        ``relative * CONTROL_MARGIN`` cũng ≈ 0, nên MỌI giá trị đối chứng đều "đạt" — kể
        cả 0,5, tức là đưa sai hẳn thứ tự kênh mà kết quả không đổi. Đó chính là kiểu
        "đúng một cách vô nghĩa" mà phép đối chứng sinh ra để chặn.
        """
        return max(self.relative * CONTROL_MARGIN, 1.0)


def verify(original: bytes, folded: bytes, target: FoldTarget) -> VerifyResult:
    """Chạy cả hai đồ thị trên cùng một ảnh và chấm điểm phép gấp.

    Hai điều kiện, cả hai đều bắt buộc:

    1. **Khớp** — đồ thị cũ (nhận ảnh đã chuẩn hoá, đúng thứ tự kênh nó được huấn luyện)
       và đồ thị mới (nhận BGR thô) phải cho cùng kết quả.
    2. **Có sức phân biệt** — đưa cố ý sai thứ tự kênh vào đồ thị mới thì kết quả phải
       lệch hẳn. Thiếu bước này thì điều kiện (1) có thể đúng một cách vô nghĩa: một đồ
       thị hỏng tới mức bỏ qua đầu vào cũng sẽ "khớp" hoàn hảo.
    """
    import onnxruntime as ort

    old_input, new_input = _reference_input(target)
    channel_axis = 3 if target.uint8_nhwc else 1
    wrong_input = np.flip(new_input, axis=channel_axis).copy()

    def run(blob: bytes, feed: Array) -> list[Array]:
        sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
        return [
            np.asarray(o, dtype=np.float32)
            for o in sess.run(None, {sess.get_inputs()[0].name: feed})
        ]

    reference = run(original, old_input)
    session = ort.InferenceSession(folded, providers=["CPUExecutionProvider"])
    feed_name = session.get_inputs()[0].name

    def gap(feed: Array) -> float:
        """Bội số của ngưỡng cho phép. ``<= 1`` nghĩa là đạt.

        Quy mọi đầu ra về một con số so được với nhau, dù chúng ở các thang khác nhau.
        """
        outputs = [np.asarray(o, dtype=np.float32) for o in session.run(None, {feed_name: feed})]
        return max(
            float((np.abs(want - got) / (ATOL + RTOL * np.abs(want))).max())
            for want, got in zip(reference, outputs, strict=True)
        )

    return VerifyResult(relative=gap(new_input), control=gap(wrong_input))


def describe(target: FoldTarget) -> str:
    scale, shift = affine(target.norm)
    parts = []
    if target.uint8_nhwc:
        parts.append("UINT8 NHWC")
    if target.swap_rb:
        parts.append("đảo RGB→BGR")
    if not _is_noop(scale, shift):
        parts.append(
            f"A={np.round(scale.ravel(), 8).tolist()} B={np.round(shift.ravel(), 6).tolist()}"
        )
    return " · ".join(parts)


def process(target: FoldTarget, *, check_only: bool) -> bool:
    source = ASSETS / target.source
    dest = ASSETS / target.dest
    if not source.exists():
        print(f"  ❌ {target.label}: thiếu {source}", file=sys.stderr)
        return False

    original = decrypt_file(source)
    if check_only:
        if not dest.exists():
            print(f"  ❌ {target.label}: chưa sinh {dest.name}", file=sys.stderr)
            return False
        folded = decrypt_file(dest)
    else:
        folded = fold(onnx.load_from_string(original), target).SerializeToString()

    result = verify(original, folded, target)
    print(
        f"  {'✅' if result.ok else '❌'} {target.label:<18}"
        f"lệch {result.relative:6.3f}x ngưỡng  (đối chứng sai kênh {result.control:9.1f}x)   "
        f"{describe(target)}"
    )
    if not result.ok:
        why = (
            f"lệch gấp {result.relative:.2f} lần ngưỡng cho phép"
            if result.relative > 1.0
            else f"phép thử KHÔNG có sức phân biệt: đưa SAI thứ tự kênh chỉ lệch "
            f"{result.control:.2f}x ngưỡng (cần > {result.control_floor:.2f}x) — ảnh kiểm "
            f"chứng gần như đơn sắc nên không phân biệt được BGR với RGB. Đổi ảnh khác."
        )
        print(f"     {why} — KHÔNG ghi {dest.name}", file=sys.stderr)
        return False

    if not check_only:
        dest.write_bytes(encrypt_bytes(folded))
        if decrypt_file(dest) != folded:
            print(f"     ❌ {dest.name} giải mã lại không khớp", file=sys.stderr)
            return False
        print(f"     → {dest.name}  ({len(folded) / 1e6:.1f} MB)")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="chỉ kiểm chứng file đã sinh")
    ap.add_argument("--only", nargs="*", help="chỉ xử lý các nhãn này")
    args = ap.parse_args(argv)

    targets = [t for t in TARGETS if not args.only or t.label in args.only]
    print(f"Gấp tiền xử lý vào đồ thị — {len(targets)} model, đích: mọi model nhận BGR thô\n")

    # Cố ý dùng list chứ không phải generator: generator dừng ở model hỏng đầu tiên, còn
    # ở đây ta muốn thấy TẤT CẢ model nào sai.
    results = [process(t, check_only=args.check) for t in targets]
    ok = all(results)
    print("\n✅ tất cả khớp" if ok else "\n❌ có model không khớp")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
