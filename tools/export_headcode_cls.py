"""Xuất model phân loại số đầu kéo (FastViT-T8) sang ONNX.

Model này **không** đi chung đường với 6 model kia vì hai lý do:

1. **Nguồn của nó là checkpoint Keras ``best_acc.h5`` (41 MB), không phải ``.t7``.** Sáu
   model kia đi qua ``tools/export_models.py`` (giải mã ``.t7`` → ONNX); model này không
   có ``.t7`` nào để giải mã.

2. **Bắt buộc phải reparameterize trước khi xuất.** Đây là điểm mấu chốt về tốc độ.

---

## Vì sao phải reparameterize

Đã kiểm chứng bằng ``h5py`` trên chính file ``best_acc.h5`` (2026-08-29):

* ``model_config.name == "fastvit_t8"``, Keras 2.15
* 261 layer, trong đó có layer ``resmlp>ChannelAffine`` — dấu nhận biết của
  ``keras_cv_attention_models`` (kecam)
* **10 layer ``Subtract``** tên ``stackN_blockM_REPARAM_TWICE_out``, mỗi cái nhận
  ``..._mixer_REPARAM_out`` và ``..._mixer_bn``

Tên layer nói thẳng ra vấn đề: đây là FastViT ở **dạng huấn luyện**, các khối RepMixer còn
nguyên nhiều nhánh song song. FastViT (và họ MobileOne nói chung) được thiết kế để huấn
luyện với nhiều nhánh rồi **hợp nhất toán học** thành một convolution duy nhất khi suy luận
— hai dạng cho **cùng kết quả**, nhưng dạng đã hợp nhất chạy nhanh hơn hẳn.

Xuất thẳng dạng train-time sang ONNX nghĩa là mang theo toàn bộ nhánh thừa: 10 khối
RepMixer x (depthwise conv + BN + BN + Subtract + Add + ChannelAffine) thay vì 10 depthwise
conv. TensorRT sẽ fuse được một phần, nhưng không thể tự làm phép hợp nhất *toán học* của
reparameterization — nó không biết hai nhánh đó tương đương một conv.

## Cách chạy

Cần TensorFlow + kecam, **không** nằm trong dependency chính của dự án (quá nặng, và chỉ
dùng một lần). Chạy trong môi trường tạm::

    uv run --with tensorflow --with keras-cv-attention-models --with tf2onnx \\
        python -m tools.export_headcode_cls \\
            --h5  /ssd1/huylg/dnp_project/smartport/assets/camera-truckNo/cls-truckHead/best_acc.h5 \\
            --out /tmp/craneops-export/craneops_headcode_cls.onnx

Sau đó ``trtexec`` dựng plan như các model khác (xem ``tools/export_models.py``).

## Bắt buộc kiểm chứng sau khi xuất

Reparameterization là phép biến đổi *toán học tương đương*, nhưng sai số dấu phẩy động và
lỗi hiện thực đều có thể len vào. Script này **luôn** so đầu ra trước/sau khi hợp nhất trên
dữ liệu ngẫu nhiên và báo sai lệch lớn nhất. Nếu vượt ngưỡng thì dừng, không xuất.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

N_CLASSES = 54
"""Khớp assets/camera-truckNo/cls-truckHead/label.txt (53 số xe + "other")."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""Chuẩn hoá kiểu "torch" của kecam. Xác nhận bằng thực nghiệm 2026-08-29: chạy ONNX trên
451 ảnh có nhãn ở cls-truckHead/samples/ cho **100,0 % top-1**, trong khi BGR được 87,6 %,
tf/inception 99,6 %, raw01 96,0 %, không chuẩn hoá 3,1 %."""

INPUT_HW = (224, 224)
"""Đo từ model_config: batch_input_shape = [None, 224, 224, 3]."""

RTOL = 1e-4
ATOL = 1e-4


def _grouped_conv_to_depthwise(model, tf, np):  # type: ignore[no-untyped-def]
    """Đổi ``Conv2D(groups == in_channels)`` thành ``DepthwiseConv2D`` tương đương.

    Vì sao cần: TensorFlow **không có kernel grouped-conv trên CPU**, nên Keras gói mỗi
    layer như vậy trong một ``tf.function``. Trong đồ thị nó hiện ra là ``PartitionedCall``,
    và ``tf2onnx`` không nhìn xuyên qua được — ONNX sinh ra không hợp lệ (đã gặp: 14 op).

    Sau khi kecam hợp nhất RepMixer, 14 layer rơi vào dạng này, ví dụ
    ``stack2_downsample_REPARAM_1_conv``: ``groups=48, in_ch=48, filters=96``. Vì
    ``groups == in_ch`` nên nó *chính là* depthwise conv với
    ``depth_multiplier = filters // in_ch``. ``DepthwiseConv2D`` có kernel CPU thật và
    chuyển sang ONNX sạch sẽ.

    Bố cục trọng số:

    * ``Conv2D`` grouped: ``[kh, kw, in_ch // groups, filters]`` = ``[kh, kw, 1, G*M]``,
      kênh ra xếp theo nhóm (nhóm ``g`` cho ra các kênh ``g*M .. g*M+M-1``)
    * ``DepthwiseConv2D``: ``[kh, kw, in_ch, depth_multiplier]`` = ``[kh, kw, G, M]``

    Nên chỉ cần bỏ chiều thứ ba rồi reshape — không hoán vị gì.
    """
    Conv2D, DepthwiseConv2D = tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D
    converted: dict[str, int] = {}

    def swap(layer):  # type: ignore[no-untyped-def]
        if (
            isinstance(layer, Conv2D)
            and not isinstance(layer, DepthwiseConv2D)
            and getattr(layer, "groups", 1) > 1
        ):
            in_ch = layer.input_shape[-1]
            if layer.groups == in_ch and layer.filters % in_ch == 0:
                converted[layer.name] = layer.filters // in_ch
                return DepthwiseConv2D(
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    dilation_rate=layer.dilation_rate,
                    depth_multiplier=layer.filters // in_ch,
                    use_bias=layer.use_bias,
                    activation=layer.activation,
                    name=layer.name,
                )
        return layer.__class__.from_config(layer.get_config())

    new_model = tf.keras.models.clone_model(model, clone_function=swap)

    for old in model.layers:
        weights = old.get_weights()
        if not weights:
            continue
        new = new_model.get_layer(old.name)
        if old.name in converted:
            mult = converted[old.name]
            k = weights[0]  # [kh, kw, 1, G*M]
            kh, kw, _, gm = k.shape
            weights = [k.reshape(kh, kw, gm // mult, mult), *weights[1:]]
        new.set_weights(weights)

    return new_model, converted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Xuất FastViT-T8 phân loại số đầu kéo sang ONNX")
    ap.add_argument("--h5", type=Path, required=True, help="checkpoint Keras best_acc.h5")
    ap.add_argument("--out", type=Path, required=True, help="file ONNX đầu ra")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument(
        "--fold-preprocess",
        action="store_true",
        help=(
            "gấp chuẩn hoá ImageNet vào model; ONNX sẽ nhận RGB thô [0,255] "
            "(khuyến nghị — xem docstring)"
        ),
    )
    ap.add_argument(
        "--nchw",
        action="store_true",
        help="chuyển input sang NCHW cho đồng bộ với 6 model kia (khuyến nghị)",
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="bỏ qua bước so sánh trước/sau hợp nhất (KHÔNG khuyến khích)",
    )
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "❌ Cần TensorFlow. Chạy lại bằng:\n"
            "   uv run --with tensorflow --with keras-cv-attention-models --with tf2onnx \\\n"
            "       python -m tools.export_headcode_cls ...",
            file=sys.stderr,
        )
        return 1

    # Layer tuỳ biến của kecam được đăng ký qua @register_keras_serializable khi module
    # được import — nếu không import trước thì load_model ném
    # "Unknown layer: 'resmlp>ChannelAffine'".
    try:
        import keras_cv_attention_models  # noqa: F401
        from keras_cv_attention_models import fastvit  # noqa: F401
    except ImportError:
        print(
            "❌ Cần keras-cv-attention-models (thêm --with keras-cv-attention-models)",
            file=sys.stderr,
        )
        return 1

    print(f"▪ nạp {args.h5}")
    model = tf.keras.models.load_model(args.h5, compile=False)
    print(f"  {model.name}  in={model.input_shape}  out={model.output_shape}")

    if model.output_shape[-1] != N_CLASSES:
        print(
            f"❌ model cho {model.output_shape[-1]} lớp, label.txt có {N_CLASSES}",
            file=sys.stderr,
        )
        return 1

    n_before = len(model.layers)
    sample = np.random.default_rng(0).random((2, *INPUT_HW, 3), dtype=np.float32)
    out_before = model(sample, training=False).numpy() if not args.skip_verify else None

    # ---- reparameterize -----------------------------------------------------
    # KHÔNG gọi thẳng fastvit.switch_to_deploy(): bước cuối của nó là
    # add_pre_post_process(..., rescale_mode=model.preprocess_input.rescale_mode), mà
    # model nạp từ .h5 là `Functional` thuần, không có thuộc tính kecam gắn thêm đó.
    #
    # Cũng KHÔNG gọi thẳng fuse_reparam_blocks(): nó giả định Conv+BN đã hợp nhất trước,
    # nếu không sẽ ném IndexError khi đọc trọng số của layer BN.
    #
    # Chuỗi dưới đây chép đúng từ mã nguồn kecam 1.4.3
    # (fastvit/fastvit.py:switch_to_deploy), bỏ mỗi bước gắn pre/post process.
    from keras_cv_attention_models.model_surgery.model_surgery import (
        convert_to_fused_conv_bn_model,
        fuse_channel_affine_to_conv_dense,
        fuse_reparam_blocks,
    )

    steps = (
        ("hợp nhất Conv+BN", lambda m: convert_to_fused_conv_bn_model(m)),
        ("gộp nhánh REPARAM_out", lambda m: fuse_reparam_blocks(m, output_layer_key="REPARAM_out")),
        (
            "gộp nhánh REPARAM_TWICE_out",
            lambda m: fuse_reparam_blocks(m, output_layer_key="REPARAM_TWICE_out"),
        ),
        ("gộp ChannelAffine vào conv", lambda m: fuse_channel_affine_to_conv_dense(m)),
        (
            "gộp nhánh REPARAM_THIRD_out",
            lambda m: fuse_reparam_blocks(m, output_layer_key="REPARAM_THIRD_out"),
        ),
        (
            "grouped Conv2D -> DepthwiseConv2D",
            lambda m: _grouped_conv_to_depthwise(m, tf, np)[0],
        ),
    )

    deployed = model
    for label, step in steps:
        before = len(deployed.layers)
        deployed = step(deployed)
        print(f"  {label:<32} {before:>4} -> {len(deployed.layers):>4} layer")

    if deployed is None:
        print(
            "❌ Không tìm thấy hàm reparameterize của kecam.\n"
            "   Thử tay: `model.switch_to_deploy()` hoặc\n"
            "   `keras_cv_attention_models.model_surgery.convert_to_deploy(model)`.\n"
            "   ⚠️ ĐỪNG xuất dạng train-time — sẽ chậm hơn nhiều. Xem docstring module.",
            file=sys.stderr,
        )
        return 1

    n_after = len(deployed.layers)
    print(f"  reparameterize: {n_before} layer -> {n_after} layer (giảm {n_before - n_after})")
    if n_after >= n_before:
        print(
            "❌ Số layer không giảm — nhiều khả năng hợp nhất KHÔNG chạy. Dừng lại.",
            file=sys.stderr,
        )
        return 1

    # ---- kiểm chứng tương đương ---------------------------------------------
    if not args.skip_verify:
        out_after = deployed(sample, training=False).numpy()
        diff = float(np.abs(out_before - out_after).max())
        print(f"  sai lệch lớn nhất trước/sau hợp nhất: {diff:.3e}")
        if not np.allclose(out_before, out_after, rtol=RTOL, atol=ATOL):
            print(
                f"❌ Hợp nhất làm ĐỔI kết quả (max {diff:.3e} > atol {ATOL}). "
                f"Không xuất — cần điều tra trước.",
                file=sys.stderr,
            )
            return 1
        argmax_same = int((out_before.argmax(-1) == out_after.argmax(-1)).all())
        print(f"  argmax giữ nguyên: {'có' if argmax_same else 'KHÔNG'}")

    # ---- gấp chuẩn hoá vào model -------------------------------------------
    # Vì sao nên gấp: DeepStream `nvinfer`/`nvinferserver` chuẩn hoá theo công thức
    # ``y = net-scale-factor * (x - offset)`` — **một** hệ số vô hướng dùng chung cho cả ba
    # kênh, cộng offset theo kênh. Nhưng std của ImageNet khác nhau theo kênh
    # (0.229 / 0.224 / 0.225), nên **không biểu diễn được** bằng cấu hình đó.
    #
    # Nếu ép dùng std trung bình thì sai lệch nhỏ nhưng âm thầm, và không có gì báo. Gấp
    # phép chuẩn hoá vào chính model thì DeepStream chỉ cần `net-scale-factor=1.0`, và
    # không còn chỗ nào để cấu hình sai.
    if args.fold_preprocess:
        scale = [1.0 / (255.0 * sd) for sd in IMAGENET_STD]
        offset = [-mu / sd for mu, sd in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]
        raw_in = tf.keras.Input(shape=(*INPUT_HW, 3), name="input", dtype=tf.float32)
        normalized = tf.keras.layers.Rescaling(scale=scale, offset=offset, name="imagenet_norm")(
            raw_in
        )
        deployed = tf.keras.Model(raw_in, deployed(normalized), name="fastvit_t8_deploy")
        print("  đã gấp chuẩn hoá ImageNet vào model (input = RGB thô [0,255])")

    # ---- xuất ONNX ----------------------------------------------------------
    # Cả `from_keras` lẫn đường SavedModel đều để lại 14 op `PartitionedCall` không chuyển
    # đổi được — ONNX sinh ra "xuất thành công" nhưng KHÔNG HỢP LỆ (onnxruntime báo
    # "No Op registered for PartitionedCall"). Nguyên nhân: phép hợp nhất của kecam tạo ra
    # các layer mà TF gói trong lời gọi tf.function (`*_mlp_pre_conv`, `*_REPARAM_1_conv`).
    #
    # Cách chắc ăn: ĐÓNG BĂNG đồ thị. `convert_variables_to_constants_v2` chạy grappler với
    # function-inlining, làm phẳng hết PartitionedCall thành op nguyên thuỷ.
    try:
        import tf2onnx
    except ImportError:
        print("❌ Cần tf2onnx (thêm --with tf2onnx)", file=sys.stderr)
        return 1

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("▪ đóng băng đồ thị (inline PartitionedCall)")
    concrete = tf.function(lambda x: deployed(x)).get_concrete_function(
        tf.TensorSpec((None, *INPUT_HW, 3), tf.float32, name="input")
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()

    n_pc = sum(1 for n in graph_def.node if n.op == "PartitionedCall")
    print(f"  node trong graph_def: {len(graph_def.node)}  PartitionedCall còn lại: {n_pc}")
    if n_pc:
        print(
            f"❌ Còn {n_pc} PartitionedCall sau khi đóng băng — ONNX sẽ không hợp lệ.",
            file=sys.stderr,
        )
        return 1

    in_names = [t.name for t in frozen.inputs]
    out_names = [t.name for t in frozen.outputs]
    print(f"  input={in_names}  output={out_names}")

    print(f"▪ xuất ONNX opset {args.opset}{' (NCHW)' if args.nchw else ''} -> {args.out}")
    tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=in_names,
        output_names=out_names,
        opset=args.opset,
        inputs_as_nchw=in_names if args.nchw else None,
        output_path=str(args.out),
    )

    # ---- đổi tên tensor cho sạch ---------------------------------------------
    # tf2onnx đặt tên theo frozen graph: "input:0" và "Identity:0". Dấu ":0" là rác của
    # TensorFlow, và trtexec không nhận tên có dấu hai chấm trong --minShapes.
    import onnx

    renamed = onnx.load(str(args.out))
    mapping = {
        renamed.graph.input[0].name: "input",
        renamed.graph.output[0].name: "head",
    }
    for vi in list(renamed.graph.input) + list(renamed.graph.output):
        vi.name = mapping.get(vi.name, vi.name)
    for node in renamed.graph.node:
        node.input[:] = [mapping.get(x, x) for x in node.input]
        node.output[:] = [mapping.get(x, x) for x in node.output]
    onnx.checker.check_model(renamed)
    onnx.save(renamed, str(args.out))
    print(f"  đổi tên tensor: {mapping}")

    # ---- kiểm chứng ONNX cho ra ĐÚNG kết quả --------------------------------
    # Số node đẹp không đồng nghĩa model chạy đúng. Bắt buộc so số học.
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  thiếu onnxruntime — BỎ QUA kiểm chứng số học (không nên)", file=sys.stderr)
    else:
        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        probe = sample * 255.0 if args.fold_preprocess else sample
        feed = probe.transpose(0, 3, 1, 2) if args.nchw else probe
        onnx_out = sess.run(None, {iname: feed})[0]
        ref = deployed(probe, training=False).numpy()
        d = float(np.abs(ref - onnx_out).max())
        ok = bool((ref.argmax(-1) == onnx_out.argmax(-1)).all())
        print(f"  Keras vs ONNX: sai lệch {d:.3e}, argmax {'khớp' if ok else 'LỆCH'}")
        if not ok or d > 1e-3:
            print("❌ ONNX cho kết quả khác Keras. Không dùng được.", file=sys.stderr)
            return 1

    print(
        f"\n✅ xong. Bước tiếp theo:\n"
        f"   trtexec --onnx={args.out} \\\n"
        f"     --saveEngine=triton/repo/craneops/craneops_headcode_cls/1/model.plan \\\n"
        f"     --fp16 --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 \\\n"
        f"     --maxShapes=input:16x3x224x224"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
