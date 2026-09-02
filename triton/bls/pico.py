"""Bộ chuyển tiếp Triton Python backend cho hai nhánh dùng PicoDet.

Cùng khuôn với :mod:`triton.bls.ccode`: bóc tensor, gọi ``internal/pkg/vision/``, đóng gói
kết quả. Không có nghiệp vụ nào ở đây.

Vì sao gộp phát hiện + phân loại vào MỘT model BLS thay vì dùng PGIE→SGIE của DeepStream:
``nvinferserver`` chỉ dựng được ``NvDsObjectMeta`` — thứ SGIE cần để biết cắt ở đâu — khi
nó tự parse được đầu ra của detector. PicoDet trả tensor thô (``tmp_16`` 3598x4 +
``concat_8.tmp_0`` Cx3598), và ``DetectionParams.nms`` trong ``nvdsinferserver_common.proto``
ghi rõ *"reserved, not supported yet"*. Đường còn lại là viết parser C++ — tức **bản thứ hai**
của NMS đã port và đã đo ở ``internal/pkg/vision/nms.py``, kể cả cái ``+1`` cố ý trong công
thức IoU. Hai bản cài đặt của một thuật toán bug-compatible là thứ chắc chắn trôi khỏi nhau.

Gộp vào BLS giữ đúng một bản NMS, và vẫn được ``dynamic_batching`` gom request từ nhiều
camera vì mỗi lời gọi con đi qua bộ lập lịch của Triton.

Hợp đồng tensor::

    craneops_crane   vào  image   UINT8 [-1, -1, 3]   khung hình BGR
                     ra   labels  BYTES [-1]          "head" | "container"
                          scores  FP32  [-1]
                          boxes   INT32 [-1, 4]       x_min, y_min, x_max, y_max (ảnh gốc)

    craneops_tcode   vào  image   UINT8 [-1, -1, 3]
                     ra   labels  BYTES [-1]          luôn là "head"
                          scores  FP32  [-1]          điểm của DETECTOR
                          boxes   INT32 [-1, 4]
                          codes   INT32 [-1]          chỉ số lớp số xe, -1 nếu không đọc
                          code_scores FP32 [-1]       điểm của CLASSIFIER

Số phần tử ở mọi output bằng nhau và thay đổi theo từng request — có thể bằng 0.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

# Mã nguồn mount ở /app (build/docker-compose.triton.yml). Python backend chạy interpreter
# riêng với sys.path của nó, nên phải tự thêm vào.
_APP_ROOT = os.environ.get("CRANEOPS_APP_ROOT", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from internal.pkg.nptypes import Array, Image  # noqa: E402
from internal.pkg.vision.pico import Detection, PicoParams, crop, detect  # noqa: E402

CLS_INPUT_SIZE = (224, 224)
"""``(cao, rộng)`` đầu vào của ``craneops_headcode_cls``."""


class _PicoBase:
    """Phần dùng chung: đọc parameters, gọi detector, trả ba output đầu."""

    def initialize(self, args: dict[str, str]) -> None:
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}

        self._det_model = params["det_model"]
        self._labels = {
            int(k): v for k, v in json.loads(params.get("labels", '{"0": "head"}')).items()
        }
        self._params = PicoParams(
            score_threshold=float(params.get("score_threshold", 0.3)),
            nms_threshold=float(params.get("nms_threshold", 0.3)),
        )
        self._setup(params)

    def _setup(self, params: dict[str, str]) -> None:
        """Móc cho lớp con đọc thêm parameters. Mặc định không làm gì."""

    def execute(self, requests: list[Any]) -> list[Any]:
        return [self._handle(request) for request in requests]

    def _handle(self, request: Any) -> Any:
        try:
            image = np.ascontiguousarray(
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            )
            found = detect(image, self._run_det, self._labels, self._params)
            extra = self._extra_outputs(image, found)
        # Một request hỏng không được giết cả instance: các camera khác dùng chung.
        except Exception as exc:
            return pb_utils.InferenceResponse(
                output_tensors=[], error=pb_utils.TritonError(f"{type(exc).__name__}: {exc}")
            )

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "labels",
                    np.array([d.label.encode("utf-8") for d in found], dtype=np.object_),
                ),
                pb_utils.Tensor("scores", np.array([d.score for d in found], dtype=np.float32)),
                pb_utils.Tensor(
                    "boxes", np.array([d.box for d in found], dtype=np.int32).reshape(-1, 4)
                ),
                *extra,
            ]
        )

    def _extra_outputs(self, image: Image, found: list[Detection]) -> list[Any]:
        return []

    # -- lời gọi BLS ---------------------------------------------------------

    def _run_det(self, tensor: Array) -> tuple[Array, Array]:
        """Trả ``(hộp, điểm)`` đã bỏ chiều batch."""
        outputs = self._infer(self._det_model, "image", tensor, ["tmp_16", "concat_8.tmp_0"])
        return outputs["tmp_16"][0], outputs["concat_8.tmp_0"][0]

    def _infer(
        self, model: str, input_name: str, tensor: Array, outputs: list[str]
    ) -> dict[str, Array]:
        response = pb_utils.InferenceRequest(
            model_name=model,
            requested_output_names=outputs,
            # GIỮ NGUYÊN kiểu dữ liệu mà đường ống tạo ra — kiểu là hợp đồng của model,
            # không phải lựa chọn của lớp chuyển tiếp này.
            inputs=[pb_utils.Tensor(input_name, np.ascontiguousarray(tensor))],
            # BẮT BUỘC: model TensorRT trả tensor nằm trong VRAM và `.as_numpy()` trên đó
            # ném "Tensor is stored in GPU and cannot be converted to NumPy". NMS chạy
            # bằng numpy trên CPU nên dù sao cũng phải chép về host.
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0),
        ).exec()
        if response.has_error():
            raise pb_utils.TritonModelException(f"{model} lỗi: {response.error().message()}")
        return {
            name: pb_utils.get_output_tensor_by_name(response, name).as_numpy() for name in outputs
        }


class CraneModel(_PicoBase):
    """``craneops_crane`` — đầu kéo + container trên camera nhìn xuống."""


class TCodeModel(_PicoBase):
    """``craneops_tcode`` — khoanh đầu kéo rồi đọc số xe trên chính khung đó.

    Phân loại **gộp một batch** cho mọi đầu kéo trong khung, không gọi từng cái một: gọi
    lẻ vô hiệu hoá ``dynamic_batching`` của ``craneops_headcode_cls``, và đó là thứ duy
    nhất làm model 806 mẫu/s @b1 lên 1 819 @b4 (HARDWARE_BUDGET §6.1).
    """

    def _setup(self, params: dict[str, str]) -> None:
        self._cls_model = params["cls_model"]
        self._cls_output = params.get("cls_output", "head")

    def _extra_outputs(self, image: Image, found: list[Detection]) -> list[Any]:
        codes = np.full(len(found), -1, dtype=np.int32)
        code_scores = np.zeros(len(found), dtype=np.float32)

        crops, rows = [], []
        for i, det in enumerate(found):
            region = crop(image, det.box)
            if region.size == 0:
                # Hộp nằm trọn ngoài ảnh. Giữ chỗ với code -1 thay vì bỏ hàng: mọi output
                # phải cùng số phần tử, và nơi gọi khớp chúng theo chỉ số.
                continue
            crops.append(region)
            rows.append(i)

        if crops:
            target_h, target_w = CLS_INPUT_SIZE
            batch = np.stack(
                [cv2.resize(c, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for c in crops]
            )
            tensor = batch.astype(np.float32).transpose(0, 3, 1, 2)
            # ⚠️ ĐÃ LÀ XÁC SUẤT, không phải logit — softmax nằm trong đồ thị ONNX. Đo
            # 2026-09-02: mỗi hàng cộng đúng 1,0, mọi giá trị trong [0, 1], không có số âm.
            #
            # Áp softmax lần nữa "cho chắc" là một lỗi câm: điểm cao nhất tụt từ 1,0 xuống
            # 0,0488, và `TCODE01` so với `head_code_thresh` 0,93 nên rule sẽ KHÔNG BAO GIỜ
            # phát signal — không exception, không log, chỉ là số xe không bao giờ đọc được.
            probs = self._infer(self._cls_model, "input", tensor, [self._cls_output])[
                self._cls_output
            ]
            codes[rows] = probs.argmax(axis=1).astype(np.int32)
            code_scores[rows] = probs.max(axis=1).astype(np.float32)

        return [
            pb_utils.Tensor("codes", codes),
            pb_utils.Tensor("code_scores", code_scores),
        ]
