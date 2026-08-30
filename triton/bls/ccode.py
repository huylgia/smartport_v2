"""Bộ chuyển tiếp Triton Python backend cho đường ống mã container.

Chỉ làm ba việc: bóc tensor ra khỏi request, gọi
:class:`internal.pkg.vision.ccode_pipeline.CCodePipeline`, đóng gói kết quả trả về. Toàn bộ
nghiệp vụ nằm ở ``internal/pkg/`` để test được mà không cần Triton.

Model repository nạp file này qua ``model.py`` mỏng trong từng thư mục model — hai model
``craneops_ccode_h`` và ``craneops_ccode_v`` dùng chung lớp này, khác nhau ở phần
``parameters`` của ``config.pbtxt``.

Hợp đồng tensor::

    vào   image   UINT8  [-1, -1, 3]   ảnh ROI, BGR, kích thước tuỳ ý
          params  BYTES  [1]           JSON tham số ROI (tuỳ chọn)
    ra    texts      BYTES [-1]        chuỗi đã đọc
          scores     FP32  [-1]        độ tin cậy OCR
          boxes      INT32 [-1, 4]     x_min, y_min, x_max, y_max trên ảnh ROI
          quads      INT32 [-1, 4, 2]  4 đỉnh
          det_scores FP32  [-1]        độ tin cậy của detector
          sharpness  FP32  [-1]        độ nét của crop

Số phần tử ở mọi output bằng nhau và **thay đổi theo từng request** — có thể bằng 0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils

# Mã nguồn được mount ở /app (xem build/docker-compose.triton.yml). Python backend chạy
# interpreter riêng với sys.path của nó, nên phải tự thêm vào.
_APP_ROOT = os.environ.get("CRANEOPS_APP_ROOT", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from internal.pkg.nptypes import Array  # noqa: E402
from internal.pkg.vision.ccode_pipeline import CCodePipeline, RoiParams  # noqa: E402
from internal.pkg.vision.ctc import load_char_dict  # noqa: E402

DEFAULT_CHAR_DICT = "/assets/char_dict/char_dict.txt"


class CCodeModel:
    """Bản dùng chung cho cả hai model. ``model.py`` chỉ đặt bí danh ``TritonPythonModel``."""

    def initialize(self, args: dict[str, str]) -> None:
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}

        self._det_model = params["det_model"]
        self._det_output = params["det_output"]
        self._rec_model = params["rec_model"]
        self._rec_output = params["rec_output"]
        vertical = params.get("vertical", "false").lower() == "true"

        char_dict_path = Path(params.get("char_dict", DEFAULT_CHAR_DICT))
        if not char_dict_path.exists():
            raise pb_utils.TritonModelException(
                f"không tìm thấy bảng ký tự: {char_dict_path}. Mount assets vào container "
                f"Triton, hoặc đặt parameters.char_dict trong config.pbtxt."
            )

        self._pipeline = CCodePipeline(
            vertical=vertical,
            # ⚠️ Chỉ số 37 (dấu cách) không bao giờ ra được: recognizer chỉ có 37 cột
            # (0..36) trong khi bảng ký tự đánh số 1..37. Đây là thuộc tính của chính
            # model đã huấn luyện, không phải lỗi ở đây — "sửa" bằng cách dịch chỉ số sẽ
            # làm lệch TOÀN BỘ bảng ký tự.
            char_dict=load_char_dict(char_dict_path),
            det_infer=self._run_det,
            rec_infer=self._run_rec,
        )

    def execute(self, requests: list[Any]) -> list[Any]:
        return [self._handle(request) for request in requests]

    def _handle(self, request: Any) -> Any:
        try:
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            raw = pb_utils.get_input_tensor_by_name(request, "params")
            params = (
                RoiParams.from_mapping(json.loads(raw.as_numpy()[0].decode("utf-8")))
                if raw is not None
                else RoiParams()
            )
            results, _stats = self._pipeline.run(np.ascontiguousarray(image), params)
        # Một request hỏng không được phép giết cả instance: các camera khác dùng chung.
        except Exception as exc:
            return pb_utils.InferenceResponse(
                output_tensors=[], error=pb_utils.TritonError(f"{type(exc).__name__}: {exc}")
            )

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "texts",
                    np.array([r.text.encode("utf-8") for r in results], dtype=np.object_),
                ),
                pb_utils.Tensor("scores", np.array([r.score for r in results], dtype=np.float32)),
                pb_utils.Tensor(
                    "boxes", np.array([r.box for r in results], dtype=np.int32).reshape(-1, 4)
                ),
                pb_utils.Tensor(
                    "quads",
                    np.array([r.quad for r in results], dtype=np.int32).reshape(-1, 4, 2),
                ),
                pb_utils.Tensor(
                    "det_scores", np.array([r.det_score for r in results], dtype=np.float32)
                ),
                pb_utils.Tensor(
                    "sharpness", np.array([r.sharpness for r in results], dtype=np.float32)
                ),
            ]
        )

    # -- lời gọi BLS ---------------------------------------------------------

    def _run_det(self, tensor: Array) -> Array:
        return self._infer(self._det_model, "x", tensor, self._det_output)

    def _run_rec(self, tensor: Array) -> Array:
        return self._infer(self._rec_model, "x", tensor, self._rec_output)

    def _infer(self, model: str, input_name: str, tensor: Array, output_name: str) -> Array:
        """Gọi một model TensorRT trong cùng Triton.

        Lời gọi đi qua bộ lập lịch của Triton, nên các request từ những instance BLS
        khác nhau (tức camera khác nhau) vẫn được ``dynamic_batching`` gom lại — đây là
        thứ giữ được cái lợi của ensemble mà không mất tính linh hoạt.
        """
        response = pb_utils.InferenceRequest(
            model_name=model,
            requested_output_names=[output_name],
            # GIỮ NGUYÊN kiểu dữ liệu mà đường ống tạo ra. Ép cứng ``np.float32`` ở đây
            # từng làm hỏng cả nhánh khi model chuyển sang nhận UINT8: Triton từ chối với
            # "data-type is 'FP32', but model expects 'UINT8'". Kiểu là hợp đồng của
            # model, không phải lựa chọn của lớp chuyển tiếp này.
            inputs=[pb_utils.Tensor(input_name, np.ascontiguousarray(tensor))],
            # BẮT BUỘC. Model TensorRT trả tensor nằm trong VRAM; `.as_numpy()` trên đó
            # ném "Tensor is stored in GPU and cannot be converted to NumPy". Hậu xử lý
            # DB chạy bằng numpy/cv2 trên CPU nên dù sao cũng phải chép về host — khai
            # báo ở đây để Triton tự chép, thay vì đi đường DLPack rồi tự chép tay.
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0),
        ).exec()
        if response.has_error():
            raise pb_utils.TritonModelException(f"{model} lỗi: {response.error().message()}")
        out: Array = pb_utils.get_output_tensor_by_name(response, output_name).as_numpy()
        return out
