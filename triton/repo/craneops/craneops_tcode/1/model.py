"""Điểm vào Python backend cho nhánh tcode.

SINH TỰ ĐỘNG bởi tools/export_models.py — đừng sửa tay.

Triton bắt buộc lớp phải tên ``TritonPythonModel`` và nằm trong ``model.py``. Toàn bộ nội
dung ở ``triton/bls/pico.py``; file này chỉ đặt bí danh.
"""

import os
import sys

_APP_ROOT = os.environ.get("CRANEOPS_APP_ROOT", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from triton.bls.pico import TCodeModel as TritonPythonModel  # noqa: E402, F401
