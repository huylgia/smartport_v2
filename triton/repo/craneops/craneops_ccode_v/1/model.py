"""Điểm vào Python backend cho mã container DỌC.

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
