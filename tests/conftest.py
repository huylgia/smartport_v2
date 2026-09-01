"""Tiện ích dùng chung cho test.

Chỉ chứa thứ mà **nhiều** thư mục test cần. Thứ riêng của một nhóm thì ở conftest của
nhóm đó.

⚠️ Không còn hàm sinh env cho camera. Trước đây URL nằm ở biến môi trường nên
``camera_code`` không tái tạo được nếu thiếu env, và test phải mang theo một fixture chứa
host + cổng thật chỉ để dựng lại đúng mã. Giờ định danh luồng nằm trong chính
``configs/cranes/*.yaml``, nên ``load_crane(GC03, env={})`` cho đúng mã production —
fixture đó biến mất cùng nguyên nhân sinh ra nó.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GC03 = REPO / "configs" / "cranes" / "GC03.yaml"
