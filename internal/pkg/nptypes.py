"""Bí danh kiểu cho mảng ảnh.

Vì sao cần: kho này chạy ``mypy --strict``, nhưng stub của OpenCV khai mọi tham số ảnh là
``Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]``. Kết quả là mọi hàm nhận
``NDArray[np.uint8]`` rồi truyền vào ``cv2`` đều sinh lỗi giả — dtype cụ thể của numpy
không khớp union của cv2. Ép kiểu ở từng lời gọi sẽ rải ``cast()`` khắp nơi và làm code
khó đọc hơn nhiều so với giá trị nó mang lại.

Dùng bí danh này cho **mảng đi qua OpenCV**. Ở những chỗ khác (logit, toạ độ, tensor
chuẩn hoá) vẫn dùng ``NDArray[np.float32]`` hay tương đương để mypy còn bắt được lỗi thật.

dtype thực tế được ghi trong docstring của từng hàm — nó là hợp đồng, chỉ không được
kiểm tự động.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

Image: TypeAlias = "np.ndarray[Any, Any]"
"""Ảnh ``(H, W, 3)`` uint8, thứ tự kênh BGR — trừ khi docstring nói khác."""

Array: TypeAlias = "np.ndarray[Any, Any]"
"""Mảng numpy bất kỳ đi qua OpenCV."""

__all__ = ["Array", "Image", "np"]
