# Triton Inference Server + modelsvc.
#
# Một image dùng cho cả hai service trong docker-compose.triton.yml: `modelsvc` chạy
# `python3 -m triton.modelsvc.main`, `triton` chạy `tritonserver`. Dùng chung image vì
# modelsvc cần `trtexec` — thứ đi kèm TensorRT trong image này.
#
#   docker build -t craneops-triton:dev -f build/triton.Dockerfile .

# 25.10 chứ không phải 24.08: TensorRT trong 24.08 chỉ hỗ trợ tới Hopper (sm_90) và báo
# "Unsupported SM: 0xc00" trên GPU Blackwell — tức máy dev (RTX 5090, sm_120) không dựng
# được engine.
#
# Máy ĐÍCH là RTX 3060 (sm_86) nên 24.08 cũng đủ cho nó, nhưng dùng chung MỘT phiên bản
# cho cả dev lẫn production thì bớt được một biến số khi truy lỗi.
ARG TRITON_VERSION=25.10
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

# dmidecode để đọc serial BIOS khi /sys/class/dmi không đọc được. Không cần sudo —
# tiến trình trong container đã là root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends dmidecode curl \
    && rm -rf /var/lib/apt/lists/*

# Hai nhóm, đừng gộp nhầm:
#   1. modelsvc  — giải mã .t7, kiểm sức khoẻ ONNX, kiểm license
#   2. Python backend của Triton (BLS mã container) — hậu xử lý DB và cắt/nắn ảnh
# Không kéo cả dependency của service nghiệp vụ vào đây.
RUN python3 -m pip install --no-cache-dir \
      "cryptography>=43,<44" \
      "onnx>=1.16,<2" \
      "numpy>=1.26,<2" \
      "opencv-python-headless>=4.9,<5" \
      "pyclipper>=1.3,<2" \
      "shapely>=2.0,<3"

WORKDIR /app

# Mã nguồn được MOUNT lúc chạy (xem compose), không COPY vào image — để đổi code không
# phải build lại image, và để image không mang theo mã nguồn.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Mặc định chạy Triton; modelsvc ghi đè bằng `command` trong compose.
CMD ["tritonserver", "--model-repository=/models"]
