# Image cho `ds_app` — DeepStream 8.0 + pyds + phụ thuộc Python của dự án.
#
#   docker build -t craneops-ds:dev -f build/ds_app.Dockerfile .
#
# Vì sao có image riêng thay vì `docker run … pip install` lúc chạy: cài đặt lúc chạy làm
# mỗi lần chạy phụ thuộc mạng, chậm, và — quan trọng hơn — **không tái lập được**. Hai lần
# chạy cách nhau một tháng sẽ có hai bộ thư viện khác nhau dưới cùng một lệnh.
#
# Mã nguồn KHÔNG nằm trong image: compose mount `..:/app:ro`. Sửa code rồi chạy lại là thấy
# ngay, không phải build lại. Đổi Dockerfile này mới cần build lại.

ARG DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.0-triton-multiarch
FROM ${DEEPSTREAM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    USE_NEW_NVSTREAMMUX=no
# ⚠️ USE_NEW_NVSTREAMMUX=no là BẮT BUỘC, và nó phải là biến môi trường chứ không phải
# thuộc tính element: plugin `nvstreammux` đọc biến này **lúc nạp** để chọn mux cũ hay mới.
# DS8 mặc định dùng mux MỚI, vốn bỏ qua toàn bộ thuộc tính của mux cũ (`batch-size`,
# `width`, `height`, `batched-push-timeout`, `live-source`) — và hậu quả là nó **không bao
# giờ đẩy một batch nào**, im lặng. Đặt ở đây để không ai chạy thiếu nó.

WORKDIR /app

# --- phụ thuộc hệ thống: plugin GStreamer cho RTSP/decode, và bộ dựng cho PyGObject -------
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        build-essential \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-libav \
        gstreamer1.0-rtsp \
        gir1.2-gstreamer-1.0 \
        libgirepository1.0-dev \
        libcairo2-dev \
        pkg-config \
        python3-dev \
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/*

# --- pyds: binding Python cho metadata của DeepStream --------------------------------------
# Cần cho probe đọc `NvDsFrameMeta`/`NvDsObjectMeta`. Image DeepStream KHÔNG kèm sẵn; nó chỉ
# kèm script cài. Ghim phiên bản: pyds phải khớp bản DeepStream, và một bản lệch sẽ hỏng ở
# tầng đọc cấu trúc C — tức dữ liệu rác, không phải ImportError.
#
# Hai cái bẫy ở bước này, cả hai đều làm mất thời gian:
#
# 1. Script kèm theo cài pyds vào **venv riêng của nó**, không vào Python hệ thống — nên
#    `import pyds` vẫn hỏng dù nó báo "Successfully installed". Phải lấy chính wheel nó tải
#    về rồi cài lại vào Python hệ thống.
#
# 2. ⚠️ **KHÔNG kiểm bằng `import pyds` trong lúc build.** Lúc `docker build` không có GPU,
#    nên `/usr/lib/x86_64-linux-gnu/libcuda.so.1` chỉ là stub rỗng và import chết với
#    `file too short`. Thư viện thật do NVIDIA Container Toolkit cấp **lúc chạy**. Kiểm ở
#    đây là kiểm một điều kiện không thể đúng; chỉ kiểm file `.so` có mặt.
ARG PYDS_VERSION=1.2.2
RUN set -eu; \
    /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version "${PYDS_VERSION}"; \
    WHEEL="$(find /opt/nvidia/deepstream -name "pyds-${PYDS_VERSION}-*.whl" | head -1)"; \
    if [ -z "$WHEEL" ]; then echo "không thấy wheel pyds sau khi chạy script cài" >&2; exit 1; fi; \
    pip3 install --no-cache-dir --break-system-packages "$WHEEL"; \
    python3 -c "import sysconfig, pathlib, sys; \
so = pathlib.Path(sysconfig.get_paths()['purelib']) / 'pyds.so'; \
sys.exit(0) if so.exists() else sys.exit(f'thiếu {so} sau khi cài wheel')"

# --- phụ thuộc Python của dự án ------------------------------------------------------------
# Chỉ những gói `ds_app` thật sự dùng. Không cài cả bộ: image đã 21 GB, và mỗi gói thừa là
# một thứ nữa phải vá khi có CVE.
RUN pip3 install --no-cache-dir --break-system-packages \
        "PyGObject==3.48.2" \
        "pydantic>=2.7,<3" \
        "pyyaml>=6,<7" \
        "numpy>=1.26,<2" \
        "loguru>=0.7,<0.8" \
        "tritonclient[grpc]>=2.41,<3" \
        "kafka-python-ng>=2.2,<3" \
        "lz4>=4.3,<5"
# tritonclient: ds_app gọi model BLS của Triton qua gRPC. Chỉ extra `grpc` — `http` kéo
# thêm aiohttp/geventhttpclient mà nhánh này không dùng.
#
# ⚠️ KHÔNG cần opencv ở đây. Phép resize về 416x416 nằm trong BLS phía Triton, và để nó ở
# đó là cố ý: một bản cv2 thứ hai trong image này là một chỗ nữa để phiên bản OpenCV lệch
# nhau, mà `INTER_CUBIC` khác bản là hộp lệch tới 17 px (HARDWARE_BUDGET §6.2).

# Không có CMD mặc định: compose chọn chế độ chạy (record / full pipeline).
CMD ["python3", "-c", "import pyds, gi; print('ds_app image sẵn sàng')"]
