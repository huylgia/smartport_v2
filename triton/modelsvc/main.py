"""Chuẩn bị Triton model repository: kiểm license → giải mã → dựng engine → xoá bản rõ.

Chạy **trước** Triton, dưới dạng ``ExecStartPre`` của unit systemd. Thoát khác 0 ⇒ Triton
không khởi động. Đây là chỗ duy nhất trong hệ thống chạm vào model ở dạng rõ.

    .t7 (mã hoá, trên đĩa)
        │  kiểm license gắn phần cứng
        ▼
    .onnx  →  /dev/shm  (tmpfs, xem SCRATCH_DIR)
        │  trtexec  (FP32 cho ccode — xem DN-008/DN-013; FP16 cho pico/cls)
        ▼
    .plan  →  giữ lại trong volume `craneops_models`
    .onnx  →  XOÁ

Vì sao là tmpfs: nội dung nằm trong RAM và mất hẳn khi container dừng, nên bản rõ không
bao giờ chạm đĩa. ``prepare_model`` **kiểm** điều này lúc chạy chứ không tin cấu hình.

Đánh đổi phải nói rõ — và đã ghi trong ``docs/DESIGN_NOTES.md``: khi Triton đang chạy thì
file ``.plan`` vẫn đọc được bởi ai có quyền root trên máy. Đây là rào cản đáng kể chứ
không phải bảo vệ tuyệt đối, và ghi ra đây để không ai nhầm.

Cách dùng::

    python -m triton.modelsvc.main --config configs/cranes/GC03.yaml
    python -m triton.modelsvc.main --check          # chỉ kiểm license, không đụng model
    python -m triton.modelsvc.main --force          # dựng lại kể cả khi plan còn mới
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from internal.pkg.security.cipher import DecryptionError, decrypt_file
from internal.pkg.security.license import LicenseError, validate
from tools.export_models import (
    ALL_SPECS,
    ASSETS,
    TRITON_REPO,
    ModelSpec,
    check_health,
    make_batch_dynamic,
)

__all__ = [
    "TRITON_REPO",
    "PreparedModel",
    "install_python_models",
    "main",
    "prepare_model",
    "prepare_repository",
]

DEFAULT_REPO = Path("/models")
"""Nơi đặt engine. Volume **thường** để engine sống qua các lần restart — dựng lại 6 model
mất ~9 phút."""

SCRATCH_DIR = Path("/dev/shm")  # noqa: S108 — tmpfs là ĐÚNG chỗ cho bản rõ
"""Nơi ghi bản rõ ONNX trong lúc dựng engine. tmpfs ⇒ không bao giờ chạm đĩa."""


def _is_tmpfs(path: Path) -> bool:
    """``path`` có thật sự nằm trên tmpfs không.

    Không tin cấu hình mà **kiểm**: compose từng khai ``/dev/shm`` như một mục trong
    ``volumes:``, và Docker biến nó thành anonymous volume trên ext4 — bản rõ ONNX ghi
    xuống ĐĨA trong khi mọi tài liệu khẳng định ngược lại. Một sai sót một dòng trong
    compose không được phép âm thầm phá bất biến bảo mật của service này.
    """
    try:
        with Path("/proc/mounts").open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == str(path):
                    return parts[2] == "tmpfs"
    except OSError:
        return False
    return False


PLAN_NAME = "model.plan"
MODEL_VERSION = "1"

PYTHON_MODELS = (
    "craneops_ccode_h",
    "craneops_ccode_v",
    "craneops_crane",
    "craneops_tcode",
)
"""Model chạy Python backend (BLS): mỗi cái điều phối trọn một nhánh nghiệp vụ.

Không phải ``ensemble``, và cùng một lý do cho cả bốn: **số tensor ra khác số tensor vào**
nên đồ thị tĩnh không diễn đạt được. Hai model ccode có cổng nét loại bớt crop (DN-007);
hai model pico trả số hộp thay đổi theo từng khung.

``craneops_crane``/``craneops_tcode`` còn một lý do riêng: chúng thay cho pattern
PGIE→SGIE của DeepStream, vốn cần ``nvinferserver`` tự parse được đầu ra PicoDet để dựng
``NvDsObjectMeta``. Nó không parse được (``DetectionParams.nms`` là *"reserved, not
supported yet"*), và đường vòng duy nhất — parser C++ — sẽ là bản thứ hai của NMS đã port
ở ``internal/pkg/vision/nms.py``."""

TRTEXEC_FALLBACKS = (
    "/usr/src/tensorrt/bin/trtexec",
    "/opt/tensorrt/bin/trtexec",
)
"""Image Triton của NVIDIA có ``trtexec`` nhưng KHÔNG đưa vào ``PATH`` — nó nằm ở
``/usr/src/tensorrt/bin``. Dò các vị trí chuẩn thay vì bắt người dùng tự truyền đường dẫn."""


def resolve_trtexec(explicit: str) -> str | None:
    """Đường dẫn ``trtexec`` dùng được, hoặc ``None``.

    Ưu tiên giá trị người dùng truyền vào, rồi ``PATH``, rồi các vị trí chuẩn của image.
    """
    if explicit != "trtexec":
        return explicit if Path(explicit).exists() else None
    found = shutil.which("trtexec")
    if found:
        return found
    return next((c for c in TRTEXEC_FALLBACKS if Path(c).exists()), None)


@dataclass(frozen=True, slots=True)
class PreparedModel:
    name: str
    plan: Path
    rebuilt: bool
    """``False`` nghĩa là dùng lại engine cũ còn mới hơn nguồn."""

    seconds: float


BUILD_STAMP = "build.json"
"""Ghi lại các cờ đã dùng để dựng engine, đặt cạnh chính engine đó."""


def _plan_is_fresh(plan: Path, source: Path, spec: ModelSpec) -> bool:
    """Engine còn dùng được nếu nó mới hơn ``.t7`` nguồn VÀ dựng bằng đúng cờ hiện tại.

    Dựng engine TensorRT tốn hàng chục giây mỗi model, nên bỏ qua khi không cần là khác
    biệt giữa khởi động lại trong 10 giây và trong 3 phút.

    Vì sao phải so cả cờ chứ không chỉ mtime: đổi ``fp16`` hay ``trt_profile`` KHÔNG làm
    file ``.t7`` mới hơn, nên nếu chỉ nhìn mtime thì engine cũ sẽ được dùng lại vĩnh viễn
    và thay đổi cấu hình âm thầm không có tác dụng. Đây là loại lỗi không báo gì cả —
    mọi thứ vẫn READY, chỉ là chạy sai thứ mình nghĩ.
    """
    if not plan.exists() or plan.stat().st_mtime < source.stat().st_mtime:
        return False
    stamp = plan.parent / BUILD_STAMP
    if not stamp.exists():
        return False
    try:
        recorded = json.loads(stamp.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(recorded.get("args") == _trtexec_args(spec))


def _trtexec_args(spec: ModelSpec) -> list[str]:
    """Cờ dựng engine, KHÔNG kể đường dẫn — dùng làm chữ ký để phát hiện engine lỗi thời."""
    args = ["--fp16"] if spec.fp16 else []
    for name, (lo, opt, hi) in spec.trt_profile.items():
        args += [
            f"--minShapes={name}:{lo}",
            f"--optShapes={name}:{opt}",
            f"--maxShapes={name}:{hi}",
        ]
    return args


def _run_trtexec(onnx: Path, spec: ModelSpec, plan: Path, *, trtexec: str) -> None:
    cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={plan}", *_trtexec_args(spec)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        # trtexec in RẤT nhiều log, phần lớn ra STDOUT chứ không phải stderr, và 15 dòng
        # cuối thường chỉ là cảnh báo. Gộp cả hai luồng rồi ưu tiên các dòng có nhãn [E].
        lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
        errors = [ln for ln in lines if "[E]" in ln]
        tail = "\n".join(errors[-12:] if errors else lines[-12:])

        # Mã thoát âm = bị tín hiệu giết. Với trtexec gần như luôn là SIGKILL do OOM:
        # trình dựng engine TensorRT ngốn RAM, nhất là với batch lớn và model transformer.
        if proc.returncode < 0:
            raise RuntimeError(
                f"trtexec bị giết bởi tín hiệu {-proc.returncode} khi dựng {spec.name} "
                f"(SIGKILL = 9, gần như chắc chắn là OOM).\n"
                f"Tăng MODELSVC_MEMORY trong .env.triton, hoặc hạ max_batch_size / "
                f"maxShapes của model này trong tools/export_models.py.\n{tail}"
            )
        if "Unsupported SM" in proc.stdout + proc.stderr:
            raise RuntimeError(
                f"trtexec không hỗ trợ kiến trúc GPU của máy này khi dựng {spec.name}.\n"
                f"Phiên bản TensorRT trong image quá cũ so với GPU. Dùng image Triton mới hơn.\n"
                f"Lưu ý: engine TensorRT gắn với KIẾN TRÚC GPU, nên phải dựng trên chính máy\n"
                f"sẽ chạy — engine dựng ở máy dev không dùng được ở máy đích.\n{tail}"
            )
        raise RuntimeError(f"trtexec thất bại cho {spec.name}:\n{tail}")


def prepare_model(
    spec: ModelSpec,
    repo: Path,
    *,
    trtexec: str = "trtexec",
    force: bool = False,
) -> PreparedModel:
    """Giải mã một model, dựng engine, rồi xoá bản rõ.

    Raises:
        FileNotFoundError: thiếu file ``.t7``.
        DecryptionError: sai mật khẩu hoặc file hỏng.
        RuntimeError: ONNX không đạt kiểm tra sức khoẻ, hoặc ``trtexec`` thất bại.
    """
    if spec.source is None:
        raise RuntimeError(
            f"{spec.name} không có nguồn .t7 — model này phải được sinh riêng "
            f"(xem tools/export_headcode_cls.py)"
        )

    started = time.monotonic()
    source = ASSETS / spec.source
    if not source.exists():
        raise FileNotFoundError(f"thiếu model đã mã hoá: {source}")

    dest_dir = repo / spec.name / MODEL_VERSION
    plan = dest_dir / PLAN_NAME

    # Đồng bộ config.pbtxt TRƯỚC kiểm tra "engine còn mới": đổi ngưỡng batching không
    # cần dựng lại engine, nhưng vẫn phải tới được Triton.
    _install_config(spec.name, repo)

    if not force and _plan_is_fresh(plan, source, spec):
        return PreparedModel(spec.name, plan, rebuilt=False, seconds=0.0)

    dest_dir.mkdir(parents=True, exist_ok=True)
    onnx_bytes = decrypt_file(source)

    problems = check_health(onnx_bytes, spec)
    if problems:
        raise RuntimeError(f"{spec.name} không đạt kiểm tra sức khoẻ: {'; '.join(problems)}")

    if spec.needs_batch_patch:
        onnx_bytes = make_batch_dynamic(onnx_bytes)

    # Bản rõ đi vào tmpfs, KHÔNG vào cùng thư mục với engine (thư mục đó nằm trên đĩa).
    if not _is_tmpfs(SCRATCH_DIR):
        raise RuntimeError(
            f"{SCRATCH_DIR} không phải tmpfs — bản rõ model sẽ bị ghi xuống đĩa.\n"
            f"Trong compose, khai nó ở mục `tmpfs:` chứ KHÔNG phải `volumes:`; một mục "
            f"`- /dev/shm` trong `volumes:` tạo anonymous volume trên đĩa.\n"
            f"Kiểm bằng: docker compose --env-file build/.env.triton "
            f"-f build/docker-compose.triton.yml run --rm --no-deps "
            f"--entrypoint sh modelsvc -c 'df -h {SCRATCH_DIR}'"
        )
    scratch = SCRATCH_DIR
    onnx_path = scratch / f"{spec.name}.onnx"
    try:
        # Ghi 0600 rồi hạ xuống 0400: không để tồn tại khoảnh khắc nào file readable
        # cho nhóm/khác.
        fd = os.open(onnx_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(onnx_bytes)
        onnx_path.chmod(0o400)

        _run_trtexec(onnx_path, spec, plan, trtexec=trtexec)
        plan.chmod(0o400)
        (dest_dir / BUILD_STAMP).write_text(
            json.dumps({"args": _trtexec_args(spec)}, indent=2), encoding="utf-8"
        )
    finally:
        # Bản rõ KHÔNG được sống sót qua hàm này, kể cả khi trtexec nổ.
        onnx_path.unlink(missing_ok=True)

    return PreparedModel(spec.name, plan, rebuilt=True, seconds=time.monotonic() - started)


def _install_config(name: str, repo: Path) -> None:
    """Chép ``config.pbtxt`` từ mã nguồn vào model repository.

    **Bắt buộc, không phải trang trí.** Triton chạy với ``--strict-model-config=false``
    nên nếu thư mục model không có ``config.pbtxt`` thì nó TỰ SUY config từ file
    ``.plan``. Config tự suy đó bỏ qua mọi thứ ta khai:

    * ``max_queue_delay_microseconds`` về **0** ⇒ dynamic batching chỉ gom được những
      request đã nằm sẵn trong hàng đợi. Crop từ 5 camera ccode đến lệch nhau vài ms sẽ
      **không** được gộp — đúng thứ tối ưu mà kiến trúc này tồn tại để làm.
    * ``instance_group.count`` về 1 ⇒ hậu xử lý Python chạy một tiến trình, không tận
      dụng được 20 luồng CPU.

    Triệu chứng khi thiếu: mọi thứ vẫn READY, kết quả vẫn đúng, chỉ throughput thấp —
    nên rất dễ không nhận ra.
    """
    src = TRITON_REPO / name / "config.pbtxt"
    if not src.exists():
        raise FileNotFoundError(f"thiếu {src} — chạy: python -m tools.export_models --emit-config")
    dest = repo / name / "config.pbtxt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def install_python_models(repo: Path, *, names: tuple[str, ...] = PYTHON_MODELS) -> list[str]:
    """Chép nguyên thư mục các model Python backend (BLS) vào repository.

    Khác model TensorRT: không có gì để dựng, không có gì để giải mã — chỉ là mã nguồn
    ``model.py`` + ``config.pbtxt``. Chép thay vì mount để Triton thấy một repository
    duy nhất, và để bản đang chạy không đổi khi ai đó sửa file nguồn giữa chừng.
    """
    installed = []
    for name in names:
        src = TRITON_REPO / name
        if not src.is_dir():
            raise FileNotFoundError(f"thiếu thư mục model Python backend: {src}")
        dest = repo / name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        installed.append(name)
    return installed


def prepare_repository(
    license_key: str,
    repo: Path = DEFAULT_REPO,
    *,
    trtexec: str = "trtexec",
    force: bool = False,
    specs: tuple[ModelSpec, ...] = ALL_SPECS,
) -> list[PreparedModel]:
    """Kiểm license rồi chuẩn bị toàn bộ model repository.

    License được kiểm **trước tiên và đúng một lần**: hỏng thì không giải mã gì cả.

    Raises:
        LicenseError: khoá không hợp lệ, hết hạn, hoặc không khớp phần cứng.
    """
    validate(license_key)

    repo.mkdir(parents=True, exist_ok=True)
    repo.chmod(0o700)

    prepared = [prepare_model(spec, repo, trtexec=trtexec, force=force) for spec in specs]
    install_python_models(repo)
    return prepared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="thư mục model repository")
    ap.add_argument("--trtexec", default="trtexec", help="đường dẫn trtexec")
    ap.add_argument("--force", action="store_true", help="dựng lại kể cả khi plan còn mới")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm license rồi thoát")
    ap.add_argument(
        "--license-key",
        default=os.environ.get("CRANEOPS_LICENSE_KEY", ""),
        help="mặc định lấy từ CRANEOPS_LICENSE_KEY",
    )
    args = ap.parse_args(argv)

    if not args.license_key:
        print(
            "❌ Thiếu khoá bản quyền. Đặt CRANEOPS_LICENSE_KEY hoặc truyền --license-key.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = validate(args.license_key)
    except LicenseError as exc:
        print(f"❌ {exc.message}", file=sys.stderr)
        for key in ("hint", "fingerprint", "expected_device", "this_device"):
            if key in exc.details:
                print(f"   {key}: {exc.details[key]}", file=sys.stderr)
        return 1
    expiry = (
        time.strftime("%Y-%m-%d", time.localtime(payload.expires_at))
        if payload.expires_at
        else "vô thời hạn"
    )
    print(f"✅ giấy phép hợp lệ  ({payload.note or 'không ghi chú'}, hạn: {expiry})")

    if args.check:
        return 0

    trtexec = resolve_trtexec(args.trtexec)
    if trtexec is None:
        print(
            f"❌ Không tìm thấy trtexec (đã thử PATH và {', '.join(TRTEXEC_FALLBACKS)}). "
            f"Nó đi kèm TensorRT — chạy service này trong container Triton, hoặc trỏ "
            f"--trtexec vào đúng đường dẫn.",
            file=sys.stderr,
        )
        return 1
    print(f"   trtexec: {trtexec}")

    try:
        prepared = prepare_repository(
            args.license_key, args.repo, trtexec=trtexec, force=args.force
        )
    except (DecryptionError, FileNotFoundError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    built = sum(1 for p in prepared if p.rebuilt)
    for p in prepared:
        state = f"dựng {p.seconds:.0f}s" if p.rebuilt else "dùng lại"
        print(f"  {p.name:<28} {state}")
    print(f"\n✅ {len(prepared)} model sẵn sàng tại {args.repo} ({built} dựng mới)")

    leftovers = list(args.repo.rglob("*.onnx"))
    if leftovers:
        print(f"❌ còn {len(leftovers)} file ONNX bản rõ sót lại: {leftovers}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
