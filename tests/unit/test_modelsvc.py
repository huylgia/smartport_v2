from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from internal.pkg.security import license as lic
from internal.pkg.security.fingerprint import DeviceFingerprint
from tools.export_models import ModelSpec, Shape
from triton.modelsvc import main as modelsvc

DEVICE = DeviceFingerprint(sources={"dmi_uuid": "u", "gpu": "g"})


def _fake_engine(repo: Path, spec: ModelSpec, *, stamp: bool = True) -> Path:
    """Engine giả đã dựng xong, kèm dấu build như prepare_model thật để lại.

    Thiếu dấu build ⇒ engine bị coi là lỗi thời (xem modelsvc._plan_is_fresh).
    """
    plan = repo / spec.name / "1" / modelsvc.PLAN_NAME
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_bytes(b"engine")
    if stamp:
        (plan.parent / modelsvc.BUILD_STAMP).write_text(
            json.dumps({"args": modelsvc._trtexec_args(spec)})
        )
    return plan


@pytest.fixture
def valid_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Giấy phép hợp lệ cho một thiết bị giả lập."""
    key = Ed25519PrivateKey.generate()
    monkeypatch.setenv(lic.ENV_PUBLIC_KEY, lic.public_key_b64(key))
    monkeypatch.setattr(modelsvc, "collect", lambda: DEVICE, raising=False)
    monkeypatch.setattr("internal.pkg.security.license.collect", lambda: DEVICE)
    return lic.issue(DEVICE.digest, key, note="test")


# ---------------------------------------------------------------- modelsvc


@pytest.fixture
def fake_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelSpec:
    """Một model giả: .t7 mã hoá thật, nhưng nội dung không phải ONNX."""
    from tests.unit.security.test_cipher import _encrypt

    assets = tmp_path / "assets"
    (assets / "fake").mkdir(parents=True)
    (assets / "fake" / "m.t7").write_bytes(_encrypt(b"khong-phai-onnx" * 100, "pw"))
    monkeypatch.setattr(modelsvc, "ASSETS", assets)
    monkeypatch.setenv("CRANEOPS_MODEL_PASSWORD", "pw")

    # config.pbtxt nguồn phải tồn tại: prepare_model chép nó sang repo trước khi dựng
    # engine. Thiếu file này là lỗi cấu hình thật, nên hàm ném FileNotFoundError.
    source_repo = tmp_path / "src_repo"
    (source_repo / "fake_model").mkdir(parents=True)
    (source_repo / "fake_model" / "config.pbtxt").write_text('name: "fake_model"\n')
    monkeypatch.setattr(modelsvc, "TRITON_REPO", source_repo)

    return ModelSpec(
        name="fake_model",
        source="fake/m.t7",
        inputs=(Shape("x", (3, 8, 8)),),
        outputs=(Shape("y", (2,)),),
        max_batch_size=1,
    )


def test_plaintext_onnx_is_deleted_even_when_trtexec_fails(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bất biến quan trọng nhất của service này.

    Nếu trtexec nổ giữa chừng mà bản rõ còn nằm lại thì model bị lộ cho tới lần reboot sau.
    """
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])

    def boom(*_args: object, **_kw: object) -> None:
        raise RuntimeError("trtexec giả vờ hỏng")

    monkeypatch.setattr(modelsvc, "_run_trtexec", boom)

    repo = tmp_path / "repo"
    scratch = tmp_path / "shm"
    scratch.mkdir()
    monkeypatch.setattr(modelsvc, "SCRATCH_DIR", scratch)
    # tmp_path nằm trên đĩa; các test này kiểm bất biến khác nên bỏ qua chốt tmpfs.
    # Bản thân chốt đó có test riêng bên dưới.
    monkeypatch.setattr(modelsvc, "_is_tmpfs", lambda _p: True)

    with pytest.raises(RuntimeError, match="trtexec giả vờ hỏng"):
        modelsvc.prepare_model(fake_spec, repo)

    assert list(repo.rglob("*.onnx")) == []
    assert list(scratch.rglob("*.onnx")) == [], "bản rõ phải bị xoá khỏi tmpfs"


def test_successful_run_leaves_only_the_plan(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])

    def fake_trtexec(onnx: Path, _spec: ModelSpec, plan: Path, **_kw: object) -> None:
        assert onnx.exists(), "trtexec phải thấy được ONNX lúc chạy"
        plan.write_bytes(b"engine gia lap")

    monkeypatch.setattr(modelsvc, "_run_trtexec", fake_trtexec)

    repo = tmp_path / "repo"
    scratch = tmp_path / "shm"
    scratch.mkdir()
    monkeypatch.setattr(modelsvc, "SCRATCH_DIR", scratch)
    # tmp_path nằm trên đĩa; các test này kiểm bất biến khác nên bỏ qua chốt tmpfs.
    # Bản thân chốt đó có test riêng bên dưới.
    monkeypatch.setattr(modelsvc, "_is_tmpfs", lambda _p: True)

    result = modelsvc.prepare_model(fake_spec, repo)

    assert list(scratch.rglob("*.onnx")) == [], "bản rõ phải bị xoá khỏi tmpfs"
    assert result.rebuilt
    assert result.plan.exists()
    assert list(repo.rglob("*.onnx")) == []
    assert stat.S_IMODE(result.plan.stat().st_mode) == 0o400


def test_unhealthy_onnx_never_reaches_trtexec(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kiểm tra sức khoẻ chặn TRƯỚC khi dựng engine — đúng lỗi của truckHeadCls_150125.t7."""
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: ["depthwise bị bung"])
    monkeypatch.setattr(
        modelsvc, "_run_trtexec", lambda *_a, **_k: pytest.fail("không được gọi trtexec")
    )

    with pytest.raises(RuntimeError, match="depthwise bị bung"):
        modelsvc.prepare_model(fake_spec, tmp_path / "repo")


def test_fresh_plan_is_reused(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dựng engine tốn hàng chục giây mỗi model — khởi động lại không được dựng lại vô ích."""
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])
    monkeypatch.setattr(
        modelsvc, "_run_trtexec", lambda *_a, **_k: pytest.fail("không được dựng lại")
    )

    repo = tmp_path / "repo"
    _fake_engine(repo, fake_spec)

    result = modelsvc.prepare_model(fake_spec, repo)
    assert not result.rebuilt


def test_force_rebuilds_even_when_fresh(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])
    calls: list[str] = []

    def record(_o: Path, _s: ModelSpec, plan: Path, **_k: object) -> None:
        calls.append("x")
        plan.write_bytes(b"moi")

    monkeypatch.setattr(modelsvc, "_run_trtexec", record)

    repo = tmp_path / "repo"
    plan = repo / fake_spec.name / "1" / "model.plan"
    plan.parent.mkdir(parents=True)
    plan.write_bytes(b"cu")

    assert modelsvc.prepare_model(fake_spec, repo, force=True).rebuilt
    assert calls == ["x"]


def test_missing_source_is_reported_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modelsvc, "ASSETS", tmp_path / "khong-ton-tai")
    spec = ModelSpec(name="m", source="a/b.t7", inputs=(), outputs=(), max_batch_size=1)
    with pytest.raises(FileNotFoundError, match="thiếu model đã mã hoá"):
        modelsvc.prepare_model(spec, tmp_path / "repo")


def test_spec_without_source_is_rejected(tmp_path: Path) -> None:
    """craneops_headcode_cls không đến từ .t7 — phải báo rõ thay vì crash mơ hồ."""
    spec = ModelSpec(name="m", source=None, inputs=(), outputs=(), max_batch_size=1)
    with pytest.raises(RuntimeError, match=r"không có nguồn \.t7"):
        modelsvc.prepare_model(spec, tmp_path / "repo")


def test_bad_license_stops_before_touching_any_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """License phải được kiểm TRƯỚC TIÊN — hỏng thì không giải mã gì cả."""
    monkeypatch.setattr(
        modelsvc, "prepare_model", lambda *_a, **_k: pytest.fail("không được giải mã")
    )
    with pytest.raises(lic.LicenseError):
        modelsvc.prepare_repository("khong-hop-le", tmp_path / "repo")


# ---------------------------------------------------------------- CLI


def test_cli_without_license_key_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRANEOPS_LICENSE_KEY", raising=False)
    assert modelsvc.main([]) == 2


def test_cli_check_validates_and_exits_0(valid_token: str) -> None:
    assert modelsvc.main(["--check", "--license-key", valid_token]) == 0


def test_cli_reports_missing_trtexec(valid_token: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("triton.modelsvc.main.shutil.which", lambda _n: None)
    assert modelsvc.main(["--license-key", valid_token]) == 1


def test_cli_rejects_bad_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        modelsvc, "prepare_repository", lambda *_a, **_k: pytest.fail("không được chạy")
    )
    assert modelsvc.main(["--license-key", "khong-hop-le"]) == 1


def test_trtexec_failure_surfaces_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Log của trtexec rất dài; chỉ giữ phần đuôi — nơi chứa lỗi thật."""

    def fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "dòng thừa\n" * 40 + "LỖI THẬT Ở ĐÂY")

    monkeypatch.setattr("triton.modelsvc.main.subprocess.run", fake_run)
    spec = ModelSpec(name="m", source="x", inputs=(), outputs=(), max_batch_size=1)
    with pytest.raises(RuntimeError, match="LỖI THẬT Ở ĐÂY"):
        modelsvc._run_trtexec(tmp_path / "a.onnx", spec, tmp_path / "a.plan", trtexec="trtexec")


# ---------------------------------------------------------------- config.pbtxt


def test_config_pbtxt_is_copied_into_repo(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bảo vệ một lỗi đã thật sự xảy ra và không hề báo lỗi.

    Triton chạy với ``--strict-model-config=false``. Thiếu ``config.pbtxt`` thì nó TỰ SUY
    config từ file ``.plan``, mọi model vẫn READY, kết quả vẫn đúng — chỉ là
    ``max_queue_delay_microseconds`` về 0 và ``instance_group.count`` về 1, tức mất phần
    lớn throughput. Không có test này thì lỗi chỉ lộ ra khi đo hiệu năng.
    """
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])
    monkeypatch.setattr(modelsvc, "_run_trtexec", lambda *_a, **_k: None)
    scratch = tmp_path / "shm"
    scratch.mkdir()
    monkeypatch.setattr(modelsvc, "SCRATCH_DIR", scratch)
    # tmp_path nằm trên đĩa; các test này kiểm bất biến khác nên bỏ qua chốt tmpfs.
    # Bản thân chốt đó có test riêng bên dưới.
    monkeypatch.setattr(modelsvc, "_is_tmpfs", lambda _p: True)

    repo = tmp_path / "repo"
    _fake_engine(repo, fake_spec)

    modelsvc.prepare_model(fake_spec, repo, trtexec="trtexec")

    assert (repo / fake_spec.name / "config.pbtxt").read_text() == 'name: "fake_model"\n'


def test_config_pbtxt_is_synced_even_when_engine_is_reused(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Đổi ngưỡng batching không cần dựng lại engine, nhưng vẫn phải tới được Triton."""
    monkeypatch.setattr(
        modelsvc, "_run_trtexec", lambda *_a, **_k: pytest.fail("không được dựng lại")
    )
    repo = tmp_path / "repo"
    plan = _fake_engine(repo, fake_spec)
    os.utime(plan, (time.time() + 100, time.time() + 100))  # engine mới hơn .t7

    source_config = modelsvc.TRITON_REPO / fake_spec.name / "config.pbtxt"
    source_config.write_text("dynamic_batching { max_queue_delay_microseconds: 9999 }\n")

    result = modelsvc.prepare_model(fake_spec, repo, trtexec="trtexec")

    assert not result.rebuilt
    assert "9999" in (repo / fake_spec.name / "config.pbtxt").read_text()


def test_missing_source_config_is_a_hard_error(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (modelsvc.TRITON_REPO / fake_spec.name / "config.pbtxt").unlink()

    with pytest.raises(FileNotFoundError, match="emit-config"):
        modelsvc.prepare_model(fake_spec, tmp_path / "repo", trtexec="trtexec")


def test_python_models_are_copied_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model Python backend không có gì để dựng — chỉ chép nguyên thư mục."""
    source_repo = tmp_path / "src"
    model = source_repo / "craneops_ccode_h"
    (model / "1").mkdir(parents=True)
    (model / "config.pbtxt").write_text('name: "craneops_ccode_h"\n')
    (model / "1" / "model.py").write_text("# noop\n")
    monkeypatch.setattr(modelsvc, "TRITON_REPO", source_repo)

    repo = tmp_path / "repo"
    installed = modelsvc.install_python_models(repo, names=("craneops_ccode_h",))

    assert installed == ["craneops_ccode_h"]
    assert (repo / "craneops_ccode_h" / "1" / "model.py").exists()
    assert (repo / "craneops_ccode_h" / "config.pbtxt").exists()


def test_python_models_replace_stale_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chép đè phải XOÁ file cũ, không để lẫn phiên bản trước."""
    source_repo = tmp_path / "src"
    (source_repo / "craneops_ccode_h" / "1").mkdir(parents=True)
    (source_repo / "craneops_ccode_h" / "1" / "model.py").write_text("moi\n")
    monkeypatch.setattr(modelsvc, "TRITON_REPO", source_repo)

    repo = tmp_path / "repo"
    (repo / "craneops_ccode_h" / "1").mkdir(parents=True)
    (repo / "craneops_ccode_h" / "1" / "rac_cu.py").write_text("cu\n")

    modelsvc.install_python_models(repo, names=("craneops_ccode_h",))

    assert not (repo / "craneops_ccode_h" / "1" / "rac_cu.py").exists()
    assert (repo / "craneops_ccode_h" / "1" / "model.py").read_text() == "moi\n"


def test_missing_python_model_dir_is_a_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modelsvc, "TRITON_REPO", tmp_path / "src")

    with pytest.raises(FileNotFoundError, match="Python backend"):
        modelsvc.install_python_models(tmp_path / "repo", names=("khong_ton_tai",))


def test_changing_build_flags_forces_a_rebuild(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Đổi FP16 -> FP32 KHÔNG làm file .t7 mới hơn, nên mtime không bắt được.

    Không có dấu build thì engine FP16 cũ sẽ được dùng lại vĩnh viễn và việc tắt FP16
    âm thầm vô tác dụng — model vẫn READY, kết quả vẫn "hợp lý", chỉ là sai.
    """
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])
    repo = tmp_path / "repo"
    _fake_engine(repo, fake_spec)  # dấu build ghi theo fake_spec (fp16=True mặc định)

    built: list[str] = []

    def record(_o: Path, _s: ModelSpec, plan: Path, **_k: object) -> None:
        built.append("x")
        plan.write_bytes(b"moi")

    monkeypatch.setattr(modelsvc, "_run_trtexec", record)
    scratch = tmp_path / "shm"
    scratch.mkdir()
    monkeypatch.setattr(modelsvc, "SCRATCH_DIR", scratch)
    # tmp_path nằm trên đĩa; các test này kiểm bất biến khác nên bỏ qua chốt tmpfs.
    # Bản thân chốt đó có test riêng bên dưới.
    monkeypatch.setattr(modelsvc, "_is_tmpfs", lambda _p: True)

    fp32 = replace(fake_spec, fp16=False)
    assert modelsvc.prepare_model(fp32, repo).rebuilt
    assert built == ["x"]

    # Lần hai với CÙNG cờ thì phải dùng lại.
    monkeypatch.setattr(
        modelsvc, "_run_trtexec", lambda *_a, **_k: pytest.fail("không được dựng lại")
    )
    assert not modelsvc.prepare_model(fp32, repo).rebuilt


def test_engine_without_build_stamp_is_rebuilt(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine từ bản cũ (chưa có dấu build) phải được dựng lại, không tin mù quáng."""
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])
    monkeypatch.setattr(modelsvc, "_run_trtexec", lambda *_a, **_k: None)
    scratch = tmp_path / "shm"
    scratch.mkdir()
    monkeypatch.setattr(modelsvc, "SCRATCH_DIR", scratch)
    # tmp_path nằm trên đĩa; các test này kiểm bất biến khác nên bỏ qua chốt tmpfs.
    # Bản thân chốt đó có test riêng bên dưới.
    monkeypatch.setattr(modelsvc, "_is_tmpfs", lambda _p: True)

    repo = tmp_path / "repo"
    _fake_engine(repo, fake_spec, stamp=False)

    assert modelsvc.prepare_model(fake_spec, repo).rebuilt


def test_tu_choi_chay_khi_scratch_khong_phai_tmpfs(
    fake_spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bất biến bảo mật của service này, được cưỡng chế bằng code chứ không bằng niềm tin.

    Compose từng khai ``/dev/shm`` trong mục ``volumes:`` thay vì ``tmpfs:``. Docker biến
    nó thành anonymous volume trên ext4, nên bản rõ ONNX được ghi xuống ĐĨA trong khi mọi
    tài liệu khẳng định ngược lại. Một sai sót một dòng trong compose không được phép âm
    thầm phá bất biến này.
    """
    monkeypatch.setattr(modelsvc, "check_health", lambda _b, _s: [])
    monkeypatch.setattr(
        modelsvc, "_run_trtexec", lambda *_a, **_k: pytest.fail("không được giải mã")
    )
    monkeypatch.setattr(modelsvc, "_is_tmpfs", lambda _p: False)

    with pytest.raises(RuntimeError, match="không phải tmpfs"):
        modelsvc.prepare_model(fake_spec, tmp_path / "repo")


def test_is_tmpfs_nhan_dien_dung() -> None:
    """Thư mục shm của Linux là tmpfs; thư mục gốc thì không."""
    assert modelsvc._is_tmpfs(Path("/dev") / "shm") is True
    assert modelsvc._is_tmpfs(Path("/")) is False
    assert modelsvc._is_tmpfs(Path("/khong-ton-tai")) is False
