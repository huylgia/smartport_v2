from __future__ import annotations

import os
from pathlib import Path

import pytest

from internal.pkg.security import cipher
from internal.pkg.security.cipher import DecryptionError, decrypt_bytes, decrypt_file

ASSETS = Path(os.environ.get("CRANEOPS_ASSETS", "/ssd1/huylg/dnp_project/smartport/assets"))
REAL_MODEL = ASSETS / "camera-crane/det-truckItems/truckItemsDetetion_111124.t7"

HEADER_LEN = 16 + 12 + 16


def _encrypt(plaintext: bytes, password: str) -> bytes:
    """Dựng một blob .t7 hợp lệ để test vòng tròn mã hoá → giải mã."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.hashes import SHA512
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(
        algorithm=SHA512(), length=32, salt=salt, iterations=500_000, backend=default_backend()
    ).derive(password.encode())
    enc = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend()).encryptor()
    body = enc.update(plaintext) + enc.finalize()
    return salt + iv + enc.tag + body


# ---------------------------------------------------------------- vòng tròn


def test_roundtrip_with_explicit_password() -> None:
    payload = b"noi dung model gia lap" * 100
    blob = _encrypt(payload, "mat-khau-test")
    assert decrypt_bytes(blob, password="mat-khau-test") == payload  # pragma: allowlist secret


def test_wrong_password_is_rejected() -> None:
    """GCM có tag xác thực nên sai mật khẩu bị phát hiện, không ra rác im lặng."""
    blob = _encrypt(b"x" * 64, "dung")
    with pytest.raises(DecryptionError, match="sai mật khẩu"):
        decrypt_bytes(blob, password="sai")  # pragma: allowlist secret


def test_tampered_ciphertext_is_rejected() -> None:
    """Đây là lý do dùng GCM thay vì CBC: sửa một byte là lộ ra ngay."""
    blob = bytearray(_encrypt(b"y" * 64, "pw"))
    blob[-1] ^= 0xFF
    with pytest.raises(DecryptionError, match="đã bị sửa đổi"):
        decrypt_bytes(bytes(blob), password="pw")  # pragma: allowlist secret


@pytest.mark.parametrize("size", [0, 1, HEADER_LEN - 1, HEADER_LEN])
def test_short_input_is_rejected_with_a_useful_message(size: int) -> None:
    with pytest.raises(DecryptionError, match="quá ngắn"):
        decrypt_bytes(b"\x00" * size)


# ---------------------------------------------------------------- mật khẩu


def test_password_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mật khẩu chỉ đến từ biến môi trường."""
    payload = b"z" * 32
    blob = _encrypt(payload, "tu-env")
    monkeypatch.setenv("CRANEOPS_MODEL_PASSWORD", "tu-env")
    assert decrypt_bytes(blob) == payload


def test_explicit_password_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"w" * 32
    blob = _encrypt(payload, "tuong-minh")
    monkeypatch.setenv("CRANEOPS_MODEL_PASSWORD", "sai-be-bet")
    assert decrypt_bytes(blob, password="tuong-minh") == payload  # pragma: allowlist secret


@pytest.mark.parametrize("value", ["", None])
def test_missing_password_fails_loudly(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Thiếu mật khẩu phải NÉM LỖI, tuyệt đối không rơi về một giá trị mặc định.

    Một mật khẩu mặc định trong source là mật khẩu công khai. Test này khoá lại điều đó:
    ai thêm lại `or "..."` vào `_password()` sẽ làm đổ test, kể cả khi mọi thứ khác chạy.
    """
    if value is None:
        monkeypatch.delenv("CRANEOPS_MODEL_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("CRANEOPS_MODEL_PASSWORD", value)

    with pytest.raises(cipher.MissingPassword, match="CRANEOPS_MODEL_PASSWORD"):
        decrypt_bytes(b"\x00" * 100)


# ---------------------------------------------------------------- model thật


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="cần thư mục assets/")
@pytest.mark.skipif(
    not os.environ.get("CRANEOPS_MODEL_PASSWORD"),
    reason="cần CRANEOPS_MODEL_PASSWORD — mật khẩu không nằm trong source",
)
def test_decrypts_a_real_model_into_valid_onnx() -> None:
    """Chứng minh định dạng .t7 tương thích với kho model đang chạy.

    Không ghi bản rõ ra đĩa — giải mã và kiểm tra hoàn toàn trong bộ nhớ.
    """
    onnx = pytest.importorskip("onnx")

    raw = decrypt_file(REAL_MODEL)
    model = onnx.load_from_string(raw)
    onnx.checker.check_model(model)

    # Shape đã đo — chính là thứ tools/export_models.py khai trong SPECS.
    inputs = {i.name for i in model.graph.input}
    outputs = {o.name for o in model.graph.output}
    assert "image" in inputs
    assert {"tmp_16", "concat_8.tmp_0"} <= outputs


def test_encrypt_decrypt_round_trip() -> None:
    payload = b"noi dung model gia lap" * 500
    blob = cipher.encrypt_bytes(payload, password="pw")  # pragma: allowlist secret

    assert cipher.decrypt_bytes(blob, password="pw") == payload  # pragma: allowlist secret


def test_encrypt_uses_fresh_salt_and_iv_each_time() -> None:
    """Cùng bản rõ, cùng mật khẩu ⇒ bản mã phải KHÁC nhau.

    Dùng lại IV với GCM là hỏng hoàn toàn: hai bản mã cùng IV cho phép suy ra XOR của
    hai bản rõ mà không cần khoá.
    """
    payload = b"x" * 1000
    a = cipher.encrypt_bytes(payload, password="pw")  # pragma: allowlist secret
    b = cipher.encrypt_bytes(payload, password="pw")  # pragma: allowlist secret

    assert a != b
    assert a[:28] != b[:28], "salt+iv phải ngẫu nhiên mỗi lần"


def test_encrypted_blob_detects_tampering() -> None:
    raw = cipher.encrypt_bytes(b"noi dung that" * 100, password="pw")  # pragma: allowlist secret
    blob = bytearray(raw)
    blob[-1] ^= 0xFF  # lật một bit trong ciphertext

    with pytest.raises(cipher.DecryptionError):
        cipher.decrypt_bytes(bytes(blob), password="pw")  # pragma: allowlist secret


def test_encrypt_rejected_by_wrong_password() -> None:
    blob = cipher.encrypt_bytes(b"bi mat" * 200, password="dung")  # pragma: allowlist secret

    with pytest.raises(cipher.DecryptionError):
        cipher.decrypt_bytes(blob, password="sai")  # pragma: allowlist secret


def test_khong_ma_hoa_noi_dung_rong() -> None:
    """Nếu cho phép, hàm này tạo blob 44 byte mà ``decrypt_bytes`` từ chối là "quá ngắn"
    — tức mã hoá xong không giải mã lại được."""
    with pytest.raises(ValueError, match="rỗng"):
        cipher.encrypt_bytes(b"", password="pw")  # pragma: allowlist secret
