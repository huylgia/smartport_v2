"""Giải mã model AES-256-GCM.

Định dạng file ``.t7`` là **cố định** — kho model hiện có đã mã hoá theo nó::

    [salt 16B][iv 12B][tag 16B][ciphertext ...]

Khoá suy từ mật khẩu bằng PBKDF2-HMAC-SHA512, 500 000 vòng.

Hai ràng buộc mà module này phải giữ:

1. **Mật khẩu KHÔNG nằm trong source, và không có giá trị dự phòng.** Chỉ đọc từ
   ``CRANEOPS_MODEL_PASSWORD``; thiếu nó thì ném :class:`MissingPassword` chứ không âm
   thầm dùng một giá trị mặc định. Xem :func:`_password` để biết vì sao "chỉ là giá trị
   dự phòng cho tiện" lại là một lỗ hổng vĩnh viễn.

2. **Không ghi bản rõ ra đĩa.** :func:`decrypt_file` trả về ``bytes``; nơi gọi tự quyết
   định. ``triton/modelsvc`` ghi vào tmpfs rồi xoá sau khi dựng xong engine — và **kiểm**
   rằng đích đến thật sự là tmpfs, xem ``docs/DESIGN_NOTES.md`` DN-004.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.hashes import SHA512
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

__all__ = [
    "DecryptionError",
    "MissingPassword",
    "decrypt_bytes",
    "decrypt_file",
    "encrypt_bytes",
]

_SALT_LEN = 16
_IV_LEN = 12
_TAG_LEN = 16
_HEADER_LEN = _SALT_LEN + _IV_LEN + _TAG_LEN
_KDF_ITERATIONS = 500_000
_KEY_LEN = 32

_ENV_VAR = "CRANEOPS_MODEL_PASSWORD"


class DecryptionError(Exception):
    """Giải mã thất bại: sai mật khẩu, file hỏng, hoặc bị sửa đổi."""


class MissingPassword(DecryptionError):
    """Chưa cấu hình mật khẩu giải mã model."""


def _password() -> str:
    """Mật khẩu giải mã, **chỉ** từ biến môi trường.

    ⚠️ **Không thêm giá trị mặc định vào đây.** Một mật khẩu mặc định trong source là mật
    khẩu công khai: ``strings`` trên binary PyInstaller moi ra được, và nếu file này từng
    được commit thì nó nằm vĩnh viễn trong lịch sử git kể cả sau khi xoá. Nối chuỗi để
    "giấu" (``"a" + "b" + ...``) không giấu được gì — nó chỉ khiến máy quét secret bỏ sót,
    tức làm tình hình tệ hơn.

    Thà hỏng lúc khởi động còn hơn chạy được bằng một bí mật ai cũng đọc được.
    """
    password = os.environ.get(_ENV_VAR)
    if not password:
        raise MissingPassword(
            f"chưa đặt {_ENV_VAR} — không có mật khẩu thì không giải mã được model. "
            f"Đặt nó trong build/.env.triton (xem build/.env.triton.example)."
        )
    return password


def _derive_key(salt: bytes, password: str) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=SHA512(),
        length=_KEY_LEN,
        salt=salt,
        iterations=_KDF_ITERATIONS,
        backend=default_backend(),
    )
    return kdf.derive(password.encode())


def encrypt_bytes(plaintext: bytes, *, password: str | None = None) -> bytes:
    """Mã hoá thành định dạng ``.t7``: ``salt(16) | iv(12) | tag(16) | ciphertext``.

    Dùng khi sinh model mới từ model cũ (gấp chuẩn hoá vào đồ thị, reparameterize…).
    Luôn ghi ra **file mới**, không bao giờ đè lên ``.t7`` gốc của khách hàng.

    ⚠️ Thứ tự ở đây quan trọng và đã từng sai một lần: ``encryptor.tag`` chỉ tồn tại
    SAU ``finalize()``. Viết ``salt + iv + enc.tag + enc.update(x) + enc.finalize()`` sẽ
    ném ``NotYetFinalized`` vì Python tính biểu thức từ trái sang phải. Phải tính thân
    bản mã trước, rồi mới đọc tag.
    """
    if not plaintext:
        # Không chặn ở đây thì hàm này tạo ra blob đúng 44 byte, và `decrypt_bytes` từ
        # chối nó là "file quá ngắn" — tức mã hoá xong không giải mã lại được. Một model
        # rỗng luôn là lỗi, nên chặn ở đầu vào thay vì để nó thành lỗi khó hiểu ở đầu ra.
        msg = "không mã hoá nội dung rỗng — một model rỗng luôn là lỗi ở nơi gọi"
        raise ValueError(msg)

    salt = os.urandom(_SALT_LEN)
    iv = os.urandom(_IV_LEN)
    key = _derive_key(salt, password or _password())
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend()).encryptor()

    body = encryptor.update(plaintext) + encryptor.finalize()
    return salt + iv + encryptor.tag + body


def decrypt_bytes(blob: bytes, *, password: str | None = None) -> bytes:
    """Giải mã nội dung một file ``.t7``.

    Args:
        blob: Toàn bộ nội dung file.
        password: Ghi đè mật khẩu; mặc định lấy từ ``CRANEOPS_MODEL_PASSWORD``.

    Raises:
        DecryptionError: file quá ngắn, sai mật khẩu, hoặc tag GCM không khớp (nghĩa là
            nội dung đã bị sửa đổi).
    """
    if len(blob) <= _HEADER_LEN:
        raise DecryptionError(
            f"file quá ngắn: {len(blob)} byte, cần > {_HEADER_LEN} "
            f"(salt {_SALT_LEN} + iv {_IV_LEN} + tag {_TAG_LEN})"
        )

    salt = blob[:_SALT_LEN]
    iv = blob[_SALT_LEN : _SALT_LEN + _IV_LEN]
    tag = blob[_SALT_LEN + _IV_LEN : _HEADER_LEN]
    ciphertext = blob[_HEADER_LEN:]

    key = _derive_key(salt, password or _password())
    decryptor = Cipher(
        algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
    ).decryptor()
    try:
        return decryptor.update(ciphertext) + decryptor.finalize()
    except Exception as exc:  # GCM tag không khớp
        raise DecryptionError(
            f"giải mã thất bại — sai mật khẩu (đặt {_ENV_VAR}?) hoặc file đã bị sửa đổi: {exc}"
        ) from exc


def decrypt_file(path: str | Path, *, password: str | None = None) -> bytes:
    """Giải mã một file ``.t7`` và trả về bản rõ.

    Cố ý **không** ghi ra đĩa: nơi gọi tự quyết định vòng đời của bản rõ.

    Raises:
        DecryptionError: xem :func:`decrypt_bytes`.
        OSError: không đọc được file.
    """
    return decrypt_bytes(Path(path).read_bytes(), password=password)
