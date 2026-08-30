"""Khoá bản quyền ràng buộc thiết bị, ký bằng Ed25519.

Cạm bẫy mà thiết kế này tránh — khoá dạng băm đối xứng::

    key = SHA512(product_key | hostname | machine | processor | serial)[:20]

Đó là một **vân tay**, không phải một **giấy phép**. Hàm sinh khoá nằm ngay trong phần mềm
được giao, nên bất kỳ ai có phần mềm cũng chạy được nó trên máy mới và tự cấp cho mình một
khoá hợp lệ. Đổi sang định danh phần cứng mạnh hơn **không sửa được điều này** — họ chỉ
việc sinh lại vân tay mới rồi tự ký.

Cách duy nhất chặn được việc nhân bản sang thiết bị khác là **mật mã bất đối xứng**:

* Bên cấp phép giữ **khoá riêng**, không bao giờ giao đi.
* Phần mềm nhúng **khoá công khai**, chỉ dùng để *xác minh*.
* Giấy phép = chữ ký Ed25519 lên (vân tay thiết bị + hạn dùng).

Khách hàng không thể tự cấp giấy phép cho máy mới vì không có khoá riêng. Chép giấy phép
sang máy khác cũng vô ích: vân tay không khớp.

Định dạng::

    CO2.<payload base64url>.<chữ ký base64url>

    payload = {"v":2, "dev":"<sha256 vân tay>", "exp":<epoch|null>,
               "iat":<epoch>, "note":"<ghi chú>"}

Tiền tố ``CO2.`` phân biệt ngay với chuỗi băm đối xứng nói trên (dạng
``XXXXXXXXXX-YYYYYYYYYY-NOEXP``). Dạng đó **không được chấp nhận** — nó không cung cấp
bảo vệ thật, và ``validate`` nói rõ điều này thay vì chỉ báo "sai định dạng".
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from internal.pkg.security.fingerprint import DeviceFingerprint, collect

__all__ = [
    "EMBEDDED_PUBLIC_KEY",
    "PREFIX",
    "LicenseError",
    "LicensePayload",
    "issue",
    "public_key_b64",
    "validate",
]

PREFIX = "CO2"
SCHEMA_VERSION = 2

ENV_PUBLIC_KEY = "CRANEOPS_LICENSE_PUBLIC_KEY"
"""Khoá công khai dạng base64url (32 byte thô). Ghi đè khoá nhúng — dùng cho test và cho
môi trường staging có bộ khoá riêng."""

EMBEDDED_PUBLIC_KEY = "N6d0Snsp432Ply19AB4eQNl2L8dEtGbtS2SOlv4a_BY"
"""Khoá công khai của bên cấp phép, base64url 32 byte.

Chỉ dùng để **xác minh** — không cấp được giấy phép từ nó. Khoá riêng tương ứng nằm ngoài
repo (mặc định ``~/craneops-license.key``, chmod 400) và **không bao giờ** được commit hay
đưa vào image.

Sinh cặp khoá: ``python -m tools.issue_license --new-keypair``.

Nếu để trống, :func:`validate` **từ chối mọi giấy phép** thay vì âm thầm cho qua — không
bao giờ có trạng thái "quên cấu hình mà vẫn chạy"."""


class LicenseError(Exception):
    """Giấy phép không hợp lệ, hết hạn, sai chữ ký, hoặc không thuộc thiết bị này."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class LicensePayload:
    device: str
    expires_at: int | None
    issued_at: int
    note: str = ""

    def to_json(self) -> bytes:
        # sort_keys + separators cố định: cùng payload luôn cho cùng byte, nếu không chữ ký
        # sẽ không lặp lại được.
        return json.dumps(
            {
                "v": SCHEMA_VERSION,
                "dev": self.device,
                "exp": self.expires_at,
                "iat": self.issued_at,
                "note": self.note,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> LicensePayload:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LicenseError(f"payload không phải JSON hợp lệ: {exc}") from exc
        if data.get("v") != SCHEMA_VERSION:
            raise LicenseError(
                f"phiên bản giấy phép {data.get('v')!r} không hỗ trợ (cần {SCHEMA_VERSION})"
            )
        return cls(
            device=str(data["dev"]),
            expires_at=data.get("exp"),
            issued_at=int(data.get("iat", 0)),
            note=str(data.get("note", "")),
        )


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _public_key() -> Ed25519PublicKey:
    raw_b64 = os.environ.get(ENV_PUBLIC_KEY) or EMBEDDED_PUBLIC_KEY
    if not raw_b64:
        raise LicenseError(
            "chưa cấu hình khoá công khai để xác minh giấy phép. Sinh cặp khoá bằng "
            "`python -m tools.issue_license --new-keypair`, đặt phần công khai vào "
            f"EMBEDDED_PUBLIC_KEY hoặc biến môi trường {ENV_PUBLIC_KEY}."
        )
    try:
        return Ed25519PublicKey.from_public_bytes(_b64d(raw_b64))
    except (ValueError, TypeError) as exc:
        raise LicenseError(f"khoá công khai không hợp lệ: {exc}") from exc


def public_key_b64(private_key: Ed25519PrivateKey) -> str:
    """Phần công khai của một khoá riêng, base64url — dán thẳng vào mã nguồn."""
    return _b64e(private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def issue(
    device_digest: str,
    private_key: Ed25519PrivateKey,
    *,
    expires_at: int | None = None,
    note: str = "",
    issued_at: int | None = None,
) -> str:
    """Cấp một giấy phép. **Chỉ chạy ở phía bên cấp phép** — cần khoá riêng.

    Args:
        device_digest: :attr:`DeviceFingerprint.digest` của máy đích, do khách hàng gửi lên.
        private_key: Khoá riêng của bên cấp phép.
        expires_at: Epoch giây; ``None`` là vô thời hạn.
        note: Ghi chú cho người đọc, ví dụ ``"GC03"``. Nằm trong phần được ký nên không sửa
            được sau khi cấp.
    """
    payload = LicensePayload(
        device=device_digest,
        expires_at=expires_at,
        issued_at=int(issued_at if issued_at is not None else time.time()),
        note=note,
    )
    body = _b64e(payload.to_json())
    return f"{PREFIX}.{body}.{_b64e(private_key.sign(body.encode()))}"


def validate(
    token: str,
    *,
    fingerprint: DeviceFingerprint | None = None,
    now: float | None = None,
    require_strong: bool = True,
) -> LicensePayload:
    """Xác minh giấy phép cho máy đang chạy.

    Thứ tự kiểm tra có chủ đích: **chữ ký trước tiên**. Payload chưa xác thực thì không tin
    được — kể cả trường hạn dùng nằm trong đó.

    Args:
        fingerprint: Ghi đè vân tay, dùng cho test.
        require_strong: Từ chối khi máy không có định danh không-đổi-được nào
            (``dmi_uuid`` hoặc ``gpu``). Tắt đi nghĩa là ràng buộc thiết bị vô nghĩa.

    Raises:
        LicenseError: sai định dạng, sai chữ ký, hết hạn, hoặc không thuộc thiết bị này.
    """
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        raise LicenseError(
            f"giấy phép sai định dạng: cần '{PREFIX}.<payload>.<chữ ký>'. "
            f"Chuỗi kiểu 'XXXXXXXXXX-YYYYYYYYYY-NOEXP' không được chấp nhận — nó chỉ là "
            f"vân tay tự sinh, không phải giấy phép có chữ ký."
        )
    _, body, sig_b64 = parts

    try:
        _public_key().verify(_b64d(sig_b64), body.encode())
    except InvalidSignature as exc:
        raise LicenseError(
            "chữ ký không hợp lệ — giấy phép đã bị sửa, hoặc được cấp bởi khoá riêng khác"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise LicenseError(f"chữ ký không giải mã được: {exc}") from exc

    payload = LicensePayload.from_json(_b64d(body))

    if payload.expires_at is not None:
        current = int(now if now is not None else time.time())
        if current > payload.expires_at:
            raise LicenseError(
                "giấy phép đã hết hạn",
                {"now": current, "expires_at": payload.expires_at},
            )

    device = fingerprint or collect()
    if require_strong and not device.is_strong:
        raise LicenseError(
            "máy này không có định danh phần cứng không-đổi-được nào (cần dmi_uuid hoặc "
            "gpu) — ràng buộc thiết bị sẽ vô nghĩa",
            {"missing": list(device.missing)},
        )

    if device.digest != payload.device:
        raise LicenseError(
            "giấy phép không thuộc thiết bị này",
            {
                "expected_device": payload.device[:16] + "...",
                "this_device": device.digest[:16] + "...",
                "hint": (
                    "Nếu đây đúng là máy được cấp phép, kiểm tra container có được cấp "
                    "ĐÚNG tập GPU như lúc cấp không — '--gpus device=0' và '--gpus all' "
                    "cho hai vân tay khác nhau."
                ),
                "fingerprint": device.describe(),
            },
        )

    return payload
