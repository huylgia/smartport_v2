from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from internal.pkg.security import license as lic
from internal.pkg.security.fingerprint import PLACEHOLDER_VALUES, DeviceFingerprint

DEVICE_A = DeviceFingerprint(
    sources={
        "dmi_uuid": "a53bf5bc-fcbc-17e7-06a7-bcfce71706a6",
        "board_serial": "241247619300243",
    }
)
DEVICE_B = DeviceFingerprint(
    sources={
        "dmi_uuid": "ffffffff-0000-1111-2222-333333333333",
        "board_serial": "999999999999999",
    }
)


@pytest.fixture
def vendor_key(monkeypatch: pytest.MonkeyPatch) -> Ed25519PrivateKey:
    """Cặp khoá của bên cấp phép.

    Ghi đè hằng số nhúng bằng ``setattr`` chứ không phải biến môi trường — xem
    ``license._public_key``: một biến môi trường ghi đè được khoá này sẽ vô hiệu hoá toàn
    bộ cơ chế cấp phép.
    """
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(lic, "EMBEDDED_PUBLIC_KEY", lic.public_key_b64(key))
    return key


# ---------------------------------------------------------------- fingerprint


def test_fingerprint_digest_is_stable_and_order_independent() -> None:
    a = DeviceFingerprint(sources={"x": "1", "y": "2"})
    b = DeviceFingerprint(sources={"y": "2", "x": "1"})
    assert a.digest == b.digest


def test_different_hardware_gives_different_digest() -> None:
    assert DEVICE_A.digest != DEVICE_B.digest


def test_changing_any_single_source_changes_the_digest() -> None:
    for key in DEVICE_A.sources:
        altered = dict(DEVICE_A.sources)
        altered[key] = "khac-di"
        assert DeviceFingerprint(sources=altered).digest != DEVICE_A.digest


def test_is_strong_needs_dmi_uuid() -> None:
    """``dmi_uuid`` là định danh không-sửa-được-bằng-phần-mềm duy nhất còn lại."""
    assert DeviceFingerprint(sources={"dmi_uuid": "v"}).is_strong


def test_board_serial_alone_is_not_considered_strong() -> None:
    """board_serial bị nhiều bo mạch để trống, nên không đủ để dựa vào một mình."""
    assert not DeviceFingerprint(sources={"board_serial": "123"}).is_strong


def test_placeholder_list_covers_what_this_machine_reports() -> None:
    """Máy dev trả đúng "System Serial Number" cho product_serial."""
    assert "system serial number" in PLACEHOLDER_VALUES
    assert "00000000-0000-0000-0000-000000000000" in PLACEHOLDER_VALUES


def test_describe_masks_values() -> None:
    text = DEVICE_A.describe()
    assert "a53bf5bc-fcb" in text
    assert DEVICE_A.sources["dmi_uuid"] not in text  # bị cắt bớt


# ---------------------------------------------------------------- issue/validate


def test_roundtrip(vendor_key: Ed25519PrivateKey) -> None:
    token = lic.issue(DEVICE_A.digest, vendor_key, note="GC03")
    payload = lic.validate(token, fingerprint=DEVICE_A)
    assert payload.device == DEVICE_A.digest
    assert payload.note == "GC03"
    assert payload.expires_at is None


def test_token_has_the_v2_prefix(vendor_key: Ed25519PrivateKey) -> None:
    assert lic.issue(DEVICE_A.digest, vendor_key).startswith(f"{lic.PREFIX}.")


def test_license_for_another_device_is_rejected(vendor_key: Ed25519PrivateKey) -> None:
    """Mục tiêu chính: chép giấy phép sang máy khác phải vô dụng."""
    token = lic.issue(DEVICE_A.digest, vendor_key)
    with pytest.raises(lic.LicenseError, match="không thuộc thiết bị này"):
        lic.validate(token, fingerprint=DEVICE_B)


def test_customer_cannot_mint_a_license_with_a_different_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Điểm cốt lõi: có phần mềm mà không có khoá riêng thì không tự cấp phép được."""
    vendor = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()

    forged = lic.issue(DEVICE_B.digest, attacker, note="tu-cap")

    monkeypatch.setattr(lic, "EMBEDDED_PUBLIC_KEY", lic.public_key_b64(vendor))
    with pytest.raises(lic.LicenseError, match="chữ ký không hợp lệ"):
        lic.validate(forged, fingerprint=DEVICE_B)


def test_tampering_with_payload_breaks_the_signature(vendor_key: Ed25519PrivateKey) -> None:
    """Sửa vân tay trong payload để dùng cho máy khác — chữ ký phải phát hiện."""
    token = lic.issue(DEVICE_A.digest, vendor_key)
    prefix, _body, sig = token.split(".")
    forged_body = lic._b64e(
        lic.LicensePayload(device=DEVICE_B.digest, expires_at=None, issued_at=0).to_json()
    )
    with pytest.raises(lic.LicenseError, match="chữ ký không hợp lệ"):
        lic.validate(f"{prefix}.{forged_body}.{sig}", fingerprint=DEVICE_B)


def test_expired_license_is_rejected(vendor_key: Ed25519PrivateKey) -> None:
    token = lic.issue(DEVICE_A.digest, vendor_key, expires_at=1_000_000)
    with pytest.raises(lic.LicenseError, match="hết hạn"):
        lic.validate(token, fingerprint=DEVICE_A, now=1_000_001)


def test_license_valid_before_expiry(vendor_key: Ed25519PrivateKey) -> None:
    token = lic.issue(DEVICE_A.digest, vendor_key, expires_at=2_000_000)
    assert lic.validate(token, fingerprint=DEVICE_A, now=1_999_999).expires_at == 2_000_000


def test_signature_is_checked_before_expiry(vendor_key: Ed25519PrivateKey) -> None:
    """Payload chưa xác thực thì không tin được — kể cả trường hạn dùng trong đó."""
    attacker = Ed25519PrivateKey.generate()
    forged = lic.issue(DEVICE_A.digest, attacker, expires_at=int(time.time()) - 1)
    with pytest.raises(lic.LicenseError, match="chữ ký"):
        lic.validate(forged, fingerprint=DEVICE_A)


@pytest.mark.parametrize(
    "bad",
    [
        "khong-phai-token",
        "CO2.chi-co-hai-phan",
        "XX.abc.def",
        "4CF7EB5126-EA2393B5C9-NOEXP",  # dạng băm đối xứng
    ],
)
def test_malformed_tokens_are_rejected(bad: str, vendor_key: Ed25519PrivateKey) -> None:
    with pytest.raises(lic.LicenseError, match="sai định dạng"):
        lic.validate(bad, fingerprint=DEVICE_A)


def test_symmetric_hash_key_error_says_why(vendor_key: Ed25519PrivateKey) -> None:
    """Chuỗi băm đối xứng phải bị từ chối kèm **lý do**, không chỉ "sai định dạng".

    Ai dán nhầm chuỗi này vào cần biết ngay rằng nó không bao giờ hợp lệ, thay vì đi tìm
    lỗi gõ trong một chuỗi vốn đúng cú pháp của chính nó.
    """
    with pytest.raises(lic.LicenseError) as exc:
        lic.validate("4CF7EB5126-EA2393B5C9-NOEXP", fingerprint=DEVICE_A)
    assert "vân tay tự sinh" in exc.value.message


def test_weak_device_is_rejected(vendor_key: Ed25519PrivateKey) -> None:
    """Không có dmi_uuid ⇒ ràng buộc thiết bị vô nghĩa ⇒ từ chối."""
    weak = DeviceFingerprint(sources={"board_serial": "1"}, missing=("dmi_uuid",))
    token = lic.issue(weak.digest, vendor_key)
    with pytest.raises(lic.LicenseError, match="không-đổi-được"):
        lic.validate(token, fingerprint=weak)


def test_weak_device_accepted_when_explicitly_allowed(vendor_key: Ed25519PrivateKey) -> None:
    weak = DeviceFingerprint(sources={"board_serial": "1"}, missing=("dmi_uuid",))
    token = lic.issue(weak.digest, vendor_key)
    assert lic.validate(token, fingerprint=weak, require_strong=False).device == weak.digest


def test_missing_public_key_rejects_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chưa cấu hình khoá công khai phải là TỪ CHỐI, không phải cho qua."""
    monkeypatch.setattr(lic, "EMBEDDED_PUBLIC_KEY", "")
    key = Ed25519PrivateKey.generate()
    token = lic.issue(DEVICE_A.digest, key)
    with pytest.raises(lic.LicenseError, match="chưa cấu hình khoá công khai"):
        lic.validate(token, fingerprint=DEVICE_A)


def test_mismatch_error_hints_at_gpu_visibility(vendor_key: Ed25519PrivateKey) -> None:
    """Bẫy thực tế: --gpus device=0 và --gpus all cho hai vân tay khác nhau."""
    token = lic.issue(DEVICE_A.digest, vendor_key)
    with pytest.raises(lic.LicenseError) as exc:
        lic.validate(token, fingerprint=DEVICE_B)
    assert "GPU" in str(exc.value.details["hint"])


def test_note_cannot_be_edited_after_issue(vendor_key: Ed25519PrivateKey) -> None:
    """Ghi chú nằm trong phần được ký."""
    token = lic.issue(DEVICE_A.digest, vendor_key, note="GC03")
    prefix, _body, sig = token.split(".")
    tampered = lic._b64e(
        lic.LicensePayload(
            device=DEVICE_A.digest, expires_at=None, issued_at=0, note="GC99"
        ).to_json()
    )
    with pytest.raises(lic.LicenseError, match="chữ ký"):
        lic.validate(f"{prefix}.{tampered}.{sig}", fingerprint=DEVICE_A)


def test_no_environment_variable_can_override_the_embedded_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Không biến môi trường nào được phép thay khoá xác minh.

    Từng có ``CRANEOPS_LICENSE_PUBLIC_KEY`` với lý do "cho test và staging". Hậu quả: ai
    đặt được biến môi trường chỉ cần tự sinh cặp khoá, đặt phần công khai vào đó, rồi tự
    ký giấy phép cho bất kỳ máy nào — cơ chế cấp phép thành vô nghĩa.

    Test này quét **mọi** biến môi trường có tên gợi ý, để ai thêm lại một đường ghi đè sẽ
    làm đổ test thay vì âm thầm mở lại cửa.
    """
    vendor = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    monkeypatch.setattr(lic, "EMBEDDED_PUBLIC_KEY", lic.public_key_b64(vendor))

    forged = lic.issue(DEVICE_B.digest, attacker, note="tu-cap")
    for name in (
        "CRANEOPS_LICENSE_PUBLIC_KEY",
        "CRANEOPS_PUBLIC_KEY",
        "LICENSE_PUBLIC_KEY",
    ):
        monkeypatch.setenv(name, lic.public_key_b64(attacker))

    with pytest.raises(lic.LicenseError, match="chữ ký không hợp lệ"):
        lic.validate(forged, fingerprint=DEVICE_B)


def test_fingerprint_does_not_depend_on_gpu_visibility() -> None:
    """Vân tay phải định danh MÁY, không phải cách khởi chạy container.

    UUID của GPU từng nằm trong vân tay. Vấn đề: container chỉ thấy GPU được cấp qua
    ``--gpus``, mà ``ds_app`` buộc phải dùng ``count: all`` (pin ``device_ids`` không cấp
    node NVDEC) còn Triton thì nên pin. Hai service trên cùng một máy sẽ ra hai digest khác
    nhau, và lỗi hiện ra là "không thuộc thiết bị này" — không gợi ý gì. Xem DN-014.
    """
    import inspect

    from internal.pkg.security import fingerprint as fp

    src = inspect.getsource(fp)
    assert "nvidia-smi" not in src, "GPU đã quay lại vân tay — xem DN-014 trước khi thêm"
    assert "gpu" not in fp.collect().sources
