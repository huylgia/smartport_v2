from __future__ import annotations

import pytest

from common.registry import DuplicateRegistration, Registry, UnknownKey, discover


@pytest.fixture
def reg() -> Registry[str]:
    return Registry("widget")


def test_register_and_get(reg: Registry[str]) -> None:
    reg.register("a", "alpha")
    assert reg.get("a") == "alpha"


def test_register_returns_value_so_it_works_as_decorator(reg: Registry[str]) -> None:
    assert reg.register("a", "alpha") == "alpha"


def test_duplicate_key_with_different_value_raises(reg: Registry[str]) -> None:
    reg.register("a", "alpha")
    with pytest.raises(DuplicateRegistration, match="đã được đăng ký"):
        reg.register("a", "beta")


def test_duplicate_key_with_same_value_is_idempotent(reg: Registry[str]) -> None:
    """Import lại cùng một module (ví dụ qua hai đường import) không được là lỗi."""
    reg.register("a", "alpha")
    reg.register("a", "alpha")
    assert len(reg) == 1


def test_unknown_key_lists_available_keys(reg: Registry[str]) -> None:
    reg.register("alpha", "A")
    reg.register("beta", "B")
    with pytest.raises(UnknownKey) as exc:
        reg.get("gamma")
    msg = str(exc.value)
    assert "gamma" in msg
    assert "alpha, beta" in msg
    assert "discover" in msg  # gợi ý cách khắc phục thường gặp nhất


def test_unknown_key_on_empty_registry_is_explicit(reg: Registry[str]) -> None:
    with pytest.raises(UnknownKey, match="chưa có gì được đăng ký"):
        reg.get("anything")


def test_case_normalisation_off_by_default(reg: Registry[str]) -> None:
    reg.register("Alpha", "A")
    assert "Alpha" in reg
    assert "alpha" not in reg


def test_case_normalisation_when_enabled() -> None:
    """Mã rule viết hoa trong config (CCODE01) nhưng thường trong tên module (ccode01)."""
    reg: Registry[str] = Registry("rule", normalize_case=True)
    reg.register("ccode01", "R")
    assert reg.get("CCODE01") == "R"
    assert reg.get("CCode01") == "R"
    assert "CCODE01" in reg


def test_case_normalisation_detects_duplicates_across_case() -> None:
    reg: Registry[str] = Registry("rule", normalize_case=True)
    reg.register("CCODE01", "A")
    with pytest.raises(DuplicateRegistration):
        reg.register("ccode01", "B")


def test_keys_and_items_are_sorted(reg: Registry[str]) -> None:
    for k in ("gamma", "alpha", "beta"):
        reg.register(k, k.upper())
    assert reg.keys() == ["alpha", "beta", "gamma"]
    assert reg.items() == [("alpha", "ALPHA"), ("beta", "BETA"), ("gamma", "GAMMA")]
    assert list(reg) == ["alpha", "beta", "gamma"]


def test_len_and_clear(reg: Registry[str]) -> None:
    reg.register("a", "A")
    reg.register("b", "B")
    assert len(reg) == 2
    reg.clear()
    assert len(reg) == 0


def test_discover_imports_submodules() -> None:
    """internal.pkg có submodule thật (timebase) nên dùng làm mẫu kiểm chứng."""
    imported = discover("internal.pkg")
    assert "internal.pkg.timebase" in imported


def test_discover_on_module_without_path_returns_empty() -> None:
    """Module thường (không phải package) không có __path__ — không được nổ."""
    assert discover("internal.pkg.timebase") == []


def test_discover_skips_private_modules() -> None:
    assert not any(m.rsplit(".", 1)[-1].startswith("_") for m in discover("internal.pkg"))


def test_gia_tri_None_khong_bi_ghi_de_am_tham() -> None:
    """Bản đầu kiểm ``self._items.get(k) is not None`` nên ``None`` lọt qua cửa trùng khoá.

    Đăng ký ``None`` là chuyện lạ, nhưng một registry fail-fast mà có đúng một giá trị
    lọt được thì không còn là fail-fast.
    """
    reg: Registry[object] = Registry("thu")
    reg.register("a", None)

    with pytest.raises(DuplicateRegistration):
        reg.register("a", "thu khac")

    assert reg.get("a") is None


def test_dang_ky_lai_CUNG_gia_tri_van_duoc() -> None:
    """Import hai lần cùng một module không được coi là trùng."""
    reg: Registry[object] = Registry("thu")
    marker = object()
    reg.register("a", marker)
    reg.register("a", marker)

    assert reg.get("a") is marker
