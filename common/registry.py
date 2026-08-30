"""Ba registry: rule, combinator, probe.

Một rule/combinator/probe được nhận diện bằng **một chuỗi trong config** (mã ``CCODE01``,
tên combinator ``majority_vote``, tên probe ``ccode_text``). Registry là chỗ duy nhất
biến chuỗi đó thành code, và làm việc đó một cách kiểm được lúc import:

* Trùng khoá ⇒ lỗi ngay, không phải "cái sau ghi đè cái trước" âm thầm.
* Tra khoá không tồn tại ⇒ lỗi có liệt kê khoá hợp lệ, không phải ``KeyError`` trần.
* ``services/ruled`` là **một** runner đọc registry, không phải một entrypoint chép tay
  cho mỗi rule.

Hệ quả thiết kế: thêm một rule là thêm **một file** trong ``internal/rules/`` cộng một
dòng trong ``rule_groups`` — không đụng vào file dùng chung nào, nên hai rule mới không
bao giờ đụng nhau ở cùng một chỗ.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterator
from typing import Generic, TypeVar

__all__ = [
    "DuplicateRegistration",
    "Registry",
    "UnknownKey",
    "combinator_registry",
    "discover",
    "probe_registry",
    "rule_registry",
]

T = TypeVar("T")


class DuplicateRegistration(RuntimeError):
    """Hai thứ khác nhau cùng đăng ký một khoá."""


class UnknownKey(KeyError):
    """Tra một khoá chưa được đăng ký."""

    def __str__(self) -> str:
        # KeyError.__str__ bọc thông điệp trong dấu nháy, làm nó khó đọc.
        return self.args[0] if self.args else ""


class Registry(Generic[T]):
    """Ánh xạ khoá → giá trị, fail-fast ở cả hai đầu.

    Args:
        kind: Tên loại, chỉ dùng cho thông báo lỗi (``"rule"``, ``"combinator"``, ``"probe"``).
        normalize_case: Nếu ``True``, khoá không phân biệt hoa thường (mã rule viết
            ``CCODE01`` trong config nhưng ``ccode01`` trong tên module).
    """

    def __init__(self, kind: str, *, normalize_case: bool = False) -> None:
        self._kind = kind
        self._normalize_case = normalize_case
        self._items: dict[str, T] = {}

    def _key(self, key: str) -> str:
        return key.upper() if self._normalize_case else key

    def register(self, key: str, value: T) -> T:
        """Đăng ký ``value`` dưới ``key``.

        Trả về chính ``value`` để dùng được như decorator.

        Raises:
            DuplicateRegistration: nếu khoá đã có và trỏ tới thứ khác.
        """
        k = self._key(key)
        # Kiểm bằng `in` chứ không phải `self._items.get(k) is not None`: bản đầu dùng
        # cách sau, nên đăng ký giá trị `None` rồi đăng ký đè lên bằng thứ khác sẽ lọt
        # âm thầm — đúng thứ registry này sinh ra để chặn.
        if k in self._items and self._items[k] is not value:
            raise DuplicateRegistration(
                f"{self._kind} {key!r} đã được đăng ký bởi {self._items[k]!r}; "
                f"không thể đăng ký lại cho {value!r}"
            )
        self._items[k] = value
        return value

    def get(self, key: str) -> T:
        """Lấy giá trị theo khoá.

        Raises:
            UnknownKey: kèm danh sách khoá hợp lệ — gõ sai mã rule trong config là lỗi
                rất dễ mắc, và thông báo trần sẽ tốn hàng giờ để lần ra.
        """
        k = self._key(key)
        try:
            return self._items[k]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "(chưa có gì được đăng ký)"
            raise UnknownKey(
                f"{self._kind} {key!r} chưa được đăng ký. Đã có: {known}. "
                f"Nếu mã đúng, kiểm tra module của nó đã được import chưa "
                f"(xem common.registry.discover)."
            ) from None

    def __contains__(self, key: str) -> bool:
        return self._key(key) in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def keys(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        return [(k, self._items[k]) for k in sorted(self._items)]

    def clear(self) -> None:
        """Chỉ dùng trong test — để mỗi test bắt đầu từ registry sạch."""
        self._items.clear()


def discover(package: str) -> list[str]:
    """Import các submodule **trực tiếp** của ``package`` để side-effect đăng ký chạy.

    Decorator ``@register_rule`` chỉ chạy khi module chứa nó được import. Gọi hàm này
    một lần lúc khởi động (``discover("internal.rules")``) thay vì bắt mỗi entrypoint
    phải nhớ import đủ.

    ⚠️ **Chỉ một tầng.** Rule đặt trong package con (``internal/rules/ccode/rule.py``) sẽ
    **không** được tìm thấy, và hậu quả là nó lặng lẽ không bao giờ đăng ký — đúng loại
    lỗi im lặng mà registry sinh ra để chặn. Nên rule phải là file phẳng ngay trong
    ``internal/rules/``; ``internal/rules/modules/`` chứa collaborator, không chứa rule,
    nên việc không đi sâu vào đó là cố ý.

    Returns:
        Tên các module đã import, theo thứ tự.
    """
    mod = importlib.import_module(package)
    paths = getattr(mod, "__path__", None)
    if paths is None:
        return []

    imported: list[str] = []
    for info in pkgutil.iter_modules(paths):
        if info.name.startswith("_"):
            continue
        name = f"{package}.{info.name}"
        importlib.import_module(name)
        imported.append(name)
    return imported


# Ba registry của hệ thống. Kiểu giá trị được nới lỏng ở đây để tránh import vòng
# (``internal.rules.base`` cần ``common.registry``, không thể ngược lại). Các facade có
# kiểu chặt nằm cạnh nơi định nghĩa: ``internal/rules/base.py``,
# ``internal/orchestration/combinators.py``, ``ds_app/src/probes/__init__.py``.

rule_registry: Registry[object] = Registry("rule", normalize_case=True)
"""Mã rule (``CCODE01``) → ``RuleSpec``. Dùng bởi ``services/ruled``."""

combinator_registry: Registry[object] = Registry("combinator")
"""Tên combinator (``majority_vote``) → lớp combinator. Dùng bởi ``orchestratord``."""

probe_registry: Registry[object] = Registry("probe")
"""Tên probe (``ccode_text``) → hàm probe. Dùng bởi ``ds_app`` để resolve theo config.

Đây là chỗ dễ bị thay bằng ``getattr(self, f"probe_{name}")``. Đừng: cách đó chỉ nổ khi
pipeline đã chạy tới pad tương ứng — giữa lúc 11 camera đang stream — và nổ bằng
``AttributeError`` không nói tên nào là hợp lệ. Registry bắt lỗi lúc dựng pipeline.
"""
