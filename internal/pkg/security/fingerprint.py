"""Vân tay thiết bị — định danh phần cứng không đổi được.

Thay bộ định danh yếu cũ (`hostname`, `machine`, `processor`, `product_serial`), vốn:

* ``hostname`` — đổi bằng một lệnh
* ``machine`` — ``x86_64`` trên mọi máy
* ``processor`` — rỗng trên nhiều bản Linux, và giống nhau trên mọi máy cùng đời CPU
* ``product_serial`` — trên máy dev trả ``"System Serial Number"``, chuỗi giữ chỗ OEM

Bộ mới, đã đo trên máy thật (2026-08-29):

======================= ========================================= ===========================
Nguồn                   Ví dụ                                     Đặc tính
======================= ========================================= ===========================
``dmi_uuid``            ``a53bf5bc-fcbc-17e7-06a7-bcfce71706a6``   UUID thật trong DMI
``board_serial``        ``241247619300243``                       Serial bo mạch
======================= ========================================= ===========================

Cả hai đọc được **bên trong container mà không cần mount gì thêm** — ``/sys`` được chia sẻ
sẵn và tiến trình trong container là root.

⚠️ **UUID của GPU CỐ Ý không nằm trong vân tay.** Nó là định danh mạnh — khắc trong card,
không sửa được bằng phần mềm — nên bỏ đi là mất một chút sức ràng buộc. Nhưng nó phụ thuộc
vào **cách cấp GPU cho container**, không phải vào máy:

* ``ds_app`` **buộc** phải chạy với ``count: all``, vì pin bằng ``device_ids`` cấp CUDA
  compute nhưng không cấp node V4L2 của NVDEC ⇒ ``nvv4l2decoder`` treo ở ``PREROLLING``.
* Triton thì nên pin để giới hạn tài nguyên trên máy dùng chung.

Đo được trên máy dev 2 GPU: ``device=0`` cho digest ``de4a5ace…``, ``all`` cho
``e7c91d40…``. Tức hai service trên **cùng một máy** sẽ cần hai giấy phép khác nhau — một
lỗi vận hành gần như chắc chắn xảy ra, và khi xảy ra thì thông báo là "giấy phép không
thuộc thiết bị này", không hề gợi ý nguyên nhân.

Vân tay phải định danh **cái máy**, không phải định danh cách khởi chạy container. Xem
``docs/DESIGN_NOTES.md`` DN-014.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PLACEHOLDER_VALUES",
    "DeviceFingerprint",
    "collect",
]

PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "system serial number",
        "to be filled by o.e.m.",
        "default string",
        "not specified",
        "not applicable",
        "none",
        "0",
        "0123456789",
        "chassis serial number",
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
    }
)
"""Giá trị OEM để trống. Chúng giống nhau trên mọi máy cùng đời bo mạch, nên coi như
KHÔNG CÓ chứ không phải một định danh."""

_DMI = Path("/sys/class/dmi/id")


def _read_dmi(name: str) -> str:
    try:
        value = (_DMI / name).read_text().strip()
    except OSError:
        return ""
    return "" if value.strip().lower() in PLACEHOLDER_VALUES else value


@dataclass(frozen=True, slots=True)
class DeviceFingerprint:
    """Định danh của một thiết bị cụ thể."""

    sources: dict[str, str] = field(default_factory=dict)
    """``{nhãn: giá trị}``, chỉ chứa nguồn đọc được và không phải giữ chỗ."""

    missing: tuple[str, ...] = ()
    """Các nhãn không đọc được hoặc trả giá trị giữ chỗ."""

    @property
    def canonical(self) -> str:
        """Chuỗi chuẩn hoá đem đi băm. Sắp xếp theo nhãn để thứ tự không ảnh hưởng."""
        return "\n".join(f"{k}={self.sources[k]}" for k in sorted(self.sources))

    @property
    def digest(self) -> str:
        """SHA-256 của :attr:`canonical`, dạng hex."""
        return hashlib.sha256(self.canonical.encode()).hexdigest()

    @property
    def is_strong(self) -> bool:
        """Có đủ định danh **không đổi được** để ràng buộc thiết bị không.

        Cần ``dmi_uuid`` — thứ duy nhất còn lại không sửa được bằng phần mềm sau khi bỏ
        ``gpu`` (xem docstring module). ``board_serial`` không đủ một mình: nhiều bo mạch
        để trống nó, và khi đó ràng buộc thiết bị trở nên vô nghĩa.
        """
        return "dmi_uuid" in self.sources

    def describe(self) -> str:
        """Mô tả cho người đọc, che bớt giá trị — dùng khi báo lỗi."""
        lines = [f"  {k:<14} {self.sources[k][:12]}…" for k in sorted(self.sources)]
        if self.missing:
            lines.append(f"  (thiếu: {', '.join(self.missing)})")
        return "\n".join(lines)


def collect() -> DeviceFingerprint:
    """Thu thập vân tay của máy đang chạy."""
    found: dict[str, str] = {}
    missing: list[str] = []

    for label, value in (
        ("dmi_uuid", _read_dmi("product_uuid")),
        ("board_serial", _read_dmi("board_serial")),
    ):
        if value:
            found[label] = value
        else:
            missing.append(label)

    return DeviceFingerprint(sources=found, missing=tuple(missing))
