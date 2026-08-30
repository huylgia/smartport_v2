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
``gpu``                 ``GPU-b2bc31c4-3d57-6c15-1dba-a4e8b4...``  Khắc trong card, bất biến
======================= ========================================= ===========================

Cả ba đọc được **bên trong container mà không cần mount gì thêm** — ``/sys`` được chia sẻ
sẵn và tiến trình trong container là root.

⚠️ **Ràng buộc về GPU:** container chỉ thấy những GPU được cấp qua ``--gpus``. Cấp
``device=0`` và cấp ``all`` cho ra hai vân tay khác nhau. Khoá phải được cấp trong **đúng
cấu hình** sẽ chạy. Máy đích chỉ có một GPU nên chuyện này không phát sinh, nhưng máy dev
có hai — nên phải nhất quán.
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
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


def _gpu_uuids() -> list[str]:
    """UUID của các GPU NVIDIA mà tiến trình này nhìn thấy, đã sắp xếp.

    Dùng ``nvidia-smi`` thay vì NVML để không cần thêm dependency Python — nó luôn có mặt
    trong image có GPU.
    """
    nvidia_smi = "/usr/bin/nvidia-smi"
    if not Path(nvidia_smi).exists():
        return []
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        out = subprocess.run(  # noqa: S603
            [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout
        return sorted(
            line.strip()
            for line in out.splitlines()
            if line.strip() and line.strip().lower() not in PLACEHOLDER_VALUES
        )
    return []


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

        Cần ít nhất một trong ``dmi_uuid`` / ``gpu`` — hai thứ duy nhất không sửa được
        bằng phần mềm. ``board_serial`` mạnh nhưng một số bo mạch để trống.
        """
        return bool(self.sources.keys() & {"dmi_uuid", "gpu"})

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
        ("gpu", ",".join(_gpu_uuids())),
    ):
        if value:
            found[label] = value
        else:
            missing.append(label)

    return DeviceFingerprint(sources=found, missing=tuple(missing))
