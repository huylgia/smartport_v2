"""Bộ lệnh ``deploy/craneops`` — kiểm dưới **terminal thật**, không chỉ qua pipe.

Vì sao phải có pty: ``print_commands`` chèn mã màu khi stdout là terminal. Một lỗi thật đã
lọt vì điều đó — ``craneops`` hỏi service "có lệnh này không" bằng cách khớp mẫu trên bảng
lệnh *dành cho người đọc*, và ``ESC[1mstatus`` làm ``\\bstatus\\b`` không khớp. Hậu quả:
``craneops status`` bỏ qua **mọi** service và thoát khác 0 — nhưng chỉ khi chạy trong
terminal. Qua pipe thì im lặng chạy đúng, nên mọi lần thử bằng công cụ tự động đều xanh.

Nguyên tắc rút ra: **đầu ra cho người đọc không được làm đầu vào cho máy đọc.** Test này
neo cả hai vế — hai bản phải liệt kê cùng một tập lệnh, và bản cho máy phải không có màu.
"""

from __future__ import annotations

import os
import pty
import re
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
SERVICES = ("triton", "ds")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _core_commands() -> list[str]:
    """Hợp đồng lõi, đọc từ ``craneops-lib.sh`` — không chép lại danh sách vào test."""
    text = (DEPLOY / "craneops-lib.sh").read_text(encoding="utf-8")
    match = re.search(r"^CORE_COMMANDS=\(([^)]*)\)", text, re.M)
    assert match, "không tìm thấy CORE_COMMANDS trong craneops-lib.sh"
    return match.group(1).split()


def _run(argv: list[str], *, tty: bool) -> str:
    """Chạy một lệnh, ép stdout là terminal hay pipe tuỳ ``tty``."""
    if not tty:
        # S603 tắt ở cả ba chỗ trong file: argv luôn là đường dẫn cố định tới deploy/,
        # dựng từ hằng số trong chính test — không có đầu vào từ ngoài.
        return subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout  # noqa: S603

    # `script` không dùng được ở đây: nó nuốt mã thoát. Cấp pty trực tiếp.
    primary, secondary = pty.openpty()
    proc = subprocess.Popen(argv, stdout=secondary, stderr=secondary, close_fds=True)  # noqa: S603
    os.close(secondary)
    chunks = []
    try:
        while True:
            try:
                data = os.read(primary, 65536)
            except OSError:  # pty đóng khi tiến trình con thoát
                break
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(primary)
        proc.wait(timeout=30)
    return b"".join(chunks).decode("utf-8", "replace").replace("\r\n", "\n")


def _declared(script: str) -> set[str]:
    """Tập lệnh mà script tự khai trong các dòng ``#:`` — nguồn sự thật."""
    text = (DEPLOY / script).read_text(encoding="utf-8")
    return {line[2:].split("|")[0].strip() for line in text.splitlines() if line.startswith("#:")}


@pytest.mark.parametrize("script", ["craneops", "craneops-triton", "craneops-ds"])
def test_help_lists_every_declared_command(script: str) -> None:
    """Bảng help phải phủ đúng tập lệnh đã khai — không thiếu, không thừa."""
    shown = _run([str(DEPLOY / script)], tty=False)
    for cmd in _declared(script):
        assert re.search(rf"^\s+{re.escape(cmd)}\s", shown, re.M), f"{script}: thiếu {cmd}"


@pytest.mark.parametrize("service", SERVICES)
def test_every_service_implements_the_core_contract(service: str) -> None:
    """Mọi service phải cài đủ lệnh lõi — đó là thứ làm `craneops <lệnh>` có nghĩa.

    Không có test này thì thêm một service mới mà quên `status` sẽ chỉ lộ ra lúc vận hành,
    và lộ dưới dạng khó hiểu nhất: `craneops status` dừng giữa chừng ở service đó.
    """
    script = f"craneops-{service}"
    missing = sorted(set(_core_commands()) - _declared(script))
    assert not missing, f"{script} thiếu lệnh lõi: {missing}"

    # Khai trong bảng `#:` mà quên nhánh `case` thì lệnh vẫn "có" nhưng chạy ra lỗi cú pháp.
    body = (DEPLOY / script).read_text(encoding="utf-8")
    for cmd in _core_commands():
        assert re.search(rf"^cmd_{cmd}\(\)", body, re.M), (
            f"{script}: khai {cmd} nhưng thiếu cmd_{cmd}()"
        )


def test_dispatcher_fans_out_exactly_the_core_contract() -> None:
    """`craneops` phải fan-out ĐÚNG bộ lõi — không thiếu, và không hứa thứ nó không có."""
    fanned = _declared("craneops") - {"services", "install", "uninstall"}
    assert fanned == set(_core_commands()), (
        f"craneops fan-out {sorted(fanned)} nhưng hợp đồng lõi là {sorted(_core_commands())}"
    )


@pytest.mark.parametrize(("service", "command"), [("triton", "status"), ("ds", "doctor")])
def test_dispatcher_does_not_skip_a_command_the_service_declares(
    service: str, command: str
) -> None:
    """Hồi quy: dưới TTY, `craneops <lệnh>` KHÔNG được bỏ qua service có khai lệnh đó.

    ⚠️ Phải chạy dispatcher THẬT, không đọc lại bảng lệnh: lỗi nằm trong ``supports()``, mà
    ``craneops services`` không gọi hàm đó. Bản đầu của test này đọc `services` và vẫn xanh
    khi cố ý trả lại đúng đoạn code hỏng — tức nó không chứng minh được gì.

    Chỉ khẳng định về **dòng tiêu đề**, thứ được in TRƯỚC khi gọi service. Nhờ vậy phép
    kiểm không phụ thuộc Docker hay Triton đang chạy: `supports()` sai thì thấy "bỏ qua",
    đúng thì thấy tên lệnh — bất kể sau đó chạy được hay không.
    """
    assert command in _declared(f"craneops-{service}"), "tiền đề của test đã đổi"

    out = _ANSI.sub("", _run([str(DEPLOY / "craneops"), command], tty=True))
    assert f"{service}: bỏ qua" not in out, (
        f"dưới TTY, dispatcher bỏ qua {service} dù nó có khai lệnh {command!r}:\n{out}"
    )
    assert f"{service}: {command}" in out


def test_machine_readable_listing_has_no_colour_under_a_tty() -> None:
    """``list_commands`` là bản cho MÁY đọc: một lệnh mỗi dòng, không mã màu.

    Neo cả hai tính chất vì cả hai đều đã hỏng một lần — mã màu làm khớp mẫu trượt, và
    ``tr -d '[:space:]'`` từng xoá luôn ký tự xuống dòng, dồn mọi lệnh vào một dòng.
    """
    for service in SERVICES:
        out = _run(
            [
                "bash",
                "-c",
                f'source "{DEPLOY}/craneops-lib.sh"; list_commands "{DEPLOY}/craneops-{service}"',
            ],
            tty=True,
        )
        assert "\x1b" not in out, f"{service}: bản cho máy đọc có mã màu"
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        assert set(lines) == _declared(f"craneops-{service}"), (
            f"{service}: mỗi lệnh phải nằm trên MỘT dòng riêng, thấy {lines}"
        )


def test_unknown_command_fails_loudly() -> None:
    """Gõ sai lệnh phải thoát khác 0 — im lặng không làm gì là cách hỏng tệ nhất."""
    for script in ("craneops", "craneops-triton", "craneops-ds"):
        done = subprocess.run(  # noqa: S603
            [str(DEPLOY / script), "khong-ton-tai"], capture_output=True, text=True, timeout=30
        )
        assert done.returncode != 0, f"{script}: lệnh sai mà vẫn thoát 0"
