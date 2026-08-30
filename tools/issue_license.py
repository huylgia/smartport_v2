"""Cấp giấy phép — **chỉ chạy ở phía bên cấp phép**.

Khoá riêng không bao giờ được đưa vào repo hay giao cho khách hàng — đó là toàn bộ giá
trị của cơ chế. Hàm *sinh* khoá nằm ở đây; phần mềm giao đi chỉ có hàm *xác minh* và khoá
công khai (``internal/pkg/security/license.py``).

Quy trình::

    # 1. Một lần duy nhất — sinh cặp khoá, cất phần riêng thật kỹ
    python -m tools.issue_license --new-keypair --out-private ~/craneops-license.key

    # 2. Trên MÁY ĐÍCH, trong ĐÚNG container sẽ chạy — lấy vân tay
    python -m tools.issue_license --fingerprint

    # 3. Bên cấp phép — ký cho vân tay đó
    python -m tools.issue_license --issue <digest> \\
        --private ~/craneops-license.key --note GC03 --days 365
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from internal.pkg.security.fingerprint import collect
from internal.pkg.security.license import issue, public_key_b64


def _new_keypair(out_private: Path) -> int:
    if out_private.exists():
        print(f"❌ {out_private} đã tồn tại — không ghi đè khoá riêng", file=sys.stderr)
        return 1

    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())

    out_private.parent.mkdir(parents=True, exist_ok=True)
    out_private.write_bytes(raw)
    out_private.chmod(0o400)

    print(f"✅ khoá riêng -> {out_private} (chmod 400)")
    print("\nDán dòng dưới vào internal/pkg/license.py:\n")
    print(f'EMBEDDED_PUBLIC_KEY = "{public_key_b64(private)}"')
    print(
        "\n⚠️  Sao lưu khoá riêng ở nơi an toàn. Mất nó là không cấp được giấy phép mới; "
        "lộ nó là bất kỳ ai cũng cấp được."
    )
    return 0


def _show_fingerprint() -> int:
    fp = collect()
    print("Vân tay thiết bị:")
    print(fp.describe())
    print(f"\ndigest: {fp.digest}")
    if not fp.is_strong:
        print(
            "\n❌ Máy này KHÔNG có định danh không-đổi-được (dmi_uuid hoặc gpu).\n"
            "   Ràng buộc thiết bị sẽ vô nghĩa. Nếu đang chạy trong container, kiểm tra\n"
            "   nó có được cấp GPU (--gpus) và đọc được /sys/class/dmi không.",
            file=sys.stderr,
        )
        return 1
    print("\nGửi digest này cho bên cấp phép.")
    return 0


def _issue(args: argparse.Namespace) -> int:
    if not args.private.exists():
        print(f"❌ không thấy khoá riêng: {args.private}", file=sys.stderr)
        return 1
    private = Ed25519PrivateKey.from_private_bytes(args.private.read_bytes())

    expires = int(time.time() + args.days * 86400) if args.days else None
    token = issue(args.issue, private, expires_at=expires, note=args.note)

    print(token)
    print(
        f"\n  thiết bị : {args.issue[:16]}...\n"
        f"  ghi chú  : {args.note or '(không)'}\n"
        f"  hạn dùng : {time.strftime('%Y-%m-%d', time.localtime(expires)) if expires else 'vô thời hạn'}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-keypair", action="store_true", help="sinh cặp khoá mới")
    ap.add_argument("--fingerprint", action="store_true", help="in vân tay máy đang chạy")
    ap.add_argument("--issue", metavar="DIGEST", help="cấp giấy phép cho một vân tay")
    ap.add_argument(
        "--private",
        type=Path,
        default=Path.home() / "craneops-license.key",
        help="đường dẫn khoá riêng",
    )
    ap.add_argument("--out-private", type=Path, help="nơi ghi khoá riêng mới")
    ap.add_argument("--note", default="", help="ghi chú, ví dụ GC03")
    ap.add_argument("--days", type=int, default=0, help="số ngày hiệu lực; 0 = vô thời hạn")
    args = ap.parse_args(argv)

    if args.new_keypair:
        return _new_keypair(args.out_private or args.private)
    if args.fingerprint:
        return _show_fingerprint()
    if args.issue:
        return _issue(args)

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
