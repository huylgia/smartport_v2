"""Điền ``code`` vào config cẩu, và đối chiếu nó với mọi thứ khoá theo mã đó.

``camera_code`` là **một chuỗi duy nhất đi xuyên cả hệ**: ds_app đặt nó lên
``PerceptionMessage``, rule tra config theo nó, evidence đặt tên thư mục segment theo nó.
Lệch một ký tự ở bất kỳ đâu thì dữ liệu của một camera đi vào hư không — và im lặng, vì
"không có config cho mã này" trông giống hệt "camera này chưa có sự kiện nào".

Mã **sinh ra** từ ``crane_id`` + host + cổng của ``stream``, rồi **ghi vào** config để nó
hiện ngay cạnh camera nó thuộc về. Sinh-rồi-kiểm chứ không phải khai tay: nó đọc được
nhưng không trôi được — ``load_crane`` từ chối một ``code`` không khớp ``stream``.

    make codes         # điền `code` vào configs/cranes/*.yaml, rồi đối chiếu
    make codes-check   # CI: không sửa gì, chỉ báo lệch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.config import CraneConfig, load_crane  # noqa: E402
from common.rule_config import rule_config_path  # noqa: E402
from internal.rules.configs import RULE_CONFIGS  # noqa: E402

CONFIGS = REPO / "configs"

_ENTRY = re.compile(r"^(?P<indent>\s*- )\{(?P<body>.*)\}\s*$")
_FIELD = re.compile(r"^\s*(?P<key>[a-z_]+):\s*(?P<value>.*?)\s*$")

_ORDER = ("code", "stream", "desc")
"""Thứ tự trường trong một dòng camera. ``code`` trước vì đó là thứ người ta dò khi đối
chiếu với config rule; ``desc`` cuối vì nó dài nhất và không dùng để khớp gì."""


def _cranes() -> list[Path]:
    return sorted((CONFIGS / "cranes").glob("*.yaml"))


def _code_for(crane_id: str, stream: str) -> str:
    """Cùng công thức với ``CameraConfig._derive_code`` — nhưng chạy được trên text thô.

    Ở đây không nạp model được: file đang sửa có thể chưa hợp lệ (đó là lý do phải chạy
    lệnh này). Sau khi ghi, :func:`check` nạp lại bằng model thật để nghiệm thu.
    """
    host = stream.split("://", 1)[1].split("/")[0]
    ip, _, port = host.partition(":")
    parts = [crane_id, ip.replace(".", "_")]
    if port:
        parts.append(port)
    return "_".join(p for p in parts if p)


def _parse_entry(body: str) -> dict[str, str]:
    """Tách một flow mapping YAML một dòng thành dict, giữ nguyên chuỗi thô.

    Không dùng ``yaml.safe_load``: nó trả về giá trị đã ép kiểu, và ghi lại bằng
    ``yaml.dump`` sẽ thêm nháy quanh những chuỗi vốn không cần — file đang tối ưu cho người
    đọc, không cho máy.
    """
    out: dict[str, str] = {}
    for part in body.split(","):
        m = _FIELD.match(part)
        if m:
            out[m.group("key")] = m.group("value")
    return out


def synced_text(path: Path) -> str:
    """Nội dung file sau khi điền ``code`` cho mỗi camera. **Không ghi.**

    Trả text chứ không ghi thẳng để ``--check`` so được: "chạy sync không đổi gì" là phép
    kiểm duy nhất nói lên ``code`` trong file đã đúng. Bản trước ``--check`` bỏ qua hẳn
    bước sync và chỉ đối chiếu mã với rule — nên một camera thêm vào mà quên ``make codes``
    thì lọt cả CI lẫn pytest.

    Sửa theo dòng chứ không nạp-rồi-ghi-lại YAML: file này đầy chú thích giải thích **vì
    sao** từng ràng buộc tồn tại, và một vòng round-trip qua PyYAML sẽ xoá sạch chúng.
    """
    text = path.read_text(encoding="utf-8")
    crane_id = next(
        (ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("crane_id:")),
        "",
    )
    if not crane_id:
        raise SystemExit(f"❌ {path}: thiếu `crane_id`")

    out: list[str] = []
    for line in text.splitlines(keepends=True):
        m = _ENTRY.match(line.rstrip("\n"))
        if not m:
            out.append(line)
            continue
        fields = _parse_entry(m.group("body"))
        if "stream" not in fields:
            raise SystemExit(f"❌ {path}: camera thiếu `stream`:\n    {line.strip()}")
        fields["code"] = _code_for(crane_id, fields["stream"])
        rest = [k for k in fields if k not in _ORDER]
        body = ", ".join(f"{k}: {fields[k]}" for k in (*_ORDER, *rest) if k in fields)
        out.append(f"{m.group('indent')}{{{body}}}\n")
    return "".join(out)


def _rule_keys(crane_id: str) -> dict[str, set[str]]:
    """Mã camera mà từng config rule đang khoá theo."""
    found: dict[str, set[str]] = {}
    for spec in RULE_CONFIGS:
        p = rule_config_path(CONFIGS, crane_id, spec.code)
        if p.is_file():
            found[spec.code] = set(json.loads(p.read_text(encoding="utf-8")))
    return found


def check(crane: CraneConfig) -> list[str]:
    """Mã trong config cẩu phải khớp mã mà mọi service khác đang dùng.

    ``load_rule`` đã kiểm chiều "config rule không được có mã lạ". Ở đây kiểm chiều còn
    lại và không cần env: **file nào đang dùng mã nào**, để một lần đổi IP camera lộ ra hết
    những chỗ phải cập nhật thay vì lộ dần qua từng service lúc chạy.
    """
    problems: list[str] = []
    known = {c.code for c in crane.record_cameras}
    for rule_code, keys in _rule_keys(crane.crane_id).items():
        for stale in sorted(keys - known):
            problems.append(
                f"{crane.crane_id}/{rule_code}: mã {stale} không còn trong config cẩu.\n"
                f"   Camera đổi IP/cổng? Mã hiện tại: {sorted(known)}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="chỉ kiểm, không sửa file (CI)")
    args = ap.parse_args(argv)

    bad = 0
    for path in _cranes():
        rel = path.relative_to(REPO)
        want = synced_text(path)
        if want != path.read_text(encoding="utf-8"):
            if args.check:
                bad += 1
                print(
                    f"  ❌ {rel}: `code` lỗi thời hoặc thiếu. Chạy: make codes",
                    file=sys.stderr,
                )
                continue
            path.write_text(want, encoding="utf-8")
            print(f"  ✏️  {rel}: đã cập nhật `code`")

        # Nạp bằng model thật — đây là phép nghiệm thu, không phải phép kiểm thứ hai.
        crane = load_crane(path, env={})
        problems = check(crane)
        if problems:
            bad += len(problems)
            for p in problems:
                print(f"  ❌ {p}", file=sys.stderr)
        else:
            print(f"  ✅ {crane.crane_id}: {len(crane.record_cameras)} mã, khớp mọi rule")

    if bad:
        print(f"\n❌ {bad} chỗ cần sửa", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
