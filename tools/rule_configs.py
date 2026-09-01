"""Dựng và kiểm config rule cho một cẩu.

Một cẩu là một "epic": nó có bộ rule của nó, và mỗi rule có config riêng cho từng camera
của cẩu đó. Với 8 rule cho ~10 camera thì viết tay là chép sai mã camera, và mã sai làm rule
**im lặng bỏ qua** camera đó — hệ chạy, log sạch, không signal nào.

Nên khung sinh từ ``configs/cranes/<CẨU>.yaml``: camera nào có vai trò rule tiêu thụ thì
tự có một khối, điền sẵn giá trị mặc định của pydantic model. Người chỉ còn sửa những thứ
phải đo trên ảnh thật — chủ yếu là vùng làn (``lane1_zone``…).

    python -m tools.rule_configs init GC04      # dựng khung, KHÔNG đè file đã có
    python -m tools.rule_configs schema         # sinh lại mọi schema.json
    python -m tools.rule_configs check          # CI: config khớp model và khớp cẩu chưa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.config import load_crane  # noqa: E402
from common.rule_config import RuleConfigError, load_rule, rule_config_path  # noqa: E402
from internal.rules.configs import RULE_CONFIGS, RuleSpec  # noqa: E402

CONFIGS = REPO / "configs"


def _cranes() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "cranes").glob("*.yaml"))


def _write_json(path: Path, payload: object) -> None:
    """Ghi JSON thường — dùng cho ``schema.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_per_camera(path: Path, cameras: dict[str, object]) -> None:
    """Ghi config rule: ánh xạ phẳng, **mỗi camera đúng một dòng**.

    Số camera được cấu hình đếm được bằng mắt và bằng ``wc -l``. Trải mỗi số ra một dòng
    thì một đa giác 26 đỉnh chiếm 78 dòng — file thành thứ không ai cuộn hết, và thiếu hẳn
    một camera cũng không ai thấy.
    """
    rows = ",\n".join(
        f"  {json.dumps(code)}: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}"
        for code, body in cameras.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\n" + rows + "\n}\n", encoding="utf-8")


def cmd_init(crane_id: str) -> int:
    """Dựng khung config cho mọi rule của một cẩu. **Không đè** file đã có."""
    try:
        crane = load_crane(CONFIGS / "cranes" / f"{crane_id}.yaml")
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    made = kept = 0
    for spec in RULE_CONFIGS:
        path = rule_config_path(CONFIGS, crane_id, spec.code)
        cams = [c for c in crane.record_cameras if c.role in spec.roles]
        if not cams:
            print(
                f"  —  {spec.code:<10} cẩu này không có camera vai trò "
                f"{sorted(r.value for r in spec.roles)}"
            )
            continue
        if path.exists():
            kept += 1
            print(f"  =  {spec.code:<10} đã có, giữ nguyên ({path.relative_to(REPO)})")
            continue

        # Mặc định lấy từ chính model — không gõ lại con số nào ở đây.
        default = spec.config_model().model_dump(mode="json")
        write_per_camera(path, {c.code: default for c in cams})
        _write_schema(spec, crane_id)
        _ensure_changelog(path.parent, spec.code, crane_id)
        made += 1
        print(f"  +  {spec.code:<10} {len(cams)} camera → {path.relative_to(REPO)}")

    print(f"\n{made} rule mới, {kept} giữ nguyên.")
    if made:
        print(
            "Còn phải điền tay: vùng làn (`lane1_zone`…) — phải đo trên ảnh thật của từng camera."
        )
    return 0


def _write_schema(spec: RuleSpec, crane_id: str) -> None:
    _write_json(
        rule_config_path(CONFIGS, crane_id, spec.code).with_name("schema.json"),
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"{spec.code} — config theo camera_code",
            "description": (
                "Sinh từ internal/rules/configs.py — đừng sửa tay. "
                "Khoá là camera_code (xem configs/cranes/*.yaml); mỗi camera một dòng."
            ),
            "type": "object",
            "additionalProperties": spec.config_model.model_json_schema(),
        },
    )


def _ensure_changelog(folder: Path, code: str, crane_id: str) -> None:
    p = folder / "changelog.md"
    if p.exists():
        return
    p.write_text(
        f"# {code} · {crane_id} — nhật ký thay đổi config\n\n"
        "Mỗi lần đổi một ngưỡng hay một vùng thì thêm một dòng: **đổi gì, vì sao, đo trên\n"
        "dữ liệu nào**. Một ngưỡng không có lý do là một ngưỡng không ai dám sửa lại.\n\n"
        "| Ngày | Camera | Đổi | Vì sao |\n|---|---|---|---|\n",
        encoding="utf-8",
    )


def cmd_schema() -> int:
    for crane_id in _cranes():
        for spec in RULE_CONFIGS:
            if rule_config_path(CONFIGS, crane_id, spec.code).exists():
                _write_schema(spec, crane_id)
    print(f"✅ sinh lại schema cho {len(_cranes())} cẩu")
    return 0


def cmd_check() -> int:
    """CI: mọi config.json phải khớp model **và** khớp cấu hình cẩu.

    Bắt được thứ mà chỉ đọc JSON không bắt được: mã camera lạ, và camera đúng vai trò mà
    thiếu config — cái sau là lỗi im lặng nhất trong cả hệ.
    """
    bad = 0
    for crane_id in _cranes():
        crane = load_crane(CONFIGS / "cranes" / f"{crane_id}.yaml")
        for spec in RULE_CONFIGS:
            path = rule_config_path(CONFIGS, crane_id, spec.code)
            if not path.exists():
                continue
            try:
                cfg = load_rule(
                    path, spec.config_model, crane=crane, roles=spec.roles, rule_code=spec.code
                )
                print(f"  ✅ {crane_id}/{spec.code:<10} {len(cfg)} camera")
            except RuleConfigError as exc:
                print(f"  ❌ {exc}", file=sys.stderr)
                bad += 1
    if bad:
        print(f"\n❌ {bad} config rule không hợp lệ", file=sys.stderr)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="dựng khung config rule cho một cẩu")
    init.add_argument("crane_id")
    sub.add_parser("schema", help="sinh lại schema.json từ pydantic model")
    sub.add_parser("check", help="CI: validate mọi config rule")
    args = ap.parse_args(argv)

    if args.cmd == "init":
        return cmd_init(args.crane_id)
    if args.cmd == "schema":
        return cmd_schema()
    return cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
