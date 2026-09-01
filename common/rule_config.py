"""Config của rule — tham số nghiệp vụ, khoá theo ``camera_code``.

Đây là tầng config **thứ ba**, tách hẳn khỏi hai tầng kia. Ranh giới theo đúng một câu
hỏi: *ai đọc nó, và đổi nó thì phải dựng lại cái gì?*

===============  ==================================  =====================  ==============
Tầng             Ở đâu                               Nội dung               Đổi thì
===============  ==================================  =====================  ==============
Triton           ``triton/repo/**/config.pbtxt``     hình dạng model,       restart Triton
                                                     batching, instance
ds_app           ``configs/cranes/<CẨU>.yaml``       đăng ký camera:        restart pipeline
                                                     URL→mã, vai trò, crop
**rule**         ``configs/rules/<CẨU>/<RULE>/``     ngưỡng, vùng, cửa sổ   **hot-reload**
===============  ==================================  =====================  ==============

Vì sao khoá theo ``camera_code`` chứ không theo khoá ngắn (``tcode1``): đó chính là chuỗi
đến trên :class:`~common.message.PerceptionMessage`. Rule tra config bằng đúng field nó
nhận được — không có bảng dịch nào ở giữa để lệch.

**Một cẩu là một "epic".** Mỗi cẩu có bộ rule của nó, mỗi rule có config riêng cho từng
camera của cẩu đó. Cây thư mục phản ánh đúng vậy::

    configs/rules/GC03/CRANE01/config.json      # khoá: camera_code
    configs/rules/GC03/CRANE01/schema.json      # sinh từ pydantic, không viết tay
    configs/rules/GC03/CRANE01/changelog.md     # đổi ngưỡng thì ghi lại vì sao

``config.json`` là ánh xạ phẳng, và **mỗi camera đúng một dòng**::

    {
      "GC03_113_160_225_15_1510": {"lane1_zone":[[0.44,0.0],…],"head_thresh":0.8,…},
      "GC03_113_160_225_15_1512": {"lane1_zone":[[0.56,0.0],…],"head_thresh":0.8,…}
    }

Một dòng một camera nên số camera được cấu hình **đếm được bằng mắt** — và bằng ``wc -l``.
Trải mỗi số ra một dòng thì một đa giác 26 đỉnh chiếm 78 dòng, file thành thứ không ai
cuộn hết, và thiếu hẳn một camera cũng không ai thấy.

⚠️ **Camera thiếu config là LỖI, không phải mặc định.** Đây là chỗ thiết kế tham khảo để
lọt: config là một dict, camera không có mặt trong đó thì rule đơn giản không xử lý nó — hệ
chạy, log sạch, và một camera lặng lẽ không sinh signal nào. :func:`load_rule` đòi đủ mọi
camera đúng vai trò, và từ chối mã lạ.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from common.config import CraneConfig
from common.enum import CameraRole

__all__ = ["RuleConfigError", "load_rule", "rule_config_path"]

T = TypeVar("T", bound=BaseModel)


class RuleConfigError(ValueError):
    """Config rule sai. Luôn kèm đường dẫn, mã rule và camera nào có vấn đề."""


def rule_config_path(root: Path, crane_id: str, rule_code: str) -> Path:
    """``configs/rules/<CẨU>/<RULE>/config.json``."""
    return root / "rules" / crane_id / rule_code / "config.json"


def load_rule(
    path: Path,
    model: type[T],
    *,
    crane: CraneConfig,
    roles: Sequence[CameraRole],
    rule_code: str = "",
) -> dict[str, T]:
    """Đọc config của một rule, trả ``{camera_code: config}``.

    Args:
        path: ``config.json`` của rule.
        model: pydantic model riêng của rule — nguồn sự thật cho schema và giá trị mặc định.
        crane: cấu hình cẩu, để biết camera nào tồn tại và vai trò gì.
        roles: các vai trò rule này tiêu thụ. Mọi camera thuộc các vai trò đó **phải** có
            mặt trong config.
        rule_code: chỉ để đưa vào thông báo lỗi.

    Raises:
        RuleConfigError: thiếu file, JSON hỏng, mã camera lạ, thiếu camera, hoặc nội dung
            không khớp ``model``.
    """
    label = f"{rule_code or path.parent.name} ({path})"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuleConfigError(f"{label}: không đọc được — {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuleConfigError(f"{label}: JSON hỏng — {exc}") from exc

    if not isinstance(raw, dict):
        raise RuleConfigError(f"{label}: nội dung phải là một ánh xạ, nhận {type(raw).__name__}")

    expected = {c.code: c for c in crane.record_cameras if c.role in roles}

    # Mã lạ TRƯỚC: nó là chẩn đoán cụ thể nhất. Một mã sai chính tả cũng làm camera thật
    # thiếu config, và báo "thiếu camera X" sẽ dẫn người đọc đi thêm một vòng.
    unknown = sorted(set(raw) - {c.code for c in crane.record_cameras})
    if unknown:
        raise RuleConfigError(
            f"{label}: mã camera không thuộc cẩu {crane.crane_id}: {unknown}\n"
            f"   Cẩu này có: {sorted(c.code for c in crane.record_cameras)}\n"
            f"   Mã camera suy từ URL — đổi IP/cổng của camera là đổi mã."
        )

    wrong_role = sorted(set(raw) - set(expected))
    if wrong_role:
        have = {c.code: c.role.value for c in crane.record_cameras}
        raise RuleConfigError(
            f"{label}: camera sai vai trò: "
            + ", ".join(f"{code} là {have[code]}" for code in wrong_role)
            + f"\n   Rule này chỉ nhận: {sorted(r.value for r in roles)}"
        )

    missing = sorted(set(expected) - set(raw))
    if missing:
        raise RuleConfigError(
            f"{label}: thiếu config cho {len(missing)} camera:\n"
            + "\n".join(f"    {code}  ({expected[code].key})" for code in missing)
            + "\n   Thiếu thì rule bỏ qua camera đó — hệ chạy, log sạch, và camera ấy "
            "không bao giờ sinh signal. Sinh khung mặc định: craneops-rules init "
            f"{crane.crane_id}"
        )

    out: dict[str, T] = {}
    for code, body in raw.items():
        try:
            out[code] = model.model_validate(body)
        except ValidationError as exc:
            raise RuleConfigError(f"{label}: camera {code} ({expected[code].key}) — {exc}") from exc
    return out
