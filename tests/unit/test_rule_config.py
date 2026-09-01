"""Config rule: khoá theo ``camera_code``, và từ chối mọi cách nó lệch khỏi cấu hình cẩu."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from common.config import CraneConfig, load_crane
from common.enum import CameraRole, Lane
from common.rule_config import RuleConfigError, load_rule, rule_config_path
from internal.pkg.geometry import Anchor
from internal.rules.configs import RULE_CONFIGS, CCode01Config, Crane01Config, TCode01Config
from tests.conftest import GC03

CONFIGS = GC03.parent.parent
ENV: dict[str, str] = {}


@pytest.fixture
def crane() -> CraneConfig:
    return load_crane(GC03, env=ENV)


def _write(tmp_path: Path, cameras: dict[str, Any]) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cameras), encoding="utf-8")
    return p


# ---------------------------------------------------------------- file thật


@pytest.mark.parametrize("spec", RULE_CONFIGS, ids=lambda s: s.code)
def test_shipped_rule_configs_load(spec: Any, crane: CraneConfig) -> None:
    """Config rule trong repo phải load được — nếu không, CI sạch mà deploy chết."""
    path = rule_config_path(CONFIGS, "GC03", spec.code)
    cfg = load_rule(path, spec.config_model, crane=crane, roles=spec.roles, rule_code=spec.code)
    assert cfg
    assert set(cfg) == {c.code for c in crane.record_cameras if c.role in spec.roles}


def test_lane_zones_live_in_rule_config_not_in_the_pipeline_config() -> None:
    """Vùng lane là tham số nghiệp vụ: nó **không** được nằm trong config của ds_app.

    Để nó ở đó thì mỗi lần chỉnh một đa giác là một lần dựng lại pipeline, tức cả 10 camera
    ngừng ghi hình — trong khi thứ vừa đổi không ảnh hưởng gì tới việc thu hình.
    """
    assert "lane1_zone" not in GC03.read_text(encoding="utf-8"), "vùng làn quay lại config ds_app"

    doc = json.loads(rule_config_path(CONFIGS, "GC03", "CRANE01").read_text(encoding="utf-8"))
    assert all("lane1_zone" in body for body in doc.values())


def test_ocr_roi_stays_whole_in_the_ds_app_config() -> None:
    """``ocr_rois`` KHÔNG tách sang rule — probe của ds_app phải điền ``lane``/``cont_dim``.

    Hợp đồng message quyết định chuyện này: :class:`OcrResult` mang sẵn hai trường đó, nên
    tách chúng sang tầng rule thì ds_app hoặc phải đọc ngược config của rule, hoặc không
    điền nổi message.
    """
    from common.config import OcrRoi
    from common.message import OcrResult

    assert {"lane", "cont_dim"} <= set(OcrRoi.model_fields)
    assert {"lane", "cont_dim"} <= set(OcrResult.model_fields)
    assert "ocr_threshold" not in OcrRoi.model_fields, "ngưỡng là tham số của rule"
    assert "ocr_threshold" in CCode01Config.model_fields


# ---------------------------------------------------------------- lỗi im lặng


def test_missing_camera_is_an_error_not_a_default(tmp_path: Path, crane: CraneConfig) -> None:
    """⚠️ Lỗi im lặng nhất trong cả hệ: camera đúng vai trò nhưng không có trong config.

    Config là một dict; camera vắng mặt thì rule đơn giản không xử lý nó. Hệ chạy, log
    sạch, và camera ấy **không bao giờ** sinh signal. Không có gì chỉ về nguyên nhân.
    """
    cams = [c for c in crane.record_cameras if c.role is CameraRole.TCODE]
    p = _write(tmp_path, {cams[0].code: {}})  # bỏ sót cams[1]

    with pytest.raises(RuleConfigError) as exc:
        load_rule(p, TCode01Config, crane=crane, roles=[CameraRole.TCODE])
    msg = str(exc.value)
    assert cams[1].code in msg
    assert cams[1].key in msg, "phải kèm tên ngắn, không bắt người đọc tra mã"


def test_unknown_camera_code_is_rejected(tmp_path: Path, crane: CraneConfig) -> None:
    """Mã lạ thường là chép config từ cẩu khác — và nó kéo theo một camera thật thiếu config."""
    p = _write(tmp_path, {"GC99_1_2_3_4_5": {}})
    with pytest.raises(RuleConfigError, match="không thuộc cẩu GC03"):
        load_rule(p, Crane01Config, crane=crane, roles=[CameraRole.CRANE])


def test_camera_with_the_wrong_role_is_rejected(tmp_path: Path, crane: CraneConfig) -> None:
    """Camera có thật nhưng sai vai trò: rule sẽ không bao giờ nhận message của nó."""
    bottom = next(c for c in crane.record_cameras if c.role is CameraRole.BOTTOM)
    p = _write(tmp_path, {bottom.code: {}})
    with pytest.raises(RuleConfigError, match="sai vai trò"):
        load_rule(p, Crane01Config, crane=crane, roles=[CameraRole.CRANE])


def test_typo_in_a_key_is_rejected(tmp_path: Path, crane: CraneConfig) -> None:
    """``extra="forbid"``: gõ sai tên tham số là lỗi, không phải giá trị mặc định im lặng."""
    crane_cam = next(c for c in crane.record_cameras if c.role is CameraRole.CRANE)
    p = _write(tmp_path, {crane_cam.code: {"head_tresh": 0.7}})
    with pytest.raises(RuleConfigError, match="head_tresh"):
        load_rule(p, Crane01Config, crane=crane, roles=[CameraRole.CRANE])


def test_broken_json_says_which_file(tmp_path: Path, crane: CraneConfig) -> None:
    p = tmp_path / "config.json"
    p.write_text("{ khong phai json", encoding="utf-8")
    with pytest.raises(RuleConfigError, match="JSON hỏng"):
        load_rule(p, Crane01Config, crane=crane, roles=[CameraRole.CRANE])


# ---------------------------------------------------------------- điểm mốc


def test_lane_zones_are_one_polygon_per_lane() -> None:
    """Ba trường phẳng, không phải ánh xạ. ``Lane`` là tập đóng đúng ba giá trị.

    Ánh xạ ``{"1": …}`` gợi ý một quan hệ nhiều-nhiều không có thật, và cho phép khai làn
    "4" mà chỉ bắt được bằng một phép kiểm riêng.
    """
    fields = set(Crane01Config.model_fields)
    assert {"lane1_zone", "lane2_zone", "lane3_zone"} <= fields
    assert "lane_zones" not in fields

    cfg = Crane01Config.model_validate({"lane1_zone": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]})
    assert set(cfg.zones()) == {Lane.ONE}, "làn để trống thì không vào ánh xạ"


def test_lane_zone_beyond_three_is_rejected() -> None:
    """Không có làn 4: khai nó là lỗi, không phải một trường bị bỏ qua im lặng."""
    with pytest.raises(ValueError, match="lane4_zone"):
        Crane01Config.model_validate({"lane4_zone": [[0.0, 0.0]]})


def test_anchor_reads_and_writes_by_name() -> None:
    """``Anchor`` là ``Flag`` nên mặc định nó ra số — và ``1`` thì người sửa config không đoán được."""
    assert Crane01Config().model_dump(mode="json")["lane_anchor"] == "CENTER"
    assert Crane01Config.model_validate({"lane_anchor": "BOTTOM"}).lane_anchor is Anchor.BOTTOM
    assert (
        Crane01Config.model_validate({"lane_anchor": "CENTER|BOTTOM"}).lane_anchor
        == Anchor.CENTER | Anchor.BOTTOM
    )


def test_anchor_rejects_a_bare_number() -> None:
    with pytest.raises(ValueError, match="không phải số"):
        Crane01Config.model_validate({"lane_anchor": 1})


def test_anchor_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="không hợp lệ"):
        Crane01Config.model_validate({"lane_anchor": "TOP"})


# ---------------------------------------------------------------- mặc định


def test_defaults_match_the_spec_table() -> None:
    """Số mặc định phải khớp bảng trong docs/RULES.md — đó là nơi người đọc tra chúng."""
    assert Crane01Config().head_thresh == 0.6
    t = TCode01Config()
    assert (t.head_thresh, t.head_code_thresh, t.min_streak) == (0.8, 0.93, 3)
    c = CCode01Config()
    assert (c.top_k, c.sharpness_min, c.pair_distance_px) == (5, 1000.0, 60.0)
    assert c.ocr_threshold == 0.95
    assert (c.bitmap_threshold, c.box_threshold, c.character_threshold) == (0.1, 0.2, 0.3)
    assert (c.iso_threshold, c.min_streak) == (0.95, 3)


@pytest.mark.parametrize("spec", RULE_CONFIGS, ids=lambda s: s.code)
def test_config_puts_one_camera_per_line(spec: Any, crane: CraneConfig) -> None:
    """Mỗi camera đúng MỘT dòng — số camera đếm được bằng mắt và bằng ``wc -l``.

    Trải mỗi số ra một dòng thì một đa giác 26 đỉnh chiếm 78 dòng: file thành thứ không ai
    cuộn hết, và thiếu hẳn một camera cũng không ai thấy.
    """
    lines = rule_config_path(CONFIGS, "GC03", spec.code).read_text(encoding="utf-8").splitlines()
    n = len([c for c in crane.record_cameras if c.role in spec.roles])
    assert lines[0] == "{" and lines[-1] == "}"
    assert len(lines) - 2 == n, f"{spec.code}: {len(lines) - 2} dòng cho {n} camera"
    for line in lines[1:-1]:
        assert line.lstrip().startswith('"GC03_'), f"dòng không mở đầu bằng mã camera: {line[:40]}"


def test_every_declared_rule_has_a_config_dir() -> None:
    """``RULE_CONFIGS`` là nguồn sự thật: khai một rule thì `init` phải sinh ra thư mục cho nó."""
    for spec in RULE_CONFIGS:
        folder = rule_config_path(CONFIGS, "GC03", spec.code).parent
        assert (folder / "config.json").is_file(), f"{spec.code}: chưa chạy `init`?"
        assert (folder / "schema.json").is_file(), f"{spec.code}: thiếu schema"
        assert (folder / "changelog.md").is_file(), f"{spec.code}: thiếu changelog"
