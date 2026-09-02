"""Vùng OCR: khai ở mục riêng, toạ độ tương đối, bơm xuống đúng camera."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.config import CraneConfig, load_crane
from common.enum import CameraRole
from tests.conftest import GC03

ENV: dict[str, str] = {}


def crane() -> CraneConfig:
    return load_crane(GC03, env=ENV)


def test_rois_reach_the_camera_they_are_keyed_to() -> None:
    by_code = {c.code: c for c in crane().record_cameras}

    assert len(by_code["GC03_113_160_225_15_1508"].ocr_rois) == 8
    assert len(by_code["GC03_113_160_225_15_1514"].ocr_rois) == 2
    assert by_code["GC03_113_160_225_15_1517"].ocr_rois == [], "crane không có vùng OCR"


def test_every_roi_is_relative_and_inside_the_frame() -> None:
    """⚠️ v1 dùng pixel TUYỆT ĐỐI trong không gian nó tự co giãn về (720p cho camera 1),
    nên chuyển sang phải chia theo độ phân giải v1 **khai**, không phải nguồn.

    Chia nhầm theo nguồn 2688x1520 đặt mọi vùng lên góc trên trái — bầu trời và dầm cẩu,
    không một container nào — mà OCR vẫn chạy và vẫn trả chuỗi rỗng.
    """
    for cam in crane().record_cameras:
        for r in cam.ocr_rois:
            assert all(0.0 <= v <= 1.0 for v in r.roi), f"{cam.code}: {r.roi} ngoài [0..1]"


def test_only_ccode_cameras_carry_rois() -> None:
    for cam in crane().record_cameras:
        if cam.ocr_rois:
            assert cam.role is CameraRole.CCODE


def test_a_roi_keyed_to_an_unknown_camera_is_rejected() -> None:
    """Gõ sai mã ở đây làm vùng biến mất không dấu vết, và camera đó chạy OCR trên không
    có vùng nào — không exception, không log."""
    raw = GC03.read_text(encoding="utf-8").replace(
        "  GC03_113_160_225_15_1508:", "  GC03_113_160_225_15_9999:", 1
    )
    with pytest.raises(ValidationError, match="không tồn tại"):
        CraneConfig.model_validate(__import__("yaml").safe_load(raw))


def test_both_shapes_are_present_where_v1_declared_them() -> None:
    """Mỗi vùng chọn cặp model ``ccode_{det,rec}_{h,v}``; mất một hình dạng là mất một
    nửa số mã đọc được."""
    by_code = {c.code: c for c in crane().record_cameras}
    shapes = {r.shape for r in by_code["GC03_113_160_225_15_1508"].ocr_rois}
    assert shapes == {"horizontal", "vertical"}


def test_syncing_camera_codes_leaves_the_roi_lines_alone() -> None:
    """⚠️ Vùng OCR cũng là dòng ``- {...}`` — cùng hình dạng dòng camera, khác hẳn ý nghĩa.

    Quét cả file thì ``make codes`` coi mỗi vùng là một camera thiếu ``stream`` và từ chối
    chạy, nên cả CI lẫn việc thêm camera đều tắc.
    """
    from tools.camera_codes import synced_text

    assert synced_text(GC03) == GC03.read_text(encoding="utf-8")
