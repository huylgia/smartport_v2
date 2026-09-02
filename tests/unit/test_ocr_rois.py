"""Vùng OCR: khai ở mục riêng, toạ độ tương đối, bơm xuống đúng camera."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.config import CraneConfig, load_crane
from common.enum import CameraRole, ContainerDim, Lane
from tests.conftest import GC03

ENV: dict[str, str] = {}


def crane() -> CraneConfig:
    return load_crane(GC03, env=ENV)


def test_rois_reach_the_camera_they_are_keyed_to() -> None:
    by_code = {c.code: c for c in crane().record_cameras}

    assert len(by_code["GC03_113_160_225_15_1508"].ocr_rois) == 4
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


def test_only_ccode1_reads_vertical_codes() -> None:
    """⚠️ Quyết định NGHIỆP VỤ, không phải port trung thành: v1 khai thêm vùng dọc cho
    ``..._1511`` và ``..._1513``, và bỏ chúng là **giảm** thứ v1 phát hiện được.

    Chỉ ccode1 nhìn được mã dọc ở vị trí dùng được.
    """
    by_code = {c.code: c for c in crane().record_cameras}

    assert {s for r in by_code["GC03_113_160_225_15_1508"].ocr_rois for s in r.shapes} == {
        "horizontal",
        "vertical",
    }
    for code in ("1511", "1513", "1514", "1515"):
        cam = by_code[f"GC03_113_160_225_15_{code}"]
        assert {s for r in cam.ocr_rois for s in r.shapes} == {"horizontal"}, code


def test_one_region_serves_every_shape_it_declares() -> None:
    """v1 chép toạ độ cho từng hình dạng nên hai bản trôi khỏi nhau được — và đã trôi ở
    ccode1 lane1/20feet: ngang (0,138)-(535,720) so dọc (0,161)-(560,683).

    v2 giữ MỘT vùng là **hợp** của chúng, nên không model nào mất diện tích nó đang có.
    """
    by_code = {c.code: c for c in crane().record_cameras}
    roi = next(
        r
        for r in by_code["GC03_113_160_225_15_1508"].ocr_rois
        if r.lane == Lane.ONE and r.cont_dim is ContainerDim.FT20
    )

    assert set(roi.shapes) == {"horizontal", "vertical"}
    # Hợp của (0,138,535,720) và (0,161,560,683) trong không gian 720p mà v1 khai.
    assert roi.roi == pytest.approx((0.0, 138 / 720, 560 / 1280, 1.0), abs=1e-4)
    # Tham số thì KHÔNG dùng chung: mỗi model một kích thước đầu vào.
    assert roi.shapes["horizontal"].input_size != roi.shapes["vertical"].input_size


def test_a_region_with_no_shape_is_rejected() -> None:
    """Vùng không chạy hình dạng nào chỉ tốn một lần cắt ảnh."""
    from common.config import OcrRoi

    with pytest.raises(ValidationError):
        OcrRoi(lane=Lane.ONE, cont_dim=ContainerDim.FT40, roi=(0.1, 0.1, 0.5, 0.5), shapes={})


def test_syncing_camera_codes_leaves_the_roi_lines_alone() -> None:
    """⚠️ Vùng OCR cũng là dòng ``- {...}`` — cùng hình dạng dòng camera, khác hẳn ý nghĩa.

    Quét cả file thì ``make codes`` coi mỗi vùng là một camera thiếu ``stream`` và từ chối
    chạy, nên cả CI lẫn việc thêm camera đều tắc.
    """
    from tools.camera_codes import synced_text

    assert synced_text(GC03) == GC03.read_text(encoding="utf-8")
