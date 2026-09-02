"""Chọn role nào ds_app chạy được — và vì sao "chạy được" khác "sẽ chạy"."""

from __future__ import annotations

from common.config import CraneConfig, load_crane
from common.enum import CameraRole
from ds_app.src.pipeline.inference import BLS_FOR_ROLE
from ds_app.src.pipeline.model import roles_with_cameras

CONFIG = "configs/cranes/GC03.yaml"


def crane() -> CraneConfig:
    return load_crane(CONFIG)


def test_only_roles_with_both_cameras_and_a_model_are_returned() -> None:
    got = roles_with_cameras(crane())
    assert set(got) == {CameraRole.CRANE, CameraRole.TCODE}


def test_ccode_is_excluded_because_it_has_no_model_yet() -> None:
    """⚠️ Hồi quy đo được: ``ccode`` có camera và ``runs_model`` là True, nên bản đầu nhận
    nó vào. Kết quả khi chạy thật: 5 camera decode suốt phiên để rồi mỗi khung ném
    ``KeyError`` — **1 503 lỗi trong 60 giây**, trong khi hai role kia vẫn đúng nên bảng
    tổng kết trông gần như bình thường.

    ``runs_model`` nói role đó *rốt cuộc* chạy model; ``BLS_FOR_ROLE`` nói hôm nay đã có
    model chưa. Hai câu khác nhau.
    """
    config = crane()
    assert config.cameras[CameraRole.CCODE], "GC03 phải có camera ccode để test có nghĩa"
    assert CameraRole.CCODE.runs_model
    assert CameraRole.CCODE not in roles_with_cameras(config)


def test_roles_that_never_decode_are_excluded() -> None:
    got = roles_with_cameras(crane())
    assert CameraRole.BOTTOM not in got
    assert CameraRole.EVIDENCE_ONLY not in got


def test_every_returned_role_has_a_model_to_call() -> None:
    """Đối chứng: nếu phép lọc hỏng thì probe sẽ gửi khung tới một model không tồn tại."""
    assert all(role in BLS_FOR_ROLE for role in roles_with_cameras(crane()))


def test_every_returned_camera_actually_decodes() -> None:
    """Camera không decode thì không bao giờ có khung; đưa nó vào muxer chỉ làm muxer chờ."""
    for cams in roles_with_cameras(crane()).values():
        assert all(cam.decodes for cam in cams)
