"""Config cẩu: validate fail-fast, nội suy secret, và suy ra camera nào được decode."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.config import CameraConfig, ConfigError, OcrRoi, load_crane
from common.enum import CameraRole, Lane

REPO = Path(__file__).resolve().parents[2]
GC03 = REPO / "configs" / "cranes" / "GC03.yaml"

ENV = {  # pragma: allowlist secret — URL giả, cố ý có dạng có credential
    f"CAM{i:02d}_RTSP": f"rtsp://u:p@10.0.0.1:{1500 + i}/s" for i in range(1, 12)
}


def _minimal(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "crane_id": "GC03",
        "berth_no": "TS03",
        "num_lane": 3,
        "cameras": [
            {"id": 1, "name": "ccode", "role": "ccode", "rtsp_record": "rtsp://x"},
            {"id": 9, "name": "day", "role": "bottom", "rtsp_record": "rtsp://y"},
        ],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- file thật


def test_gc03_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """File cấu hình thật trong repo phải load được — nếu không, CI sạch mà deploy chết."""
    cfg = load_crane(GC03, env=ENV)
    assert cfg.crane_id == "GC03"
    assert len(cfg.cameras) == 10


def test_gc03_decode_set_matches_the_nvdec_budget() -> None:
    """8/10 camera vào nhánh model; `bottom` và `evidence_only` chỉ ghi hình.

    Đây là ràng buộc NVDEC, không phải sở thích: decode cả 10 camera 2688x1520@30 vượt
    trần một NVDEC của GA106. Xem HARDWARE_BUDGET §2.2.
    """
    cfg = load_crane(GC03, env=ENV)
    assert len(cfg.model_cameras) == 8
    assert len(cfg.record_cameras) == 10, "MỌI camera phải được ghi hình"
    assert {c.id for c in cfg.cameras} - {c.id for c in cfg.model_cameras} == {2, 9}


def test_record_covers_every_camera_including_undecoded() -> None:
    """Ảnh bằng chứng 6 mặt cần cả camera không decode — nên nhánh ghi phủ hết."""
    cfg = load_crane(GC03, env=ENV)
    assert [c.id for c in cfg.record_cameras] == [c.id for c in cfg.cameras]


# ---------------------------------------------------------------- secret


def test_env_reference_is_resolved() -> None:
    cfg = load_crane(GC03, env=ENV)
    assert cfg.camera(1).rtsp_record == ENV["CAM01_RTSP"]
    assert "${" not in cfg.camera(1).rtsp_record


def test_missing_env_var_fails_with_the_variable_name(tmp_path: Path) -> None:
    """Thiếu secret phải nói RÕ thiếu biến nào — không thì người vận hành mò từng camera."""
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {"id": 1, "name": "a", "role": "ccode", "rtsp_record": "${CAM_KHONG_CO}"},
                ]
            )
        )
    )
    with pytest.raises(ConfigError, match="CAM_KHONG_CO"):
        load_crane(p, env={})


def test_partial_interpolation_is_rejected(tmp_path: Path) -> None:
    """Chỉ nhận TOÀN BỘ chuỗi là tham chiếu; ghép từ nhiều mảnh bị TỪ CHỐI.

    Ghép URL (``rtsp://${USER}:${PW}@host``) là cách dễ nhất để lộ mật khẩu vào log khi
    một mảnh thiếu: chuỗi nửa vời vẫn "trông hợp lệ" và vẫn được in ra. Trước đây nó chỉ
    được để nguyên; nay validator chặn hẳn, vì một URL còn ``${...}`` không bao giờ kết
    nối được — hỏng lúc load tốt hơn hỏng lúc chạy.
    """
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {"id": 1, "name": "a", "role": "ccode", "rtsp_record": "rtsp://${U}@h/s"},
                ]
            )
        )
    )
    with pytest.raises(ConfigError, match="chưa nội suy"):
        load_crane(p, env={"U": "bimat"})


# ---------------------------------------------------------------- fail-fast


def test_typo_in_key_is_rejected(tmp_path: Path) -> None:
    """`extra="forbid"` cho config: gõ sai khoá là lỗi lúc load.

    Ngược với message contract (`extra="ignore"`), và cố ý: message đi qua ranh giới
    process nên phải chịu được nâng cấp lệch pha; config thì không.
    """
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {
                        "id": 1,
                        "name": "a",
                        "role": "ccode",
                        "rtsp_record": "rtsp://h/s",
                        "rstp_model": "sai",
                    },
                ]
            )
        )
    )
    with pytest.raises(ConfigError, match="rstp_model"):
        load_crane(p, env={})


def test_duplicate_camera_id_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {"id": 1, "name": "a", "role": "ccode", "rtsp_record": "rtsp://h/s"},
                    {"id": 1, "name": "b", "role": "tcode", "rtsp_record": "rtsp://h/s"},
                ]
            )
        )
    )
    with pytest.raises(ConfigError, match="trùng"):
        load_crane(p, env={})


def test_config_with_no_model_camera_is_rejected(tmp_path: Path) -> None:
    """Không camera nào decode ⇒ hệ chạy mà không bao giờ ra kết quả.

    Loại lỗi im lặng tốn nhiều giờ nhất để lần, và gần như luôn là gõ sai `role`.
    """
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {"id": 9, "name": "day", "role": "bottom", "rtsp_record": "rtsp://h/s"},
                ]
            )
        )
    )
    with pytest.raises(ConfigError, match="không camera nào chạy model"):
        load_crane(p, env={})


def test_lane_zone_beyond_num_lane_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                num_lane=2,
                cameras=[
                    {
                        "id": 3,
                        "name": "a",
                        "role": "tcode",
                        "rtsp_record": "rtsp://h/s",
                        "lane_zones": {"3": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
                    },
                ],
            )
        )
    )
    with pytest.raises(ConfigError, match="chỉ có 2 lane"):
        load_crane(p, env={})


def test_ocr_rois_on_a_non_ccode_camera_is_rejected() -> None:
    with pytest.raises(ValueError, match="chỉ vai trò 'ccode'"):
        CameraConfig(
            id=3,
            name="a",
            role=CameraRole.TCODE,
            rtsp_record="rtsp://h/s",
            ocr_rois=[
                OcrRoi(
                    shape="horizontal",
                    lane=Lane.ONE,
                    roi=(0.1, 0.1, 0.5, 0.5),
                    input_size=(640, 672),
                )
            ],
        )


def test_lane_zones_on_a_camera_that_never_decodes_is_rejected() -> None:
    """Khai vùng lane cho camera không chạy model là lỗi thầm lặng: nó không bao giờ dùng tới."""
    with pytest.raises(ValueError, match="không chạy model"):
        CameraConfig(
            id=9,
            name="day",
            role=CameraRole.BOTTOM,
            rtsp_record="rtsp://h/s",
            lane_zones={Lane.ONE: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]},
        )


def test_relative_coordinates_reject_pixels() -> None:
    """Dán nhầm toạ độ pixel vào trường mong đợi [0..1] — lỗi hay gặp nhất. Xem DN-002."""
    with pytest.raises(ValueError):
        CameraConfig(
            id=3,
            name="a",
            role=CameraRole.TCODE,
            rtsp_record="rtsp://h/s",
            lane_zones={Lane.ONE: [(505.0, 81.0), (1115.0, 81.0), (1115.0, 662.0)]},
        )


def test_inverted_roi_is_rejected() -> None:
    with pytest.raises(ValueError, match="lật ngược"):
        OcrRoi(shape="vertical", lane=Lane.ONE, roi=(0.5, 0.5, 0.1, 0.1), input_size=(576, 608))


# ---------------------------------------------------------------- mã camera


def test_camera_code_includes_the_port() -> None:
    """⚠️ Cả 10 camera GC03 dùng CHUNG một IP — gateway NAT, mỗi cổng một camera.

    Mã chỉ gồm IP sẽ giống hệt nhau cho cả 10, tức không định danh được gì.
    """
    cfg = load_crane(GC03, env=ENV)
    codes = [c.code for c in cfg.cameras]

    assert len(set(codes)) == len(codes), "mã camera phải phân biệt được"
    assert all(c.startswith("GC03_") for c in codes)


def test_camera_code_format() -> None:
    """``<mã cẩu>_<ip>_<cổng>`` — tiền tố cẩu để hai cẩu sau NAT riêng không trùng mã."""
    cam = CameraConfig(
        crane_id="GC03",
        id=1,
        name="x",
        role=CameraRole.CCODE,
        rtsp_record="rtsp://u:p@113.160.225.15:1508//CH001.sdp",
    )
    assert cam.code == "GC03_113_160_225_15_1508"


def test_crane_id_is_stamped_onto_cameras_at_load() -> None:
    """Camera không khai `crane_id` trong YAML — nó được bơm xuống lúc load.

    Khai hai chỗ là hai chỗ có thể lệch nhau.
    """
    cfg = load_crane(GC03, env=ENV)
    assert all(c.crane_id == "GC03" for c in cfg.cameras)


def test_declared_crane_id_on_a_camera_is_overridden(tmp_path: Path) -> None:
    """Người dùng khai đè `crane_id` cho camera phải bị bỏ qua — nếu không một camera có
    thể tự nhận thuộc cẩu khác, và dữ liệu của nó đi nhầm chỗ."""
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {
                        "id": 1,
                        "name": "a",
                        "role": "ccode",
                        "rtsp_record": "rtsp://10.0.0.1:1/s",
                        "crane_id": "CAI-KHAC",
                    },
                ]
            )
        )
    )
    cfg = load_crane(p, env={})
    assert cfg.camera(1).crane_id == "GC03"
    assert cfg.camera(1).code.startswith("GC03_")


def test_camera_code_ignores_credentials() -> None:
    """URL có credential; mã thì không — nó đi vào log và tên file."""
    cam = CameraConfig(
        crane_id="GC03",
        id=1,
        name="x",
        role=CameraRole.CCODE,
        rtsp_record="rtsp://admin:secret@10.0.0.1:554/s",  # pragma: allowlist secret
    )
    assert "admin" not in cam.code and "secret" not in cam.code
    assert cam.code == "GC03_10_0_0_1_554"


def test_camera_code_without_a_port() -> None:
    cam = CameraConfig(
        crane_id="GC03", id=1, name="x", role=CameraRole.CCODE, rtsp_record="rtsp://10.0.0.1/s"
    )
    assert cam.code == "GC03_10_0_0_1"


def test_name_is_only_a_description() -> None:
    """``name`` là mô tả cho người đọc, KHÔNG phải định danh — nó đổi khi ai đó sửa cho dễ hiểu."""
    cfg = load_crane(GC03, env=ENV)
    cam = cfg.camera(1)
    assert cam.name == "Mặt phải trước"
    assert cam.code != cam.name


def test_no_fps_knob_remains() -> None:
    """Không còn núm chỉnh fps: decimate ở decoder KHÔNG tiết kiệm NVDEC.

    Nguồn là IPPP, GOP 50, không khung B — mọi khung đều phải giải mã, ``drop-frame-interval``
    chỉ vứt output SAU đó. Giảm nhịp là việc của tầng nghiệp vụ. Xem HARDWARE_BUDGET §2.2.
    """
    assert not hasattr(CameraConfig, "keep_interval")
    assert "drop_frame_interval" not in CameraConfig.model_fields


# ---------------------------------------------------------------- thứ tự nguồn


def test_model_camera_order_is_the_streammux_pad_order() -> None:
    """Thứ tự khai báo = `pad_index` của nvstreammux, và probe dùng nó để biết khung của ai.

    Đổi thứ tự trong YAML là đổi ánh xạ khung→camera, nên test này khoá nó lại.
    """
    cfg = load_crane(GC03, env=ENV)
    assert [c.id for c in cfg.model_cameras] == [1, 4, 6, 7, 8, 3, 5, 10]


def test_unknown_camera_id_lists_the_known_ones() -> None:
    cfg = load_crane(GC03, env=ENV)
    with pytest.raises(KeyError, match="đang có"):
        cfg.camera(99)


def test_missing_file_says_which_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="khong-ton-tai"):
        load_crane(tmp_path / "khong-ton-tai.yaml")


def test_crane_config_is_frozen() -> None:
    """Config bất biến: không service nào được sửa nó lúc chạy."""
    cfg = load_crane(GC03, env=ENV)
    with pytest.raises(ValueError):
        cfg.crane_id = "GC99"


# ---------------------------------------------------------------- URL RTSP


def test_url_with_a_delimiter_is_rejected() -> None:
    """⚠️ Lỗi đã xảy ra thật: trích URL từ định dạng phân tách bằng `|` mà quên dừng.

    GStreamer KHÔNG báo lỗi — nó giữ phần thừa trong path và gửi
    `SETUP //CH001.sdp|h265|10|||`. Camera đang dùng tình cờ bỏ qua, nên nó chạy được và
    che mất lỗi cho tới khi gặp firmware khác.
    """
    with pytest.raises(ValueError, match="phân tách"):
        CameraConfig(
            id=1,
            name="x",
            role=CameraRole.CCODE,
            rtsp_record="rtsp://h:1508//CH001.sdp|h265|10|||",
        )


def test_non_rtsp_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="rtsp://"):
        CameraConfig(id=1, name="x", role=CameraRole.CCODE, rtsp_record="http://h/s")


def test_unresolved_placeholder_is_rejected() -> None:
    """Biến chưa nội suy lọt xuống đây nghĩa là secret không được nạp — đừng chạy tiếp."""
    with pytest.raises(ValueError, match="chưa nội suy"):
        CameraConfig(id=1, name="x", role=CameraRole.CCODE, rtsp_record="${CAM01_RTSP}")


def test_rtsps_is_accepted() -> None:
    cam = CameraConfig(id=1, name="x", role=CameraRole.CCODE, rtsp_record="rtsps://h/s")
    assert cam.rtsp_record.startswith("rtsps://")


def test_whitespace_is_trimmed_not_tolerated_inside() -> None:
    cam = CameraConfig(id=1, name="x", role=CameraRole.CCODE, rtsp_record="  rtsp://h/s  ")
    assert cam.rtsp_record == "rtsp://h/s"
    with pytest.raises(ValueError, match="phân tách"):
        CameraConfig(id=1, name="x", role=CameraRole.CCODE, rtsp_record="rtsp://h/s extra")


def test_duplicate_camera_code_is_rejected(tmp_path: Path) -> None:
    """Hai camera cùng điểm cuối ⇒ cùng mã ⇒ dữ liệu bị gán nhầm, im lặng.

    Đã suýt xảy ra: fixture test dùng `rtsp://h/1`, `rtsp://h/2`… — khác path nhưng CÙNG
    host và không cổng, nên cả 10 camera ra cùng một mã.
    """
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            _minimal(
                cameras=[
                    {"id": 1, "name": "a", "role": "ccode", "rtsp_record": "rtsp://h/1"},
                    {"id": 2, "name": "b", "role": "ccode", "rtsp_record": "rtsp://h/2"},
                ]
            )
        )
    )
    with pytest.raises(ConfigError, match="mã camera trùng"):
        load_crane(p, env={})
