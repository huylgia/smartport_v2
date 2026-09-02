"""Config cẩu: validate fail-fast, nội suy secret, và suy ra camera nào được decode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from common.config import CameraConfig, ConfigError, OcrRoi, load_crane
from common.enum import CameraRole, ContainerDim, Lane
from tests.conftest import GC03

REPO = Path(__file__).resolve().parents[2]

ENV: dict[str, str] = {}


def _minimal(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "crane_id": "GC03",
        "berth_no": "TS03",
        "num_lane": 3,
        "cameras": {
            "ccode": [{"stream": "rtsp://h:1/s"}],
            "bottom": [{"stream": "rtsp://h:2/s"}],
        },
    }
    base.update(over)
    return base


def _write(tmp_path: Path, cfg: dict[str, Any]) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _env(**urls: str) -> dict[str, str]:
    """Env theo tên ngắn. ``_env(ccode1="rtsp://h:1/s")``."""
    return {k.upper(): v for k, v in urls.items()}


# ---------------------------------------------------------------- file thật


def test_gc03_loads() -> None:
    """File cấu hình thật trong repo phải load được — nếu không, CI sạch mà deploy chết."""
    cfg = load_crane(GC03, env=ENV)
    assert cfg.crane_id == "GC03"
    assert len(cfg.record_cameras) == 10
    assert set(cfg.cameras) == set(CameraRole)


def test_gc03_decode_set_matches_the_nvdec_budget() -> None:
    """8/10 camera vào nhánh model; `bottom` và `evidence_only` chỉ ghi hình.

    Đây là ràng buộc NVDEC, không phải sở thích: decode cả 10 camera 2688x1520@30 vượt
    trần một NVDEC của GA106. Xem HARDWARE_BUDGET §2.2.
    """
    cfg = load_crane(GC03, env=ENV)
    assert len(cfg.model_cameras) == 8
    assert len(cfg.record_cameras) == 10, "MỌI camera phải được ghi hình"
    only_recorded = {c.key for c in cfg.record_cameras} - {c.key for c in cfg.model_cameras}
    assert only_recorded == {"bottom1", "evidence_only1"}


def test_record_covers_every_camera_including_undecoded() -> None:
    """Ảnh bằng chứng 6 mặt cần cả camera không decode — nên nhánh ghi phủ hết."""
    cfg = load_crane(GC03, env=ENV)
    assert len(cfg.record_cameras) == sum(len(g) for g in cfg.cameras.values())


# ---------------------------------------------------------------- khoá & mã


def test_yaml_carries_the_stream_identity() -> None:
    """Host, cổng, path nằm TRONG config; chỉ credential ở env."""
    # Đọc camera THẬT, không khớp chuỗi trên text: một dòng chú thích nhắc `stream:` cũng
    # khớp, và test khi đó đo văn bản chứ không đo cấu hình.
    cams = load_crane(GC03, env={}).record_cameras
    assert len(cams) == 10
    assert all(c.stream.startswith("rtsp://") for c in cams)
    assert not [c for c in cams if "@" in c.stream], "credential lọt vào config"


def test_unknown_role_group_is_rejected(tmp_path: Path) -> None:
    """Gõ sai tên nhóm là lỗi lúc load.

    Khoá nhóm là :class:`CameraRole` chứ không phải chuỗi tự do — nếu không, ``ccodee:``
    tạo một nhóm mà không gì tiêu thụ, và hệ chạy trong khi camera đó không bao giờ được
    dùng tới.
    """
    p = _write(tmp_path, _minimal(cameras={"ccodee": [{"stream": "rtsp://h:1/s"}]}))
    with pytest.raises(ConfigError):
        load_crane(p, env={})


# ---------------------------------------------------------------- fail-fast


def test_typo_in_key_is_rejected(tmp_path: Path) -> None:
    """`extra="forbid"` cho config: gõ sai khoá là lỗi lúc load.

    Ngược với message contract (`extra="ignore"`), và cố ý: message đi qua ranh giới
    process nên phải chịu được nâng cấp lệch pha; config thì không.
    """
    p = _write(tmp_path, _minimal(cameras={"ccode": [{"stream": "rtsp://h:1/s", "xyz": 1}]}))
    with pytest.raises(ConfigError, match="xyz"):
        load_crane(p, env={})


def test_config_with_no_model_camera_is_rejected(tmp_path: Path) -> None:
    """Không camera nào decode ⇒ hệ chạy mà không bao giờ ra kết quả.

    Loại lỗi im lặng tốn nhiều giờ nhất để lần, và gần như luôn là gõ sai `role`.
    """
    p = _write(tmp_path, _minimal(cameras={"bottom": [{"stream": "rtsp://h:1/s"}]}))
    with pytest.raises(ConfigError, match="không camera nào chạy model"):
        load_crane(p, env={})


def _roi(**over: object) -> OcrRoi:
    base: dict[str, object] = {
        "shape": "horizontal",
        "lane": Lane.ONE,
        "cont_dim": ContainerDim.FT40,
        "roi": (0.1, 0.1, 0.5, 0.5),
        "input_size": (640, 672),
    }
    base.update(over)
    return OcrRoi.model_validate(base)


def test_ocr_rois_on_a_non_ccode_camera_is_rejected() -> None:
    with pytest.raises(ValueError, match="chỉ vai trò 'ccode'"):
        CameraConfig(role=CameraRole.TCODE, stream="rtsp://h:1/s", ocr_rois=[_roi()])


def test_ocr_roi_carries_the_meaning_the_message_needs() -> None:
    """``lane`` và ``cont_dim`` ở đây vì :class:`OcrResult` mang chúng trên dây.

    Probe của ds_app phải điền hai trường đó. Tách chúng sang config rule thì ds_app hoặc
    phải đọc ngược config của rule, hoặc không điền nổi message.
    """
    assert set(OcrRoi.model_fields) == {
        "shape",
        "lane",
        "cont_dim",
        "roi",
        "input_size",
        "expand_ratio",
    }
    assert "ocr_threshold" not in OcrRoi.model_fields, "ngưỡng là tham số hiệu chỉnh của rule"


def test_lane_zones_are_not_a_pipeline_setting() -> None:
    """Vùng lane thuộc config của rule, không phải của ds_app. Xem common/rule_config.py."""
    assert "lane_zones" not in CameraConfig.model_fields


def test_relative_coordinates_reject_pixels() -> None:
    """Dán nhầm toạ độ pixel vào trường mong đợi [0..1] — lỗi hay gặp nhất. Xem DN-002."""
    with pytest.raises(ValueError):
        _roi(roi=(505.0, 81.0, 1115.0, 662.0))


def test_inverted_roi_is_rejected() -> None:
    with pytest.raises(ValueError, match="lật ngược"):
        _roi(roi=(0.5, 0.5, 0.1, 0.1))


# ---------------------------------------------------------------- mã camera


def test_camera_code_includes_the_port() -> None:
    """⚠️ Cả 10 camera GC03 dùng CHUNG một IP — gateway NAT, mỗi cổng một camera.

    Mã chỉ gồm IP sẽ giống hệt nhau cho cả 10, tức không định danh được gì.
    """
    cfg = load_crane(GC03, env=ENV)
    codes = [c.code for c in cfg.record_cameras]

    assert len(set(codes)) == len(codes), "mã camera phải phân biệt được"
    assert all(c.startswith("GC03_") for c in codes)


def test_camera_code_format() -> None:
    """``<mã cẩu>_<ip>_<cổng>`` — tiền tố cẩu để hai cẩu sau NAT riêng không trùng mã."""
    cam = CameraConfig(
        crane_id="GC03",
        role=CameraRole.CCODE,
        stream="rtsp://113.160.225.15:1508//CH001.sdp",
    )
    assert cam.code == "GC03_113_160_225_15_1508"


def test_camera_code_does_not_follow_the_position() -> None:
    """Đổi vị trí trong nhóm KHÔNG đổi mã trên dây.

    Khoá (``ccode3``) là handle cho người; mã là định danh trong Kafka và tên thư mục
    segment. Mã bám theo THIẾT BỊ (ip + cổng), không bám theo file cấu hình — nên dời một
    camera trong file không làm đứt liên kết với dữ liệu đã ghi.
    """
    common: dict[str, Any] = {
        "crane_id": "GC03",
        "role": CameraRole.CCODE,
        "stream": "rtsp://10.0.0.1:1508/s",
    }
    a, b = CameraConfig(index=1, **common), CameraConfig(index=7, **common)
    assert a.key == "ccode1" and b.key == "ccode7"
    assert a.code == b.code == "GC03_10_0_0_1_1508"


def test_crane_id_and_key_are_stamped_at_load() -> None:
    """Camera không khai `crane_id` lẫn `key` trong thân — cả hai được bơm xuống lúc load.

    Khai hai chỗ là hai chỗ có thể lệch nhau.
    """
    cfg = load_crane(GC03, env=ENV)
    assert all(c.crane_id == "GC03" for c in cfg.record_cameras)
    for role, group in cfg.cameras.items():
        assert [c.key for c in group] == [f"{role.value}{i}" for i in range(1, len(group) + 1)]


def test_declared_identity_on_a_camera_is_overridden(tmp_path: Path) -> None:
    """Khai đè `crane_id`/`key` trong thân phải bị bỏ qua.

    Nếu không, một camera có thể tự nhận thuộc cẩu khác — hoặc tự nhận khoá khác, và khi
    đó nó đọc URL của camera khác.
    """
    p = _write(
        tmp_path,
        _minimal(
            cameras={
                "ccode": [
                    {"stream": "rtsp://h:1/s", "crane_id": "CAI-KHAC", "role": "bottom", "index": 9}
                ],
                "bottom": [{"stream": "rtsp://h:2/s"}],
            }
        ),
    )
    cfg = load_crane(p, env={})
    cam = cfg.camera("ccode1")
    assert cam.crane_id == "GC03"
    assert cam.role is CameraRole.CCODE, "vai trò phải theo NHÓM chứa nó"
    assert cam.index == 1


def test_camera_code_ignores_credentials() -> None:
    """URL có credential; mã thì không — nó đi vào log và tên file."""
    cam = CameraConfig(
        crane_id="GC03",
        role=CameraRole.CCODE,
        stream="rtsp://10.0.0.1:554/s",
        credential="admin:secret",  # pragma: allowlist secret
    )
    assert "admin" not in cam.code and "secret" not in cam.code
    assert cam.code == "GC03_10_0_0_1_554"


def test_camera_code_without_a_port() -> None:
    cam = CameraConfig(crane_id="GC03", role=CameraRole.CCODE, stream="rtsp://10.0.0.1/s")
    assert cam.code == "GC03_10_0_0_1"


def test_desc_is_only_a_description() -> None:
    """``desc`` là mô tả cho người đọc, KHÔNG phải định danh — nó đổi khi ai đó sửa cho dễ hiểu."""
    cfg = load_crane(GC03, env=ENV)
    cam = cfg.camera("ccode1")
    assert cam.desc == "Mặt phải trước"
    assert cam.code != cam.desc
    assert cam.key != cam.desc


def test_desc_is_optional() -> None:
    """Mô tả để trống được — nó không mang chức năng nào."""
    cam = CameraConfig(role=CameraRole.CCODE, stream="rtsp://h:1/s")
    assert cam.desc == ""


def test_model_fps_becomes_a_decoder_divisor() -> None:
    """``model_fps`` là nhịp rule CẦN; ds_app quy ra ``drop-frame-interval``.

    Số mặc định theo HARDWARE_BUDGET §2.7. Property của decoder có nghĩa "giữ 1 khung mỗi
    N khung", nên N = fps nguồn / fps mục tiêu.
    """
    for role, want, n in (
        (CameraRole.CCODE, 5.0, 6),
        (CameraRole.CRANE, 3.3, 9),
        (CameraRole.TCODE, 2.0, 15),
    ):
        cam = CameraConfig(role=role, stream="rtsp://h:1/s", source_fps=30.0, model_fps=want)
        assert cam.drop_frame_interval == n
        assert abs(cam.effective_fps - want) < 0.05


def test_no_decimation_by_default() -> None:
    """Không khai thì giữ nguyên fps nguồn — decoder không bỏ khung nào."""
    cam = CameraConfig(role=CameraRole.CCODE, stream="rtsp://h:1/s", source_fps=30.0)
    assert cam.drop_frame_interval == 0
    assert cam.effective_fps == 30.0


def test_effective_fps_shows_the_rounding() -> None:
    """Chia rồi làm tròn nên nhịp thật hiếm khi đúng bằng ``model_fps`` — phải nhìn thấy được."""
    cam = CameraConfig(role=CameraRole.CCODE, stream="rtsp://h:1/s", source_fps=30.0, model_fps=7.0)
    assert cam.drop_frame_interval == 4, "30/7 = 4,3 → làm tròn 4"
    assert cam.effective_fps == 7.5, "không phải 7,0 — đó là lý do có effective_fps"


def test_model_fps_on_a_camera_that_never_decodes_is_rejected() -> None:
    """Nhánh decode của camera chỉ-ghi-hình không ai kéo; đặt nhịp cho nó là vô nghĩa."""
    with pytest.raises(ValueError, match="không chạy model"):
        CameraConfig(role=CameraRole.BOTTOM, stream="rtsp://h:1/s", model_fps=5.0)


def test_model_fps_above_the_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="lớn hơn fps nguồn"):
        CameraConfig(role=CameraRole.CCODE, stream="rtsp://h:1/s", source_fps=30.0, model_fps=60.0)


def test_shipped_config_decimates_every_model_camera() -> None:
    """Không đặt thì tải suy luận là fps NGUỒN — gấp 6 lần fps mục tiêu (§2.2)."""
    cfg = load_crane(GC03, env={})
    for cam in cfg.model_cameras:
        assert cam.model_fps is not None, f"{cam.key} chạy model mà không giảm nhịp"
        assert cam.drop_frame_interval > 1
    decoding = {c.key for c in cfg.model_cameras}
    for cam in cfg.record_cameras:
        if cam.key not in decoding:
            assert cam.model_fps is None, f"{cam.key} chỉ ghi hình mà khai model_fps"


def test_no_hand_allocated_id_remains() -> None:
    """Không còn ``id``: nó là tên thứ ba cho cùng một camera, và nó trôi được khỏi URL.

    Định danh giờ là ``key`` (cho người) và ``code`` (trên dây, suy từ URL).
    """
    assert "id" not in CameraConfig.model_fields
    assert "name" not in CameraConfig.model_fields
    assert "key" not in CameraConfig.model_fields, "khoá phải dẫn xuất, không phải trường"


# ---------------------------------------------------------------- thứ tự nguồn


def test_model_camera_order_is_the_streammux_pad_order() -> None:
    """Thứ tự khai báo = `pad_index` của nvstreammux, và probe dùng nó để biết khung của ai.

    Đổi thứ tự trong YAML là đổi ánh xạ khung→camera, nên test này khoá nó lại.

    Ánh xạ giữ được thứ tự vì dict Python giữ thứ tự chèn và PyYAML nạp theo thứ tự file.
    """
    cfg = load_crane(GC03, env=ENV)
    assert [c.key for c in cfg.model_cameras] == [
        "ccode1",
        "ccode2",
        "ccode3",
        "ccode4",
        "ccode5",
        "tcode1",
        "tcode2",
        "crane1",
    ]


def test_unknown_camera_key_lists_the_known_ones() -> None:
    cfg = load_crane(GC03, env=ENV)
    with pytest.raises(KeyError, match="đang có"):
        cfg.camera("ccode99")


def test_missing_file_says_which_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="khong-ton-tai"):
        load_crane(tmp_path / "khong-ton-tai.yaml")


def test_crane_config_is_frozen() -> None:
    """Config bất biến: không service nào được sửa nó lúc chạy."""
    cfg = load_crane(GC03, env=ENV)
    with pytest.raises(ValueError):
        cfg.crane_id = "GC99"


# ---------------------------------------------------------------- thêm camera


def test_adding_a_backup_camera_is_one_yaml_entry(tmp_path: Path) -> None:
    """Thêm camera cho một chức năng ĐANG CÓ = một mục YAML. Không hơn.

    Không đặt tên, không cấp phát số, không đụng file env lẫn docker-compose. ``make codes``
    điền ``code`` rồi đối chiếu với mọi service khoá theo mã đó.
    """
    raw = yaml.safe_load(GC03.read_text(encoding="utf-8"))
    raw["cameras"]["ccode"].append(
        {"desc": "Mặt phải trước - dự phòng", "stream": "rtsp://113.160.225.15:1599//CH001.sdp"}
    )
    cfg = load_crane(_write(tmp_path, raw), env={})

    cam = cfg.camera("ccode6")
    assert cam.role is CameraRole.CCODE
    assert cam.code == "GC03_113_160_225_15_1599", "mã tự suy, không phải khai tay"
    assert len(cfg.by_role(CameraRole.CCODE)) == 6
    assert len(cfg.model_cameras) == 9, "camera ccode mới phải vào nhánh model"


def test_adding_a_camera_does_not_disturb_the_other_groups(tmp_path: Path) -> None:
    """Thêm vào một nhóm KHÔNG dời mã của nhóm khác.

    Mã bám theo ``stream`` của chính camera đó, nên các nhóm hoàn toàn độc lập — thêm một
    camera ccode không đụng gì tới config rule của tcode.
    """
    raw = yaml.safe_load(GC03.read_text(encoding="utf-8"))
    before = {c.key: c.code for c in load_crane(_write(tmp_path, raw), env={}).record_cameras}

    raw["cameras"]["ccode"].append(
        {"desc": "dự phòng", "stream": "rtsp://113.160.225.15:1599//CH001.sdp"}
    )
    after = load_crane(_write(tmp_path, raw), env={})

    for cam in after.record_cameras:
        if cam.key != "ccode6":
            assert before[cam.key] == cam.code, f"{cam.key} bị dời sang camera khác"


# ---------------------------------------------------------------- stream


def test_stream_with_a_credential_is_rejected() -> None:
    """URL trong config KHÔNG được mang ``user:pass`` — file này nằm trong git.

    Credential khai ở ``rtsp_credential`` cấp cẩu và lấy từ env. Tách như vậy để mọi thứ
    *định danh* luồng ở lại trong config: nhờ đó ``code`` suy được mà không cần env.
    """
    with pytest.raises(ValueError, match="chứa credential"):
        CameraConfig(
            role=CameraRole.CCODE,
            stream="rtsp://admin:secret@10.0.0.1:554/s",  # pragma: allowlist secret
        )


def test_stream_with_a_delimiter_is_rejected() -> None:
    """⚠️ Lỗi đã xảy ra thật: trích URL từ định dạng phân tách bằng `|` mà quên dừng.

    GStreamer KHÔNG báo lỗi — nó giữ phần thừa trong path và gửi
    `SETUP //CH001.sdp|h265|10|||`. Camera đang dùng tình cờ bỏ qua, nên nó chạy được và
    che mất lỗi cho tới khi gặp firmware khác.
    """
    with pytest.raises(ValueError, match="phân tách"):
        CameraConfig(role=CameraRole.CCODE, stream="rtsp://h:1508//CH001.sdp|h265|10|||")


def test_non_rtsp_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="rtsp://"):
        CameraConfig(role=CameraRole.CCODE, stream="http://h/s")


def test_unresolved_placeholder_is_rejected() -> None:
    with pytest.raises(ValueError, match="chưa nội suy"):
        CameraConfig(role=CameraRole.CCODE, stream="${CCODE1}")


def test_rtsps_is_accepted() -> None:
    assert CameraConfig(role=CameraRole.CCODE, stream="rtsps://h/s").stream.startswith("rtsps://")


def test_whitespace_is_trimmed_not_tolerated_inside() -> None:
    cam = CameraConfig(role=CameraRole.CCODE, stream="  rtsp://h/s  ")
    assert cam.stream == "rtsp://h/s"
    with pytest.raises(ValueError, match="phân tách"):
        CameraConfig(role=CameraRole.CCODE, stream="rtsp://h/s extra")


def test_credential_is_injected_only_when_connecting() -> None:
    """``stream`` là thứ trong git; ``rtsp_record`` là thứ đưa cho GStreamer."""
    cam = CameraConfig(role=CameraRole.CCODE, stream="rtsp://10.0.0.1:554/s", credential="u:p")
    assert cam.stream == "rtsp://10.0.0.1:554/s"
    assert cam.rtsp_record == "rtsp://u:p@10.0.0.1:554/s"  # pragma: allowlist secret


def test_config_loads_without_any_env() -> None:
    """⭐ Điểm chính: mã camera đọc được từ config, không cần biến môi trường nào.

    Trước đây URL nằm ở env, nên ``camera_code`` — thứ các service khác khoá config theo —
    không tái tạo được lúc review một diff hay chạy CI. Test này chạy với ``env={}``.
    """
    cfg = load_crane(GC03, env={})
    assert len(cfg.record_cameras) == 10
    assert all(c.code.startswith("GC03_") for c in cfg.record_cameras)
    assert cfg.rtsp_credential == "", "credential không được nằm trong config"


def test_declared_code_that_disagrees_with_the_stream_is_rejected(tmp_path: Path) -> None:
    """``code`` ghi trong file phải khớp URL. Sinh-rồi-kiểm: hiện được, nhưng không trôi được.

    Để lệch thì các service khác khoá config theo một mã không camera nào phát ra — và im
    lặng, vì "chưa có sự kiện" trông giống hệt "sai mã".
    """
    p = _write(tmp_path, _minimal(cameras={"ccode": [{"stream": "rtsp://h:1/s", "code": "SAI"}]}))
    with pytest.raises(ConfigError, match="code khai"):
        load_crane(p, env={})


def test_duplicate_camera_code_is_rejected(tmp_path: Path) -> None:
    """Hai camera cùng điểm cuối ⇒ cùng mã ⇒ dữ liệu bị gán nhầm, im lặng.

    Đã suýt xảy ra: fixture test dùng `rtsp://h/1`, `rtsp://h/2`… — khác path nhưng CÙNG
    host và không cổng, nên cả 10 camera ra cùng một mã.
    """
    p = _write(
        tmp_path,
        _minimal(cameras={"ccode": [{"stream": "rtsp://h/1"}, {"stream": "rtsp://h/2"}]}),
    )
    with pytest.raises(ConfigError, match="mã camera trùng"):
        load_crane(p, env={})


def test_a_camera_can_declare_its_own_source_fps() -> None:
    """⚠️ Đo 2026-09-02: camera ``..._1517`` phát **18 fps** trong khi mọi camera khác 30.

    Một giá trị chung cả cẩu không diễn đạt nổi điều đó, và hệ quả không chỉ là một con số
    xấu: ``drop_frame_interval`` suy từ nó, nên camera đó chạy 2,11 fps thay vì 3,33 (lệch
    40 %) và ``PerceptionMessage.fps`` báo sai 30 ra ngoài dây.
    """
    crane = load_crane(GC03, env=ENV)
    by_code = {c.code: c for c in crane.model_cameras}

    odd = by_code["GC03_113_160_225_15_1517"]
    assert odd.source_fps == 18.0
    assert odd.drop_frame_interval == 5, "18/3.3 làm tròn = 5, không phải 9 của 30 fps"

    others = [c for c in crane.model_cameras if c.code != odd.code]
    assert all(c.source_fps == crane.source_fps for c in others), "còn lại lấy mặc định của cẩu"
