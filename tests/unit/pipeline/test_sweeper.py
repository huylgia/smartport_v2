"""Dọn segment: giữ đủ bằng chứng, và không bao giờ phá bằng chứng để cứu dung lượng."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from ds_app.src.pipeline.sweeper import SweepPolicy, sweep

NOW = 1_800_000_000.0
MB = 1024**2


def _segment(root: Path, cam: str, name: str, *, age_sec: float, size: int = MB) -> Path:
    d = root / cam
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.mp4"
    f.write_bytes(b"\0" * size)
    os.utime(f, (NOW - age_sec, NOW - age_sec))
    return f


# ---------------------------------------------------------------- theo tuổi


def test_old_segments_are_deleted(tmp_path: Path) -> None:
    old = _segment(tmp_path, "1", "cu", age_sec=3600)
    young = _segment(tmp_path, "1", "moi", age_sec=60)
    _segment(tmp_path, "1", "dang_ghi", age_sec=0)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1800), now=NOW)

    assert old in result.deleted
    assert young.exists()
    assert result.freed_bytes == MB


def test_nothing_deleted_when_all_young(tmp_path: Path) -> None:
    _segment(tmp_path, "1", "a", age_sec=60)
    _segment(tmp_path, "1", "b", age_sec=120)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1800), now=NOW)

    assert result.deleted == ()
    assert result.is_healthy


def test_sweeps_every_camera(tmp_path: Path) -> None:
    for cam in ("1", "4", "10"):
        _segment(tmp_path, cam, "cu", age_sec=3600)
        _segment(tmp_path, cam, "dang_ghi", age_sec=0)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1800), now=NOW)

    assert len(result.deleted) == 3
    assert {p.parent.name for p in result.deleted} == {"1", "4", "10"}


# ---------------------------------------------------------------- đoạn đang ghi


def test_the_file_being_written_is_never_deleted(tmp_path: Path) -> None:
    """File mới nhất của mỗi camera là đoạn ``splitmuxsink`` đang ghi dở.

    Xoá nó là rút file khỏi dưới chân muxer — đoạn hiện tại hỏng, và hỏng ngay lúc có sự
    kiện đang diễn ra.
    """
    _segment(tmp_path, "1", "cu", age_sec=99999)
    current = _segment(tmp_path, "1", "hien_tai", age_sec=99998)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1, min_age_sec=0), now=NOW)

    assert current.exists(), "không được xoá đoạn đang ghi dù nó đã quá tuổi"
    assert current not in result.deleted


def test_keep_newest_can_be_disabled_for_offline_cleanup(tmp_path: Path) -> None:
    """Dọn thủ công khi ds_app đã dừng thì không có đoạn nào đang ghi."""
    _segment(tmp_path, "1", "a", age_sec=99999)
    _segment(tmp_path, "1", "b", age_sec=99998)

    result = sweep(
        tmp_path, SweepPolicy(max_age_sec=1, min_age_sec=0), now=NOW, keep_newest_per_dir=False
    )

    assert len(result.deleted) == 2


# ---------------------------------------------------------------- theo dung lượng


def test_size_cap_deletes_oldest_first_even_when_within_age(tmp_path: Path) -> None:
    """Chỉ theo tuổi là không đủ: một camera bitrate bất thường vẫn làm đầy đĩa."""
    oldest = _segment(tmp_path, "1", "a", age_sec=1500, size=5 * MB)
    middle = _segment(tmp_path, "1", "b", age_sec=1400, size=5 * MB)
    newest = _segment(tmp_path, "1", "c", age_sec=1300, size=5 * MB)
    _segment(tmp_path, "1", "dang_ghi", age_sec=0, size=MB)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1800, max_bytes=8 * MB), now=NOW)

    assert oldest in result.deleted
    assert middle in result.deleted
    assert newest.exists(), "chỉ xoá vừa đủ để về dưới trần"
    assert result.is_healthy


def test_size_sweep_stops_as_soon_as_it_is_under_budget(tmp_path: Path) -> None:
    for i in range(5):
        _segment(tmp_path, "1", f"s{i}", age_sec=1500 - i, size=2 * MB)
    _segment(tmp_path, "1", "dang_ghi", age_sec=0, size=MB)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1800, max_bytes=9 * MB), now=NOW)

    assert result.remaining_bytes <= 9 * MB
    assert len(result.deleted) == 1, "xoá vừa đủ, không xoá thừa"


# ---------------------------------------------------------------- SÀN bằng chứng


def test_evidence_floor_blocks_deletion_even_when_over_budget(tmp_path: Path) -> None:
    """⚠️ Ràng buộc quan trọng nhất của file này.

    Vượt trần dung lượng mà mọi đoạn đều còn trẻ hơn ``min_age_sec`` ⇒ **không xoá gì**.
    ``evidenced`` vẫn đang cần chúng; xoá là đổi một sự cố ồn ào (đĩa đầy) lấy một sự cố im
    lặng (bằng chứng biến mất).
    """
    for i in range(5):
        _segment(tmp_path, "1", f"s{i}", age_sec=60 + i, size=5 * MB)

    result = sweep(tmp_path, SweepPolicy(max_age_sec=1800, max_bytes=MB, min_age_sec=300), now=NOW)

    assert result.deleted == (), "không được xoá đoạn bằng chứng còn cần"
    assert not result.is_healthy, "phải BÁO là còn vượt trần"
    assert result.over_budget_bytes > 0


def test_floor_still_allows_deleting_what_is_old_enough(tmp_path: Path) -> None:
    """Sàn chặn đoạn trẻ, không chặn đoạn đã đủ già."""
    old = _segment(tmp_path, "1", "du_gia", age_sec=600, size=5 * MB)
    _segment(tmp_path, "1", "con_tre", age_sec=60, size=5 * MB)
    _segment(tmp_path, "1", "dang_ghi", age_sec=0, size=MB)

    result = sweep(
        tmp_path, SweepPolicy(max_age_sec=1800, max_bytes=6 * MB, min_age_sec=300), now=NOW
    )

    assert old in result.deleted
    assert len(result.deleted) == 1


def test_over_budget_is_reported_not_hidden(tmp_path: Path) -> None:
    """``over_budget_bytes`` là tín hiệu vận hành: ghi vào nhanh hơn mức giữ cho phép.

    Nới ``min_age_sec`` để "sửa" nó là phá bằng chứng — cách đúng là hạ fps nguồn hoặc cấp
    thêm đĩa.
    """
    for i in range(4):
        _segment(tmp_path, "1", f"s{i}", age_sec=10 + i, size=10 * MB)

    result = sweep(tmp_path, SweepPolicy(max_bytes=5 * MB, min_age_sec=300), now=NOW)

    assert result.over_budget_bytes == 40 * MB - 5 * MB


# ---------------------------------------------------------------- biên


def test_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    """ds_app khởi động trước khi có đoạn nào — không phải lỗi."""
    result = sweep(tmp_path / "chua-ton-tai")
    assert result.deleted == ()
    assert result.is_healthy


def test_empty_tree_is_healthy(tmp_path: Path) -> None:
    (tmp_path / "1").mkdir()
    assert sweep(tmp_path).is_healthy


def test_non_mp4_files_are_left_alone(tmp_path: Path) -> None:
    """Chỉ đụng vào ``*.mp4``. Thư mục ghi hình có thể chứa thứ khác."""
    _segment(tmp_path, "1", "cu", age_sec=3600)
    other = tmp_path / "1" / "ghi-chu.txt"
    other.write_text("dung xoa")
    os.utime(other, (NOW - 99999, NOW - 99999))

    sweep(tmp_path, SweepPolicy(max_age_sec=1, min_age_sec=0), now=NOW)

    assert other.exists()


def test_floor_must_be_below_the_age_limit() -> None:
    """Sàn ≥ hạn tuổi thì không bao giờ xoá được gì — đĩa đầy trong im lặng."""
    with pytest.raises(ValueError, match="nhỏ hơn max_age_sec"):
        SweepPolicy(max_age_sec=300, min_age_sec=300)


def test_zero_size_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="phải dương"):
        SweepPolicy(max_bytes=0)


def test_defaults_match_the_documented_budget() -> None:
    """Mặc định phải khớp HARDWARE_BUDGET §2.6, không phải con số tuỳ hứng."""
    p = SweepPolicy()
    assert p.max_age_sec == 30 * 60, "30 phút = cửa sổ triage"
    assert p.min_age_sec == 5 * 60, "5 phút = cửa sổ evidence xa nhất (-35s + delay 40s)"
    assert p.max_bytes == 20 * 1024**3, "20 GB, chừa biên trong yêu cầu 50 GB"


def test_thirty_minutes_at_measured_bitrate_fits_the_cap() -> None:
    """Kiểm ngân sách tự nhất quán: giữ 30 phút ở 21,3 Mbps phải nằm dưới trần.

    Nếu không thì trần dung lượng sẽ luôn cắt trước hạn tuổi, và ``max_age_sec`` thành vô
    nghĩa mà không ai nhận ra.
    """
    p = SweepPolicy()
    bytes_per_sec = 21.3e6 / 8
    assert bytes_per_sec * p.max_age_sec < p.max_bytes


def test_delete_failure_is_reported_not_swallowed(tmp_path: Path) -> None:
    """⚠️ Sweeper xoá không được thì đĩa VẪN đầy — im lặng là mất luôn lý do tồn tại của nó.

    Xảy ra thật: file do container ghi (root), sweeper chạy bằng user thường ⇒ mọi lần xoá
    đều ``PermissionError``. Bản đầu nuốt ``OSError`` nên báo "xoá 0" y hệt như khi không
    có gì để xoá.
    """
    f = _segment(tmp_path, "1", "cu", age_sec=3600)
    _segment(tmp_path, "1", "dang_ghi", age_sec=0)
    (tmp_path / "1").chmod(0o555)  # thư mục chỉ đọc ⇒ không unlink được
    try:
        result = sweep(tmp_path, SweepPolicy(max_age_sec=1800), now=NOW)
    finally:
        (tmp_path / "1").chmod(0o755)

    assert result.deleted == ()
    assert len(result.failed) == 1
    assert result.failed[0][0] == f
    assert "Permission" in result.failed[0][1]
    assert not result.is_healthy, "xoá thất bại phải làm lượt quét KHÔNG healthy"


def test_a_vanished_file_is_not_a_failure(tmp_path: Path) -> None:
    """Ai đó xoá trước (dọn tay, log rotate) — bình thường, không phải sai cấu hình."""
    import unittest.mock

    _segment(tmp_path, "1", "cu", age_sec=3600)
    _segment(tmp_path, "1", "dang_ghi", age_sec=0)

    with unittest.mock.patch.object(Path, "unlink", side_effect=FileNotFoundError):
        result = sweep(tmp_path, SweepPolicy(max_age_sec=1800), now=NOW)

    assert result.failed == ()
    assert result.deleted == ()


# ---------------------------------------------------------------- lịch quét


class _FakeGLib:
    """``GLib`` giả: giữ callback lại để test gọi tay."""

    def __init__(self) -> None:
        self.scheduled: list[tuple[int, Callable[[], bool]]] = []

    def timeout_add_seconds(self, seconds: int, callback: Callable[[], bool]) -> int:
        self.scheduled.append((seconds, callback))
        return len(self.scheduled)


def _real_segment(root: Path, cam: str, name: str, *, age_sec: float, size: int = MB) -> Path:
    """Như ``_segment`` nhưng đặt tuổi theo giờ THẬT.

    ``schedule()`` gọi ``sweep()`` không truyền ``now``, nên nó dùng ``time.time()``. Dùng
    ``NOW`` giả ở đây thì file hoá ra nằm ở tương lai và không bao giờ bị coi là già.
    """
    now = time.time()
    d = root / cam
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.mp4"
    f.write_bytes(b"\0" * size)
    os.utime(f, (now - age_sec, now - age_sec))
    return f


def test_schedule_registers_a_repeating_timer(tmp_path: Path) -> None:
    from ds_app.src.pipeline.sweeper import schedule

    glib = _FakeGLib()
    schedule(glib, tmp_path, every_sec=60)

    assert glib.scheduled[0][0] == 60
    assert glib.scheduled[0][1]() is True, "phải trả True để GLib giữ lịch"


def test_scheduled_sweep_actually_deletes(tmp_path: Path) -> None:
    from ds_app.src.pipeline.sweeper import schedule

    old = _real_segment(tmp_path, "1", "cu", age_sec=3600)
    _real_segment(tmp_path, "1", "dang_ghi", age_sec=0)
    glib = _FakeGLib()
    schedule(glib, tmp_path, SweepPolicy(max_age_sec=1800))

    glib.scheduled[0][1]()

    assert not old.exists()


def test_quiet_when_healthy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quét sạch mà log mỗi phút thì log thành tiếng ồn và không ai đọc lúc cần."""
    from ds_app.src.pipeline.sweeper import schedule

    _real_segment(tmp_path, "1", "moi", age_sec=10)
    glib = _FakeGLib()
    schedule(glib, tmp_path)
    glib.scheduled[0][1]()

    assert capsys.readouterr().out == ""


def test_permission_failure_names_the_likely_cause(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Xoá không được thì phải chỉ thẳng nghi phạm số một, không chỉ in errno."""
    from ds_app.src.pipeline.sweeper import schedule

    _real_segment(tmp_path, "1", "cu", age_sec=3600)
    _real_segment(tmp_path, "1", "dang_ghi", age_sec=0)
    (tmp_path / "1").chmod(0o555)
    try:
        glib = _FakeGLib()
        schedule(glib, tmp_path, SweepPolicy(max_age_sec=1800))
        glib.scheduled[0][1]()
    finally:
        (tmp_path / "1").chmod(0o755)

    out = capsys.readouterr().out
    assert "khác user" in out


def test_over_budget_warns_against_lowering_the_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cảnh báo phải nói rõ cách sửa ĐÚNG — nới sàn là phá bằng chứng."""
    from ds_app.src.pipeline.sweeper import schedule

    for i in range(3):
        _real_segment(tmp_path, "1", f"s{i}", age_sec=10 + i, size=10 * MB)
    glib = _FakeGLib()
    schedule(glib, tmp_path, SweepPolicy(max_bytes=MB, min_age_sec=300))
    glib.scheduled[0][1]()

    assert "ĐỪNG nới min_age_sec" in capsys.readouterr().out
