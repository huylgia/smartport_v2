"""Đo nhịp nguồn lúc chạy — vì không đường nào khác nói cho ta biết."""

from __future__ import annotations

from ds_app.src.pipeline.ratecheck import MIN_SAMPLES, RateCheck, SourceRate


def feed(
    rc: RateCheck, code: str, *, real_fps: float, declared: float, interval: int, n: int
) -> list[SourceRate]:
    """Bơm ``n`` khung của một camera chạy ``real_fps`` thật."""
    warned: list[SourceRate] = []
    fid, ts = 0, 1_000_000.0
    for _ in range(n):
        odd = rc.observe(code, fid, ts, declared_fps=declared, drop_frame_interval=interval)
        if odd is not None:
            warned.append(odd)
        fid += interval
        ts += interval / real_fps
    return warned


def test_a_wrongly_declared_fps_is_caught() -> None:
    """⚠️ Đo thật trên GC03: camera ``..._1517`` chạy 18 fps trong khi config khai 30.

    Nó chỉ lộ ra sau **30 phút** chạy khi có người để ý bảng tổng kết — và hệ quả là
    ``drop_frame_interval`` sai 40 %, còn ``PerceptionMessage.fps`` báo 30 ra ngoài dây.
    """
    rc = RateCheck()

    warned = feed(rc, "cam", real_fps=18.0, declared=30.0, interval=5, n=20)

    assert warned, "khai 30 mà chạy 18 phải bị bắt"
    assert warned[0].measured is not None
    assert round(warned[0].measured, 1) == 18.0


def test_a_correct_declaration_stays_quiet() -> None:
    """Đối chứng. Không có nó thì một lớp báo-mọi-lúc cũng 'bắt' được mọi lỗi."""
    rc = RateCheck()

    assert feed(rc, "cam", real_fps=30.0, declared=30.0, interval=15, n=40) == []


def test_it_warns_once_per_camera_not_once_per_frame() -> None:
    """Cảnh báo mỗi khung sẽ nhấn chìm mọi thứ khác trong log của một tiến trình 24/7."""
    rc = RateCheck()

    assert len(feed(rc, "cam", real_fps=18.0, declared=30.0, interval=5, n=200)) == 1


def test_it_waits_for_enough_samples() -> None:
    """Vài khung đầu của nguồn RTSP còn đang ổn định nhịp; kết luận sớm là kết luận nhiễu."""
    rc = RateCheck()

    assert feed(rc, "cam", real_fps=18.0, declared=30.0, interval=5, n=MIN_SAMPLES - 1) == []
    rate = rc.rate("cam")
    assert rate is not None
    assert rate.measured is None


def test_an_outage_does_not_move_the_estimate() -> None:
    """Dùng trung vị chứ không dùng trung bình: một khoảng trống 30 s kéo trung bình đi rất
    xa trong khi trung vị không nhúc nhích. Đo được: mất mạng 30 s là chuyện thường."""
    rc = RateCheck()
    feed(rc, "cam", real_fps=30.0, declared=30.0, interval=15, n=40)

    # Một khoảng trống 30 giây, rồi nhịp bình thường trở lại.
    rc.observe("cam", 10_000, 1_000_100.0, declared_fps=30.0, drop_frame_interval=15)
    warned: list[SourceRate] = []
    fid, ts = 10_015, 1_000_100.5
    for _ in range(20):
        odd = rc.observe("cam", fid, ts, declared_fps=30.0, drop_frame_interval=15)
        if odd:
            warned.append(odd)
        fid += 15
        ts += 0.5

    assert warned == [], "một đợt mất mạng không được biến thành 'khai sai fps'"
    rate = rc.rate("cam")
    assert rate is not None and rate.measured is not None
    assert round(rate.measured, 1) == 30.0


def test_cameras_are_measured_independently() -> None:
    rc = RateCheck()
    feed(rc, "a", real_fps=30.0, declared=30.0, interval=15, n=30)
    feed(rc, "b", real_fps=18.0, declared=18.0, interval=5, n=30)

    got = {code: round(r.measured, 1) for code, r in rc.report() if r.measured}
    assert got == {"a": 30.0, "b": 18.0}
