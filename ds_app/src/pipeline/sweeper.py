"""Dọn segment cũ — theo **cả tuổi lẫn dung lượng**.

Không có nó thì nhánh ghi chạy tới khi đầy đĩa. Đo được 21,3 Mbps tổng ⇒ **9,6 GB/giờ**;
với yêu cầu 50 GB thì đĩa đầy sau khoảng 5 giờ, và lúc đó ghi hình dừng — mất bằng chứng.

**Vì sao phải có cả hai ngưỡng**, không chỉ một:

* Chỉ theo **tuổi** thì một camera bitrate bất thường vẫn làm đầy đĩa trong cửa sổ giữ.
* Chỉ theo **dung lượng** thì lúc bình thường ta giữ ít hơn cần thiết mà không biết.

Ràng buộc quan trọng nhất, và là chỗ dễ làm sai nhất:

⚠️ **Có một SÀN tuổi mà sweeper không bao giờ được vượt qua, kể cả khi đĩa đầy.**
``evidenced`` cắt clip lùi tới ``-35 s`` với ``delay`` 40 s, nên đoạn trẻ hơn ~5 phút vẫn
đang được cần tới. Một sweeper "giải phóng đĩa bằng mọi giá" sẽ xoá đúng những đoạn đó —
tức phá bằng chứng để cứu dung lượng, đổi một sự cố ồn ào lấy một sự cố im lặng. Khi vượt
ngân sách mà mọi thứ đều còn trẻ, đúng việc phải làm là **kêu to và không xoá**.

Xem ``docs/HARDWARE_BUDGET.md`` §2.6.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SweepPolicy", "SweepResult", "schedule", "sweep"]


@dataclass(frozen=True, slots=True)
class SweepPolicy:
    """Ngưỡng dọn dẹp.

    Args:
        max_age_sec: Xoá đoạn già hơn mức này. Mặc định 30 phút — cửa sổ để người vận hành
            còn xem lại được khi có sự cố (≈4,8 GB ở 30 fps).
        max_bytes: Trần dung lượng cứng cho toàn bộ thư mục ghi. Vượt thì xoá tiếp đoạn
            già nhất, kể cả khi chúng chưa tới ``max_age_sec``. Mặc định 20 GB, chừa biên
            trong yêu cầu 50 GB.
        min_age_sec: **Sàn không thể vượt.** Không bao giờ xoá đoạn trẻ hơn mức này, kể cả
            khi đã vượt ``max_bytes``. Mặc định 5 phút = cửa sổ xa nhất mà ``evidenced``
            còn cần (``-35 s`` + ``delay`` 40 s, cộng biên).
    """

    max_age_sec: float = 30 * 60
    max_bytes: int = 20 * 1024**3
    min_age_sec: float = 5 * 60

    def __post_init__(self) -> None:
        if self.min_age_sec >= self.max_age_sec:
            raise ValueError(
                f"min_age_sec ({self.min_age_sec}) phải nhỏ hơn max_age_sec "
                f"({self.max_age_sec}) — nếu không thì không bao giờ xoá được gì"
            )
        if self.max_bytes <= 0:
            raise ValueError(f"max_bytes phải dương, nhận {self.max_bytes}")


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Kết quả một lượt quét."""

    deleted: tuple[Path, ...] = ()
    freed_bytes: int = 0
    remaining_bytes: int = 0

    over_budget_bytes: int = 0
    """Còn vượt trần bao nhiêu SAU khi đã xoá hết những gì được phép.

    Khác 0 nghĩa là sàn ``min_age_sec`` đã chặn — đĩa đang đầy vì **ghi vào nhanh hơn mức
    giữ cho phép**, không phải vì sweeper lười. Nới ``min_age_sec`` sẽ phá bằng chứng; cách
    đúng là hạ bitrate/fps nguồn hoặc cấp thêm đĩa."""

    failed: tuple[tuple[Path, str], ...] = ()
    """Đoạn đáng xoá nhưng xoá không được, kèm lý do.

    ⚠️ Khác rỗng gần như luôn là **sai quyền**: tiến trình ghi chạy bằng user khác với
    tiến trình quét (ds_app chạy root trong container). Khi đó sweeper không xoá được gì,
    và nếu nuốt lỗi thì đĩa vẫn đầy trong im lặng — đúng thứ nó sinh ra để chặn."""

    @property
    def is_healthy(self) -> bool:
        """Lượt quét này có ổn không.

        Hai cách hỏng khác hẳn nhau, và phải phân biệt được: ``failed`` là **sai cấu
        hình** (sửa quyền), ``over_budget_bytes`` là **tín hiệu dung lượng** (hạ bitrate
        hoặc cấp thêm đĩa).
        """
        return self.over_budget_bytes == 0 and not self.failed


def sweep(
    root: str | Path,
    policy: SweepPolicy | None = None,
    *,
    now: float | None = None,
    keep_newest_per_dir: bool = True,
) -> SweepResult:
    """Quét một lượt trên cây thư mục ghi hình.

    Args:
        root: Thư mục gốc; cấu trúc ``<root>/<mã camera>/*.mp4`` (và ``*.mp4.part`` cho
            đoạn chưa chốt).
        policy: Ngưỡng; mặc định :class:`SweepPolicy`.
        now: Thời điểm coi là "bây giờ", epoch giây. Truyền vào để test.
        keep_newest_per_dir: Giữ lại file mới nhất của mỗi camera. Đó là đoạn **đang được
            ghi**; xoá nó thì ``splitmuxsink`` mất file dưới chân và đoạn hiện tại hỏng.

    Returns:
        :class:`SweepResult`. Kiểm ``is_healthy`` — ``False`` nghĩa là đĩa vượt trần mà
        không xoá thêm được nếu không phá bằng chứng.
    """
    policy = policy or SweepPolicy()
    now = time.time() if now is None else now
    base = Path(root)
    if not base.is_dir():
        return SweepResult()

    # (mtime, đường dẫn, kích thước) cho mọi segment, và tập file đang được ghi.
    entries: list[tuple[float, Path, int]] = []
    in_progress: set[Path] = set()
    for camera_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        newest: tuple[float, Path] | None = None
        # `*.mp4*` để bắt CẢ `.mp4.part`. Đoạn dở dang cũng chiếm đĩa, và một `.part` mồ
        # côi để lại sau khi tiến trình chết sẽ nằm đó mãi nếu không quét tới — mà nó luôn
        # là đoạn to nhất trong thư mục (chưa bị cắt).
        #
        # Đoạn ĐANG ghi cũng mang đuôi `.part`; nó được `keep_newest_per_dir` và `min_age`
        # bảo vệ, nên không cần phân biệt riêng.
        for f in camera_dir.glob("*.mp4*"):
            try:
                st = f.stat()
            except OSError:
                continue  # bị xoá giữa chừng bởi ai đó khác — không phải lỗi
            entries.append((st.st_mtime, f, st.st_size))
            if newest is None or st.st_mtime > newest[0]:
                newest = (st.st_mtime, f)
        if keep_newest_per_dir and newest is not None:
            in_progress.add(newest[1])

    entries.sort()  # già nhất trước
    total = sum(size for _, _, size in entries)

    deleted: list[Path] = []
    failed: list[tuple[Path, str]] = []
    freed = 0

    def _remove(path: Path, size: int) -> None:
        nonlocal freed
        try:
            path.unlink()
        except FileNotFoundError:
            return  # ai đó xoá trước; không phải lỗi
        except OSError as exc:
            # KHÔNG nuốt: nếu quét mà không xoá được thì đĩa vẫn đầy, và nguyên nhân
            # (thường là sai quyền) phải nhìn thấy được.
            failed.append((path, f"{type(exc).__name__}: {exc.strerror or exc}"))
            return
        deleted.append(path)
        freed += size

    # Lượt 1 — theo TUỔI. Đây là chế độ bình thường.
    for mtime, path, size in entries:
        if path in in_progress:
            continue
        if now - mtime <= policy.max_age_sec:
            break  # đã sắp xếp: những cái sau còn trẻ hơn
        _remove(path, size)

    # Lượt 2 — theo DUNG LƯỢNG, chỉ khi vẫn vượt trần. Vẫn tôn trọng sàn min_age_sec.
    remaining = total - freed
    if remaining > policy.max_bytes:
        for mtime, path, size in entries:
            if remaining <= policy.max_bytes:
                break
            if path in deleted or path in in_progress:
                continue
            if now - mtime < policy.min_age_sec:
                # Đã chạm sàn. Mọi thứ còn lại đều trẻ hơn (danh sách đã sắp xếp), nên
                # dừng hẳn thay vì quét tiếp vô ích.
                break
            _remove(path, size)
            remaining = total - freed

    return SweepResult(
        deleted=tuple(deleted),
        freed_bytes=freed,
        remaining_bytes=total - freed,
        over_budget_bytes=max(0, (total - freed) - policy.max_bytes),
        failed=tuple(failed),
    )


def schedule(
    Glib: Any,
    root: str | Path,
    policy: SweepPolicy | None = None,
    *,
    every_sec: int = 60,
    on_result: Callable[[SweepResult], None] | None = None,
) -> int:
    """Chạy :func:`sweep` định kỳ trên main loop của GLib.

    Đặt trong cùng tiến trình với nhánh ghi, không phải một service riêng: nó cần cùng
    quyền với tiến trình đang ghi file, và một sweeper chạy khác user sẽ ``PermissionError``
    trên mọi file (xem :attr:`SweepResult.failed`).

    ``every_sec`` mặc định 60: quét dày hơn không giúp gì — đoạn chỉ đóng mỗi ~10 s — mà
    mỗi lượt là một lần ``stat`` toàn bộ cây.

    Args:
        Glib: Module ``gi.repository.GLib``. Chuyền vào để module này không phải import
            ``gi``, và để test được không cần GTK.
        on_result: Gọi sau mỗi lượt. Mặc định chỉ in khi có gì bất thường — quét sạch mà
            log mỗi phút thì log sẽ toàn tiếng ồn và không ai đọc lúc cần.

    Returns:
        Id của nguồn GLib, dùng để huỷ.
    """

    def _tick() -> bool:
        result = sweep(root, policy)
        if on_result is not None:
            on_result(result)
        elif not result.is_healthy:
            if result.failed:
                print(  # noqa: T201
                    f"[sweep] ❌ {len(result.failed)} file xoá không được — "
                    f"{result.failed[0][1]}. Sweeper chạy khác user với tiến trình ghi?",
                    flush=True,
                )
            else:
                print(  # noqa: T201
                    f"[sweep] ⚠️ vượt trần {result.over_budget_bytes / 1e9:.1f} GB mà mọi "
                    f"đoạn còn trong cửa sổ bằng chứng — hạ fps nguồn hoặc cấp thêm đĩa, "
                    f"ĐỪNG nới min_age_sec",
                    flush=True,
                )
        return True  # giữ lịch

    return int(Glib.timeout_add_seconds(every_sec, _tick))
