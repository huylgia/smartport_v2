"""Giải mã CTC cho recognizer SVTR.

Model trả về ma trận ``(L, C)``: L=25 vị trí, C=37 lớp (35 ký tự trong ``char_dict`` +
1 dấu cách + 1 lớp blank ở chỉ số 0). Giải mã CTC là: lấy argmax mỗi vị trí, bỏ ký tự
lặp liền nhau, bỏ blank.

⚠️ **Thứ tự các bước ở đây lệch so với CTC chuẩn, và đó là cố ý** — xem :func:`decode`
trước khi "sửa".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CtcConfig:
    """Ngưỡng giải mã.

    Mặc định lấy từ ``CCRecognizer``, KHÔNG phải
    từ ``OCRConfig`` — hai chỗ đó khác nhau (0.3 vs 0.4) và chỗ gọi thắng.
    """

    character_threshold: float = 0.3
    """Bỏ vị trí có xác suất argmax thấp hơn ngưỡng này."""

    score_threshold: float = 0.8
    """Điểm trung bình dưới ngưỡng ⇒ trả chuỗi rỗng (coi như không đọc được)."""

    remove_duplicate: bool = True


@dataclass(frozen=True, slots=True)
class CtcResult:
    text: str
    score: float


def load_char_dict(path: str | Path, *, use_space_char: bool = True) -> dict[int, str]:
    """Nạp ``char_dict``: một ký tự mỗi dòng, chỉ số bắt đầu từ **1** (0 là blank)."""
    lines = _read_lines(path)
    if use_space_char:
        lines.append(" ")
    return {i + 1: ch for i, ch in enumerate(lines)}


def _read_lines(path: str | Path) -> list[str]:
    from pathlib import Path as _Path

    raw = _Path(path).read_text(encoding="utf-8").split("\n")
    # Bỏ phần tử cuối nếu nó rỗng, giữ nếu không: file bảng ký tự có thể kết thúc bằng
    # newline hoặc không, và một dòng rỗng thừa sẽ làm lệch TOÀN BỘ chỉ số lớp.
    return raw[:-1] if not raw[-1] else raw


def restrict_to_groups(
    char_dict: Mapping[int, str], groups: Sequence[str]
) -> tuple[dict[int, str], list[int]]:
    """Giới hạn bộ ký tự về chữ số và/hoặc chữ cái.

    Dùng khi đã biết trước phần nào của mã container đang đọc: 4 ký tự đầu (owner code)
    luôn là chữ, 7 ký tự sau luôn là số. Thu hẹp không gian ký tự loại hẳn các nhầm lẫn
    kinh điển 0↔O, 1↔I, 5↔S, 8↔B.

    Returns:
        ``(char_dict mới đánh số lại từ 1, danh sách chỉ số cột GỐC cần giữ)``. Nơi gọi
        phải cắt ma trận logit theo ``[0, *chỉ_số_gốc]`` — 0 là cột blank, luôn giữ.
    """
    kept: dict[int, str] = {}
    for group in groups:
        if group == "digit":
            kept.update({k: v for k, v in char_dict.items() if v.isdigit()})
        elif group == "alphabet":
            kept.update({k: v for k, v in char_dict.items() if v.isalpha()})
        else:
            msg = f"nhóm ký tự không hợp lệ: {group!r} (chỉ có 'digit', 'alphabet')"
            raise ValueError(msg)

    original_indices = list(kept.keys())
    return {i + 1: v for i, v in enumerate(kept.values())}, original_indices


def decode(
    logits: NDArray[np.float32],
    char_dict: Mapping[int, str],
    cfg: CtcConfig,
) -> CtcResult:
    """Ma trận ``(L, C)`` → chuỗi.

    ⚠️ **Thứ tự các bước khác CTC chuẩn — cố ý.** CTC chuẩn bỏ ký tự lặp TRƯỚC rồi mới
    bỏ blank. Ở đây lọc theo ``character_threshold`` trước, rồi mới bỏ lặp trên chuỗi ĐÃ
    lọc. Hệ quả: hai ký tự giống nhau bị ngăn cách bởi một blank xác suất thấp sẽ dính
    lại thành một — ``"XX"`` đọc thành ``"X"``.

    Vì sao giữ: mọi ngưỡng đang chạy (``character_threshold``, ngưỡng bình chọn ở tầng
    rule) đã được hiệu chỉnh trên hành vi này. Với mã container, ca xấu cũng hiếm — ISO
    6346 không cho hai chữ cái giống nhau liền kề ở owner code. Đổi thứ tự sẽ đổi kết quả
    trên **toàn bộ** ROI, nên nếu muốn đổi: PR riêng, đo lại toàn bộ tập nhãn.
    """
    indices = logits.argmax(axis=-1)
    probs = logits.max(axis=-1)

    keep = probs > cfg.character_threshold
    indices, probs = indices[keep], probs[keep]

    selection = np.ones(len(indices), dtype=bool)
    if cfg.remove_duplicate:
        selection[1:] = indices[1:] != indices[:-1]
    selection &= indices != 0  # bỏ blank

    indices, probs = indices[selection], probs[selection]

    score = float(np.mean(probs)) if len(probs) else 0.0
    if score < cfg.score_threshold:
        return CtcResult(text="", score=score)

    try:
        text = "".join(char_dict[int(i)] for i in indices)
    except KeyError as exc:
        # Model xuất nhiều lớp hơn bảng ký tự có. Xảy ra khi thay model mà quên thay
        # char_dict — `KeyError: 5` trần thì mất hàng giờ mới lần ra nguyên nhân.
        raise ValueError(
            f"model xuất chỉ số ký tự {exc.args[0]} nhưng bảng ký tự chỉ có "
            f"{len(char_dict)} mục (khoá 1..{max(char_dict, default=0)}). "
            f"Nhiều khả năng model và char_dict không cùng một bộ."
        ) from None
    return CtcResult(text=text, score=score)
