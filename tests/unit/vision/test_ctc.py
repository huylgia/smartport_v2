"""Kiểm giải mã CTC cho recognizer SVTR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from internal.pkg.vision.ctc import CtcConfig, decode, load_char_dict, restrict_to_groups
from tests.unit.vision.conftest import CHAR_DICT, make_logits

# ---------------------------------------------------------------- giải mã CTC


def test_ctc_bo_blank_va_ky_tu_lap() -> None:
    # 'A'=11, 'B'=12 trong bảng bắt đầu từ 1
    got = decode(make_logits([11, 11, 0, 12, 12]), CHAR_DICT, CtcConfig(score_threshold=0.0))

    assert got.text == "AB"


def test_ctc_duoi_nguong_diem_tra_chuoi_rong() -> None:
    got = decode(make_logits([11, 12], prob=0.5), CHAR_DICT, CtcConfig(score_threshold=0.8))

    assert got.text == ""
    assert got.score == pytest.approx(0.5)


def test_ctc_loc_nguong_truoc_khi_bo_lap() -> None:
    """Lọc ngưỡng TRƯỚC khi bỏ lặp ⇒ 'AA' cách nhau bởi blank yếu bị dính thành 'A'.

    Lệch so với CTC chuẩn, và **cố ý** — mọi ngưỡng đang chạy đã hiệu chỉnh trên hành vi
    này. Nếu test đổ vì ai đó "sửa lỗi" thì phải đo lại toàn bộ tập nhãn trước khi nhận.
    """
    matrix = make_logits([11, 0, 11])
    matrix[1] = 0.01
    matrix[1, 0] = 0.2  # blank có xác suất thấp hơn character_threshold

    got = decode(matrix, CHAR_DICT, CtcConfig(character_threshold=0.3, score_threshold=0.0))

    assert got.text == "A", "CTC chuẩn cho 'AA'; ở đây cố ý cho 'A' — xem ctc.decode"


def test_restrict_to_groups_chi_giu_chu_so() -> None:
    narrowed, original = restrict_to_groups(CHAR_DICT, ["digit"])

    assert list(narrowed.values()) == list("0123456789")
    assert original == list(range(1, 11))


def test_restrict_to_groups_bao_loi_voi_nhom_la() -> None:
    with pytest.raises(ValueError, match="không hợp lệ"):
        restrict_to_groups(CHAR_DICT, ["chu-han"])


def test_load_char_dict_danh_so_tu_1(tmp_path: Path) -> None:
    path = tmp_path / "d.txt"
    path.write_text("A\nB\nC", encoding="utf-8")

    got = load_char_dict(path)

    assert got == {1: "A", 2: "B", 3: "C", 4: " "}


def test_chi_so_ngoai_bang_ky_tu_bao_loi_ro_rang() -> None:
    """Thay model mà quên thay char_dict ⇒ ``KeyError: 5`` trần, mất hàng giờ mới lần ra."""
    logits = np.zeros((3, 37), np.float32)
    logits[:, 5] = 10.0

    with pytest.raises(ValueError, match="không cùng một bộ"):
        decode(logits, {1: "A"}, CtcConfig(score_threshold=0.0))
