"""Kiểm chữ số ISO 6346.

Các mã dùng ở đây là **mã thật đã xác minh**, không phải mã tự chế: bốn mã hợp lệ lấy từ
``assets/camera-containerNo/rec-containerNo/samples`` (tên file chính là nhãn), hai mã sai
là hai lỗi OCR **thật sự đã đo được** trên tập đó. Nhờ vậy test này neo vào chuẩn và vào
dữ liệu thật, chứ không neo vào chi tiết cài đặt.
"""

from __future__ import annotations

import pytest

from internal.pkg.ccode import check_digit, is_container_code

VALID = ["MRKU6904673", "VNLU2092734", "VSGU4240925", "MSNU7986097", "TLLU3464022", "MSMU1323180"]

# Hai lỗi OCR thật, đo được trên chính tập mẫu.
# Đây là bằng chứng tầng này bắt được lỗi model. Xem HARDWARE_BUDGET.md §6.2.
REAL_OCR_ERRORS = [("MRKU6904673", "MRKU6934673"), ("VNLU2092734", "VKLU2092734")]


@pytest.mark.parametrize("code", VALID)
def test_ma_that_deu_hop_le(code: str) -> None:
    assert is_container_code(code)


@pytest.mark.parametrize(("truth", "misread"), REAL_OCR_ERRORS)
def test_loi_ocr_that_bi_bat(truth: str, misread: str) -> None:
    """Đây là lý do tầng này tồn tại: chặn mã sai trước khi nó tới nghiệp vụ."""
    assert is_container_code(truth)
    assert not is_container_code(misread)


def test_doi_mot_chu_so_lam_hong_ma() -> None:
    """Chữ số kiểm tra phải nhạy với mọi vị trí, không chỉ vài vị trí."""
    base = "MRKU6904673"
    broken = 0
    for i in range(len(base) - 1):
        if not base[i].isdigit():
            continue
        digit = int(base[i])
        candidate = base[:i] + str((digit + 1) % 10) + base[i + 1 :]
        broken += not is_container_code(candidate)
    assert broken >= 5, "đổi một chữ số mà mã vẫn hợp lệ quá thường xuyên"


@pytest.mark.parametrize(
    "junk",
    ["", "U", "MAERSK", "45G1", "MRKU690467", "MRKU69046733", "MRKU690467X", "mrku6904673!"],
)
def test_chuoi_rac_tra_ve_false_chu_khong_nem_loi(junk: str) -> None:
    """Hàm này nằm trên đường xử lý đầu ra OCR — chuỗi rác là chuyện bình thường.

    Không kiểm độ dài lẫn ký tự thì hàm sẽ ném ``IndexError``/``KeyError`` với các đầu vào
    này (`ccode_utils.py:184`).
    """
    assert is_container_code(junk) is False


def test_chu_thuong_van_nhan_dien_duoc() -> None:
    assert is_container_code("mrku6904673")


def test_check_digit_khop_gia_tri_chuan() -> None:
    """Bảng chữ cái ISO 6346 bỏ qua bội số của 11 — dễ chép sai, nên neo lại."""
    assert check_digit("MRKU690467") == 3
    assert check_digit("VNLU209273") == 4
    assert check_digit("TLLU346402") == 2
