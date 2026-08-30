"""Kiểm tra mã container theo chuẩn ISO 6346.

Một mã hợp lệ gồm 11 ký tự: 4 chữ cái (3 ký tự chủ sở hữu + 1 ký tự loại thiết bị),
6 chữ số sê-ri, và 1 **chữ số kiểm tra** suy ra từ 10 ký tự trước. Chữ số đó tồn tại đúng
để bắt lỗi khi con người đọc nhầm hoặc máy nhận dạng sai.

Đây là tầng chặn cuối cùng của nhánh ccode. Đo được trên tập mẫu: hai lỗi OCR duy nhất
(`MRKU6904673` → `MRKU6934673`, `VNLU2092734` → `VKLU2092734`) **đều bị hàm này loại**,
nên chúng không sinh ra mã sai mà chỉ sinh ra lượt đọc bị từ chối — và tầng bình chọn theo
chuỗi chờ khung tiếp theo. Xem ``docs/HARDWARE_BUDGET.md`` §6.2.

"""

from __future__ import annotations

CODE_LENGTH = 11
"""4 chữ cái + 6 chữ số sê-ri + 1 chữ số kiểm tra."""

LETTER_VALUES = {
    # ISO 6346 gán A=10 rồi tăng dần, nhưng **bỏ qua mọi bội số của 11** (11, 22, 33…).
    # Đó là lý do bảng này không liên tục và phải viết ra thay vì tính bằng ord().
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18,
    "I": 19, "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27,
    "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36,
    "Y": 37, "Z": 38,
}  # fmt: skip


def check_digit(body: str) -> int:
    """Chữ số kiểm tra của 10 ký tự đầu.

    Mỗi ký tự đổi sang số rồi nhân với ``2^vị_trí``; tổng lấy dư 11, rồi lấy dư 10 — bước
    thứ hai để số dư 10 quy về 0, đúng như chuẩn quy định.
    """
    total = sum(
        (LETTER_VALUES[ch] if ch.isalpha() else int(ch)) * (2**i) for i, ch in enumerate(body)
    )
    return int(total) % 11 % 10


def is_container_code(code: str) -> bool:
    """``True`` nếu ``code`` là mã container hợp lệ theo ISO 6346.

    Không kiểm độ dài lẫn ký tự thì chuỗi rác sẽ làm nó ném
    ``IndexError``/``KeyError``. Ở đây kiểm trước và trả ``False`` — hàm này nằm trên
    đường xử lý đầu ra OCR, nơi chuỗi rác là chuyện bình thường chứ không phải lỗi.
    """
    if len(code) != CODE_LENGTH:
        return False
    body, last = code[:-1], code[-1]
    if not last.isdigit():
        return False
    if not all(ch.isalpha() or ch.isdigit() for ch in body):
        return False
    if any(ch.isalpha() and ch.upper() not in LETTER_VALUES for ch in body):
        return False

    return int(last) == check_digit(body.upper())
