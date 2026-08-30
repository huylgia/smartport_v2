"""Từ pixel ra phát hiện và chữ — thuần, không I/O, không biết gì về Triton.

Các module ở đây dính chặt vào nhau theo đúng thứ tự của một khung hình:

    preprocess  →  [model det]  →  dbpost  →  textcrop  →  [model rec]  →  ctc
                                      nms (nhánh PicoDet, thay cho dbpost)

``ccode_pipeline`` ghép chúng lại thành đường ống mã container. Nó nhận hai hàm suy luận
qua tham số, nên chạy được với model giả trong test và với BLS của Triton lúc thật. Đó là
**điểm trừu tượng hoá backend duy nhất** của hệ thống — hai tham số hàm, không phải một
tầng lớp Protocol/adapter. Đủ để test, và không phải trả giá cho một tầng chỉ có một bản
cài đặt thật.

Tách khỏi luồng streaming của GStreamer là quyết định kiến trúc, không phải sở thích:
xem ``docs/DESIGN_NOTES.md`` DN-007 và DN-010.
"""
