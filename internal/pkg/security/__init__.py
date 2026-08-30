"""Bảo vệ model: giải mã ``.t7`` và giấy phép gắn phần cứng.

Ba module này đi cùng nhau vì cùng phục vụ một luồng duy nhất trong
``triton/modelsvc``: **kiểm giấy phép → giải mã model → dựng engine → xoá bản rõ**.

    fingerprint  →  license  →  cipher

``license`` ký/kiểm bằng Ed25519 trên vân tay do ``fingerprint`` thu thập; hỏng giấy phép
thì ``cipher`` không bao giờ được gọi. Xem ``docs/DESIGN_NOTES.md`` DN-005.
"""
