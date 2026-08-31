# smartport_v2 — DaNang Smart Port CraneOps AI

Nhận dạng mã container và số đầu kéo cho cẩu bờ tại cảng Đà Nẵng: 11 camera RTSP →
OCR + định vị xe → dựng sự kiện thao tác cẩu → đối chiếu manifest Oracle → sinh ảnh/clip
bằng chứng → đẩy MinIO và dashboard.

Kiến trúc: **DeepStream** cho perception, **Triton** cho inference, **Kafka** làm bus,
một process cho mỗi nhóm rule, service evidence riêng cho việc nặng.

**Trạng thái: Phase 2 xong** — 9 model chạy trên Triton, đã đo hiệu năng và độ chính xác.
**Phase 3 đang làm** — nguồn camera và nhánh ghi hình passthrough đã xong và có test;
nhánh model (`nvstreammux` → `nvinferserver` → probe → Kafka) là phần còn lại.

## Bắt đầu

```bash
make setup                 # uv sync --extra dev + pre-commit
make check                 # ruff + mypy + pytest — chạy trước mỗi commit
make up                    # dựng Triton: license → giải mã → engine → serve
make status                # 9/9 model READY?
```

Cần `build/.env.triton` với `CRANEOPS_LICENSE_KEY`, `CRANEOPS_MODEL_PASSWORD`,
`CRANEOPS_ASSETS`. File này **không** nằm trong repo — không có secret nào nằm trong repo.

## Tài liệu

| | |
|---|---|
| [docs/USE_CASES.md](docs/USE_CASES.md) | **Quy trình nghiệp vụ nhập/xuất, map 1-1 sang rule, luồng bắn event** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Nguyên tắc, sơ đồ, 3 registry, cách đóng gói |
| [docs/MESSAGE_CONTRACT.md](docs/MESSAGE_CONTRACT.md) | Hợp đồng 7 topic Kafka |
| [docs/HARDWARE_BUDGET.md](docs/HARDWARE_BUDGET.md) | **Nguồn sự thật duy nhất cho mọi con số** — VRAM/CPU/đĩa, hiệu năng, độ chính xác |
| [docs/RULES.md](docs/RULES.md) | Đặc tả 8 rule |
| [docs/RUNBOOK_TRITON.md](docs/RUNBOOK_TRITON.md) | Triển khai và vận hành **Triton** |
| [docs/RUNBOOK_DS_APP.md](docs/RUNBOOK_DS_APP.md) | Triển khai và vận hành **ds_app** — hiện có tính năng `record` |
| [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) | 14 quyết định đã chốt, kèm lý do đủ để biết đổi nó sẽ phá cái gì |

## Bố cục

```
ds_app/       DeepStream pipeline (container — KHÔNG PyInstaller)
  src/pipeline/   elements (hằng số tinh chỉnh), sources (nvurisrcbin), recorder (ghi passthrough)
triton/
  repo/           9 model: 4 ccode (det/rec × h/v), 2 BLS ghép chuỗi, 2 pico, 1 cls
  bls/            Business Logic Scripting — det → hậu xử lý → crop → rec
  modelsvc/       Kiểm license → giải mã .t7 → dựng engine → xoá bản rõ
services/     Entrypoint từng service — CHỈ wiring
common/       config, logging, registry, message contract, service runner
gateway/      Adapter ra ngoài: bus, triton, minio, oracle, dashboard
internal/
  rules/          Mỗi solution = một rule đăng ký được
  orchestration/  Kết hợp signal thành sự kiện thao tác cẩu
  evidence/       Ảnh, clip, mosaic, upload
  pkg/            Thuần, không I/O — lớp dễ test nhất
    vision/         tiền xử lý, hậu xử lý DB, NMS, cắt ảnh, CTC
    security/       mã hoá model, vân tay thiết bị, giấy phép
configs/      YAML có schema sinh từ pydantic
deploy/       systemd unit, craneopsctl, watchdog
build/        Dockerfile + compose cho ds_app và Triton
tools/        export model, gấp tiền xử lý, benchmark, đo độ chính xác
tests/        unit / golden / integration
```

Phụ thuộc chảy **một chiều**: `services` → `internal` → `internal/pkg`; mọi I/O ra ngoài
đi qua `gateway`.

## Quy ước

- **Không `print()`** trong `common/`, `gateway/`, `internal/` — ruff `T20` chặn. Dùng loguru.
- **Không mutable default args** — ruff `B006`.
- **Không secret trong repo.** Secret đi qua `.env` / systemd `EnvironmentFile`.
- **Schema JSON và `config.pbtxt` được sinh ra, không viết tay.** `make check` fail nếu ai
  sửa nguồn mà quên regenerate.
- **Số đo chỉ ghi ở `HARDWARE_BUDGET.md`.** Chép sang nơi khác thì hai bản sẽ trôi khỏi nhau.
- Conventional Commits, scope theo service: `fix(ruled/ccode01): …`

## Máy dev ≠ máy đích

Máy dev là workstation 2× RTX 5090 / 48 luồng. Máy triển khai tại cảng là
**RTX 3060 / i7-12700 / 20 luồng**. Mọi benchmark trên máy dev là *giới hạn trên*.
Xem [docs/HARDWARE_BUDGET.md](docs/HARDWARE_BUDGET.md).
