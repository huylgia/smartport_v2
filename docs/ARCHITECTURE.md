# Kiến trúc smartport_v2

## Nguyên tắc

1. **Phụ thuộc chảy một chiều**: `services/` → `internal/` → `internal/pkg/`. (Thư mục
   entrypoint tên `services/`, **không phải `cmd/`** như kiểu Go: một package tên `cmd` ở
   mức trên cùng che module `cmd` của thư viện chuẩn, làm hỏng `pdb` và `breakpoint()` cho
   mọi thứ chạy trong repo.) Mọi I/O ra ngoài
   đi qua `gateway/contract/`. Business logic **không bao giờ** import Kafka/MinIO/Oracle
   trực tiếp.
2. **`internal/pkg/` là thuần**: không I/O, không state toàn cục, không config. Đây là lớp
   dễ test nhất và phải phủ ≥ 80 %. Nhóm theo miền, và `tests/unit/` phản chiếu đúng cấu
   trúc đó:

   ```
   internal/pkg/
   ├── vision/      pixel -> phát hiện/chữ; các module dính chặt nhau theo thứ tự một khung
   │                preprocess → [det] → dbpost → textcrop → [rec] → ctc   (nms cho PicoDet)
   │                ccode_pipeline ghép chúng lại thành đường ống mã container
   ├── security/    model đã mã hoá + giấy phép: fingerprint → license → cipher
   ├── ccode.py     chữ số kiểm tra ISO 6346  (nghiệp vụ)
   ├── geometry.py  vùng lane dạng đa giác     (không gian)
   ├── timebase.py  trục thời gian suy từ frame (thời gian)
   └── nptypes.py   bí danh kiểu mảng, dùng xuyên suốt
   ```

   Chỉ nhóm khi có **phụ thuộc thật** giữa các module, không nhóm theo cảm giác: ba file ở
   mức trên cùng không phụ thuộc gì và cũng không ai trong `pkg` dùng, nên gom chúng vào
   một thư mục `domain/` sẽ là gom-vì-loại-trừ, tệ hơn là để phẳng.
3. **Mỗi solution là một rule đăng ký được**; một tầng điều phối kết hợp signal của các rule.
4. **Đường realtime phải mỏng**: việc nặng (ffmpeg, upload, hậu xử lý DB) đẩy sang service
   khác hoặc Triton. Probe của DeepStream chỉ được đọc metadata.
5. **Config là dữ liệu có schema**, sinh từ pydantic, validate fail-fast lúc load.
6. **Thời gian suy từ frame**, không dùng wall-clock trên đường xử lý.

## Sơ đồ

```
 10 camera RTSP (live) — GC03: tất cả 2688×1520 HEVC; nhịp 30 fps trừ 3 camera (18/27/24)
      │  rtspsrc → rtph265depay → h265parse config-interval=-1
      ▼
┌───────────────── ds_app (DeepStream, container) ─────────────────┐
│  tee TẠI TẦNG BITSTREAM (trước decode)                            │
│   ├─► splitmuxsink → rec/<mã cam>/<epoch>.mp4                     │  ⭐ 0 NVENC
│   │            segment 10 s + sweeper · CẢ 10 camera              │     0 NVDEC thêm
│   └─► queue → nvv4l2decoder → nvstreammux                        │  ⚠️ CHỈ 8 camera
│         (cam 2 evidence-only và cam 9 bottom KHÔNG decode)        │
│         ├ crane  : nvinferserver(truckitems_pico)                 │
│         ├ tcode  : nvinferserver(truckhead) → nvinferserver(cls)  │──gRPC──► Triton
│         ├ ccode  : nvdspreprocess(ROI) → nvinferserver(ensemble)  │
│         └ bottom : không inference                                │
│       → src-pad probe (MỎNG) → kafka_sender (process riêng)       │
└──────────────────────────┬────────────────────────────────────────┘
        craneops.perception.{ccode,tcode,crane,bottom}
   ┌──────────────┬────────┴───────┬──────────────┐
   ▼              ▼                ▼              ▼
ruled@ccode   ruled@tcode     ruled@crane    ruled@bottom
[CCODE01,02]  [TCODE01]      [CRANE01..04]    [BOT01]
   └──────────────┴────────────────┴──────────────┘
                craneops.signals
                       ▼
              orchestratord
              CompositionEngine + Combinator + EventStore(SQLite)
           ┌───────────┴────────────┐
   craneops.evidence.fast/.slow   outbox
           ▼                        ▼
      evidenced                  syncd ──► craneops.manifest (compacted) ──► ruled
      → MinIO                    → dashboard API (retry+backoff)
```

## Vì sao tách ở tầng bitstream

RTX 3060 có **1 NVENC** và card GeForce giới hạn số phiên encode đồng thời (3–8 tuỳ
driver). Transcode 10 luồng để ghi hình sẽ vượt trần. Tách trước decode ⇒ nhánh ghi chỉ
mux H.265 gốc vào MP4: **0 phiên encode, 0 phiên decode thêm, ~0 % CPU**, giữ nguyên chất
lượng gốc để vẽ evidence.

⚠️ Camera không gửi VPS/SPS/PPS đầu kết nối (đã kiểm chứng — MP4 ghi bằng `-c copy` không
đọc lại được). Nhánh ghi **bắt buộc** dùng `h265parse config-interval=-1` để chèn tham số
bộ giải mã trước mỗi keyframe, nếu không mỗi segment sẽ không tự đứng được.

## Vì sao chỉ 8 trong 10 camera đi vào nhánh model

Cả 10 camera đều phát 2688×1520. Nhịp thì **không đồng nhất**: bảy camera 30 fps, còn
`..._1517` 18, `..._1516` 27, `..._1509` 24 — đo 2026-09-02 bằng `craneops-ds probe`
(HARDWARE_BUDGET §6.3). Lấy 30 fps cho cả mười là ước lượng **thiên cao**, và ở đây thiên
cao là an toàn: ngân sách NVDEC tính theo trần.

Nếu decode hết ở 30 fps thì tải NVDEC là **1 226 Mpixel/s ≈ 4,9× một luồng 4K30** — sát
hoặc vượt trần một NVDEC của GA106.

Cắt được hai camera ngay: **camera 2** không có vai trò xử lý nào (`CommonCamera.handle_frame`
chỉ gọi `super()` — không làm gì), và **camera 9** (soi đáy) không chạy model, ảnh mosaic
dựng từ segment ở `evidenced`. Cả hai chỉ cần nhánh ghi.

Còn 8 camera ⇒ 981 Mpixel/s ≈ 3,9× 4K30. Vẫn cao. Khuyến nghị chính là **đặt camera về
10 fps tại nguồn** (rule nhanh nhất cũng chỉ tiêu thụ 5 fps) ⇒
327 Mpixel/s ≈ 1,3× 4K30. Chi tiết và phương án dự phòng:
[HARDWARE_BUDGET.md §2.2–2.3](HARDWARE_BUDGET.md).

## Vì sao DB postprocess nằm trong Triton chứ không trong probe

Probe chạy trên luồng streaming của GStreamer. Hậu xử lý DB (`pyclipper` + `shapely`)
cho 6 camera ccode trên CPU 20 luồng sẽ nghẽn cả pipeline. Đưa xuống Triton Python
backend với `instance_group count: 3` ⇒ 3 process thật, không vướng GIL, và
`dynamic_batching` gom crop OCR của cả 6 camera vào một batch.

## Ba tầng config

Ranh giới theo đúng một câu hỏi: **ai đọc nó, và đổi nó thì phải dựng lại cái gì?**

| Tầng | Ở đâu | Nội dung | Đổi thì |
|---|---|---|---|
| Triton | `triton/repo/**/config.pbtxt` | hình dạng model, batching, instance | restart Triton |
| ds_app | `configs/cranes/<CẨU>.yaml` | đăng ký camera (URL→mã), vai trò, vùng cắt | restart pipeline |
| rule | `configs/rules/<CẨU>/<RULE>/config.json` | ngưỡng, vùng lane, cửa sổ thời gian | **hot-reload** |

**Một cẩu là một "epic"**: nó có bộ rule của nó, và mỗi rule có config riêng cho từng
camera của cẩu đó. Đăng ký URL camera ở tầng ds_app sinh ra `camera_code`; chính mã đó là
khoá của config rule.

```
configs/cranes/GC03.yaml           ccode1 … tcode2 … → GC03_113_160_225_15_1508 …
                                                              │
configs/rules/GC03/CCODE01/config.json   { "GC03_113_160_225_15_1508": {…} }
configs/rules/GC03/TCODE01/config.json   { "GC03_113_160_225_15_1510": {…} }
```

Khoá theo `camera_code` chứ không theo tên ngắn (`tcode1`) vì đó chính là chuỗi đến trên
`PerceptionMessage`: rule tra config bằng đúng field nó nhận được, không có bảng dịch nào
ở giữa để lệch.

⚠️ **Hệ quả:** khoá của config rule phải **tái tạo được mà không cần secret**. Vì vậy toàn
bộ định danh luồng — host, cổng, path — nằm trong `configs/cranes/*.yaml` dưới `stream`, và
chỉ credential ở env (`CRANEOPS_RTSP_CRED`). `load_crane(…, env={})` cho đúng mã production,
nên CI xác thực được config đã commit và người review một diff biết mã nào ứng với camera nào.

`code` được **sinh ra rồi kiểm lại**: `make codes` ghi nó vào YAML cạnh camera nó thuộc về,
và `load_crane` từ chối một `code` không khớp `stream`. Đọc được, nhưng không trôi được.
`make codes` cũng báo mọi config rule còn khoá theo mã cũ sau khi một camera đổi IP/cổng.

`ocr_rois` nằm **nguyên cụm** ở tầng ds_app, kể cả `lane` và `cont_dim`. Không tách, vì
hợp đồng message quyết định: `OcrResult` mang sẵn hai trường đó, nên probe của ds_app phải
biết chúng để điền — tách sang rule thì ds_app phải đọc ngược config của rule. Thứ duy nhất
ở tầng rule là `ocr_threshold`: nó là bộ lọc áp **sau** khi đọc, và đo trên v1 thấy cả 8
vùng dùng chung một giá trị, nên nó là tham số hiệu chỉnh chứ không phải thuộc tính của vùng.

`config.json` là ánh xạ phẳng và **mỗi camera đúng một dòng**, nên số camera được cấu hình
đếm được bằng mắt lẫn bằng `wc -l`. Vùng làn khai phẳng — `lane1_zone`, `lane2_zone`,
`lane3_zone` — vì `Lane` là tập đóng đúng ba giá trị và mỗi làn đúng một đa giác.

Thiếu config cho một camera là **lỗi**, không phải mặc định: config là một dict, camera
vắng mặt thì rule không xử lý nó — hệ chạy, log sạch, và camera ấy không bao giờ sinh
signal. `common/rule_config.py` đòi đủ mọi camera đúng vai trò và từ chối mã lạ;
`tools/rule_configs.py init <CẨU>` sinh khung để không ai phải chép mã tay.

## Ba registry

| Registry | Đăng ký cái gì | Dùng ở đâu |
|---|---|---|
| `register_rule` | Một solution nghiệp vụ (`CCODE01`, `CRANE03`…) | `services/ruled` |
| `register_combinator` | Chiến lược kết hợp signal (`majority_vote`, `manifest_crosscheck`…) | `orchestratord` |
| `register_probe` | Hàm probe của DeepStream, resolve theo tên từ config | `ds_app` |

Không có registry thì thêm một rule nghĩa là copy ~190 dòng
`services/<RULE>/main.py`, nhân với ~30 rule thành ~5 700 dòng gần như giống hệt nhau.
Ở v2, thêm rule = một file trong `internal/rules/` + một file config + một dòng
trong `rule_groups`.

## Gom rule theo topic

`CRANE01..04` cùng tiêu thụ một topic từ cùng camera 10. Chạy 4 process nghĩa là
deserialize cùng một message 4 lần. `services/ruled` host được **một nhóm rule**:

```yaml
rule_groups:
  ccode:  [CCODE01]
  tcode:  [TCODE01]
  crane:  [CRANE01, CRANE02, CRANE03, CRANE04]
  bottom: [BOT01]
```

⇒ 4 process thay vì 8, một lần deserialize. Vẫn tách được thành 8 nếu cần cô lập lỗi —
chỉ đổi config, không đổi code.

## Đóng gói

| Thành phần | Cách đóng gói | Vì sao |
|---|---|---|
| `ruled`, `orchestratord`, `evidenced`, `syncd`, `mediaapi`, `modelsvc` | **PyInstaller + systemd** | Python thuần; giữ license key + model mã hoá |
| `ds_app` | **Container** | Phụ thuộc `pyds` + GStreamer plugin registry + `libnvds_*` — PyInstaller không đóng gói nổi plugin registry |
| Triton | **Container** | Server C++ của NVIDIA |
| redpanda | Binary + systemd | Tương thích Kafka API, không JVM, không ZooKeeper |

## Tài liệu liên quan

- [MESSAGE_CONTRACT.md](MESSAGE_CONTRACT.md) — hợp đồng 7 topic
- [HARDWARE_BUDGET.md](HARDWARE_BUDGET.md) — ngân sách VRAM/CPU/đĩa và số đo
- [RULES.md](RULES.md) — đặc tả 8 rule
- [USE_CASES.md](USE_CASES.md) — quy trình nghiệp vụ nhập/xuất, map sang rule
- [DESIGN_NOTES.md](DESIGN_NOTES.md) — các quyết định đã chốt và lý do
- [RUNBOOK_TRITON.md](RUNBOOK_TRITON.md) — triển khai và vận hành Triton
- [RUNBOOK_DS_APP.md](RUNBOOK_DS_APP.md) — triển khai và vận hành ds_app
