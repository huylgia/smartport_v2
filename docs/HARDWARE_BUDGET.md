# Ngân sách phần cứng

Tài liệu này ghi **số đo thật**, không phải ước lượng — và là **nguồn sự thật duy nhất**
cho các con số. `DESIGN_NOTES.md` ghi quyết định và lý do, rồi trỏ về đây; đừng chép bảng
số sang cả hai nơi.

Mỗi lần đổi pipeline hoặc model, chạy lại và cập nhật §6.1 (hiệu năng) / §6.2 (độ chính
xác). Lệnh đầy đủ ở §7; ngắn gọn thì `craneops-triton bench` và `craneops-triton accuracy`.

---

## 1. Hai máy khác nhau — đừng nhầm

| | Máy DEV (workstation) | Máy ĐÍCH (triển khai tại cảng) |
|---|---|---|
| GPU | 2 × **RTX 5090**, 32 607 MiB mỗi card | 1 × **RTX 3060 (GA106, Lite Hash Rate)** |
| VRAM | 65 GB tổng | **12 GB** ✅ xác nhận 2026-08-29 |
| CPU | AMD Threadripper 9960X, 24 core / **48 luồng** | Intel i7-12700, 12 core / **20 luồng** |
| RAM | 251 GB | ≥ 8 GB |
| OS | Ubuntu 24.04.4 | Ubuntu 22.04 |
| Driver | 580.173.02 (CUDA 13.0) | **≥ 535** (DeepStream 7.x yêu cầu) |

Máy dev mạnh hơn máy đích khoảng một bậc. Mọi benchmark trên máy dev là *giới hạn trên*.
Spike A/B/C **bắt buộc chạy trên máy đích**.

---

## 2. Nguồn vào thật — đo trực tiếp 2026-08-29

Đo bằng `ffmpeg -rtsp_transport tcp -c copy -f hevc`, 15 giây mỗi camera, trên cẩu **GC03**
(`smartport/assets/config.yaml`). Tất cả 10/10 camera kết nối được.

| id | config khai báo | **thực tế đo được** | Mbps | GB/ngày | vai trò | tên |
|---:|---|---|---:|---:|---|---|
| 1 | `720p`, `fps 10` | **2688×1520 @ 30** | 1,84 | 19,8 | ccode | Mặt phải trước |
| 2 | `720p`, `fps 10` | **2688×1520 @ 30** | 2,39 | 25,8 | *(chỉ evidence)* | Hông trái - Trước |
| 3 | `720p`, `fps 10` | **2688×1520 @ 30** | 2,52 | 27,2 | tcode | Đầu kéo - Lane 2 |
| 4 | `4MP`, `fps 10` | **2688×1520 @ 30** | 1,77 | 19,2 | ccode | Mặt trước - Lane 2 |
| 5 | `720p`, `fps 10` | **2688×1520 @ 30** | 2,42 | 26,1 | tcode | Đầu kéo - Lane 1 |
| 6 | `4MP`, `fps 10` | **2688×1520 @ 30** | 2,45 | 26,4 | ccode | Mặt trước - Lane 1 |
| 7 | `4MP`, `fps 10` | **2688×1520 @ 30** | 2,18 | 23,5 | ccode | Cửa sau - Lane 1 |
| 8 | `4MP`, `fps 10` | **2688×1520 @ 30** | 2,44 | 26,3 | ccode | Cửa sau - Lane 2 |
| 9 | `720p`, `fps 10` | **2688×1520 @ 30** | 0,80 | 8,7 | bottom | Soi đáy |
| 10 | `720p`, `fps 10` | **2688×1520 @ 30** | 2,51 | 27,1 | crane | Trần container |
| | | **Tổng** | **21,3** | **230** | | |

### 2.1 Ba điều chỉnh so với giả định ban đầu

**(a) 10 camera, không phải 11.** Con số 11 đến từ một danh sách ID cứng gộp **nhiều cẩu**
lại; GC03 thực tế không có camera 11. Đây là lý do vai trò camera phải nằm trong config
**của từng cẩu** thay vì một bảng dùng chung.

**(b) Camera 2 không có vai trò xử lý.** Nó chỉ nằm trong `SHORT_VIDEO_CAM_IDS`;
`CommonCamera.handle_frame` chỉ gọi `super()` — tức là **không làm gì**
Camera này phát 2688×1520@30 nhưng chỉ cần một
JPEG mỗi 5 giây. Ở v2 nó là **evidence-only: chỉ ghi passthrough, không decode**.

**(c) `resolution` và `fps` trong config KHÔNG phải thuộc tính nguồn.**
Đây là điều chỉnh quan trọng nhất. Cả 10 camera đều phát **2688×1520 @ 30 fps** giống hệt
nhau. Hai trường đó là *đích xử lý*:

* `resolution` → `videoscale` về `RESOLUTION[...]` (
  `720p = 1280×720`, `4MP = 2688×1520`) — tức camera khai `720p` đang bị **downscale
  2688×1520 → 1280×720**, còn camera khai `4MP` giữ nguyên.
* `fps` → `videorate` cap về 10.

Cả hai đều đặt **sau** `nvh265dec`, nên nếu để vậy thì **vẫn phải decode đủ 30 fps của cả 10 camera** rồi
mới vứt bớt. Đây là lãng phí lớn nhất trong pipeline hiện tại.

*(⚠️ Trường `fps` trong config camera rất dễ bị đọc sai thành hằng số cứng — một lỗi đã
xảy ra và không ai phát hiện suốt thời gian dài, vì giá trị khai báo tình cờ trùng mặc
định. Hệ quả: không cẩu nào đổi được fps qua config. Khi hiện thực Phase 3, hãy có test
khẳng định giá trị trong YAML thực sự tới được decoder.)*

### 2.2 ⛔ `drop-frame-interval` KHÔNG giảm tải NVDEC — đã chứng minh

Đây là câu hỏi quyết định của toàn bộ bài toán real-time, và cấu trúc GOP trả lời dứt điểm.

Đo `pict_type` của 150 khung đầu, cả 10 camera (2026-08-29):

| | I | P | B | GOP |
|---|---:|---:|---:|---:|
| Mỗi camera (150 khung) | 3 | 147 | **0** | **50** |
| Tổng 1 500 khung | 30 (2,0 %) | 1 470 (98,0 %) | **0 (0 %)** | |

Stream là **IPPP thuần, GOP 50, không có một khung B nào**.

Hệ quả: mỗi khung P tham chiếu khung ngay trước nó. Để lấy được khung thứ *N*, decoder
**bắt buộc** phải giải mã đủ các khung từ I-frame gần nhất đến *N*. Không tồn tại khung
non-reference nào để bỏ qua.

`drop-frame-interval` vì thế chỉ **vứt output sau khi đã giải mã**. Nó tiết kiệm
`nvstreammux`, inference, và copy buffer — **không** tiết kiệm NVDEC.

Điều đó không có nghĩa là bỏ nó đi: NVDEC không phải nút thắt duy nhất, và phần phía sau
decoder giảm đúng 6 lần khi hạ 30 fps xuống 5. Cấu hình qua `model_fps` — xem §2.7.

Lựa chọn duy nhất bỏ được decode thật là `skip-frames=2` (chỉ giải mã I-frame), nhưng
GOP 50 @ 30 fps ⇒ **0,6 fps**, trong khi rule chậm nhất (`tcode`) cũng cần 2 fps. Không
dùng được.

**Kết luận: ngân sách NVDEC phải tính theo 30 fps đầy đủ cho mọi luồng được decode.**

### 2.3 Bài toán real-time — tính lại

Đơn vị quy chiếu: 1 luồng 4K30 HEVC = 249 Mpixel/s · 1 luồng 1080p30 = 62,2 Mpixel/s.
Một camera smartport ở main-stream = 2688×1520×30 = **122,6 Mpixel/s**.

**Trước hết: model thực ra cần độ phân giải nào?**

Trường `resolution` trong config là đích `videoscale`, nên đây là độ phân giải mà model
model thật sự nhìn thấy — và nó **không đồng nhất**:

| Camera | Vai trò | Downscale về | Ghi chú |
|---|---|---|---|
| 1 | ccode | **1280×720** | ROI `[505,81,1115,662]` nằm trong khung 720p |
| 4, 6, 7, 8 | ccode | **2688×1520** | ROI cam 4 `[977,963,1710,1520]` — full res |
| 3, 5 | tcode | **1280×720** | |
| 10 | crane | **1280×720** | Lane line `1-351-1271-363` — khung 720p |

**Sub-stream có gì:** đã dò 13 đường dẫn RTSP. Camera chỉ có **hai** luồng thật —
main `2688×1520` và sub **`640×360`** (đường `h264/ch1/sub/av_stream`, cũng nhận
`hevc/ch1/sub/...` và `h264/ch01/sub/...`). **Không có mức trung gian 1280×720.**

Vậy sub-stream không phải phương án thay thế "không mất gì" như tôi viết ở bản trước:
640×360 chỉ bằng **một nửa mỗi chiều** so với 1280×720 mà model đang nhận.

**Ngân sách theo từng phương án:**

| # | Cấu hình | Luồng decode | Mpixel/s | ≈ 4K30 | ≈ 1080p30 |
|---|---|---|---:|---:|---:|
| 0 | Decode cả 10 camera | 10 main | 1 226 | 4,9× | 20 |
| **1** | **v2 cơ sở: bỏ cam 2 + cam 9** | **8 main** | **981** | **3,9×** | **16** |
| 2 | (1) + crane dùng sub 640×360 | 7 main + 1 sub | 865 | 3,5× | 14 |
| 3 | (2) + tcode dùng sub 640×360 | 5 main + 3 sub | 634 | 2,5× | 10 |
| 4 | (2) + gắn/tháo nguồn theo lane, 1 lane hoạt động | 2 main + 1 sub | 252 | 1,0× | 4 |

Cắt được cam 2 và cam 9 ngay vì cả hai chỉ cần nhánh ghi: `CommonCamera.handle_frame`
không làm gì, còn ảnh soi đáy dựng từ segment ở `evidenced`.

**Phương án 1 (981 Mpixel/s) có chạy nổi trên một NVDEC của GA106 không?**

Đây là con số phải đo, không phải suy đoán — và nó là **việc đầu tiên của Spike A**,
trước mọi thứ khác. Lý do phải nghiêm khắc: đây là bài toán real-time, không có lựa chọn
"chạy chậm lại". Nếu decode không theo kịp 30 fps × 8 luồng thì độ trễ tăng vô hạn hoặc
khung bị rơi ngay tại nguồn RTSP, và hệ thống trượt dần khỏi thời gian thực.

Tiêu chí đạt, đo trên **máy đích**, chạy liên tục ≥ 30 phút:

* `nvidia-smi dmon -s u` → cột `dec` **< 80 %** duy trì (biên cho lúc cao điểm).
* Độ sâu queue GStreamer ổn định, không tăng đơn điệu.
* Không có `QoS`/drop message trên bus.
* Chênh lệch `frame_ts` giữa hai message liên tiếp của cùng camera bằng đúng nghịch đảo
  nhịp phát của nó, không trôi.

**Nếu phương án 1 không đạt**, thứ tự áp dụng:

1. **Crane sang sub-stream** (−12 %). Camera 10 phát hiện vật thể lớn (đầu kéo, container).
   640×360 so với 1280×720 là giảm một nửa mỗi chiều — khả thi nhưng **là thay
   đổi hành vi**, phải qua golden test.
2. **Gắn/tháo nguồn theo lane** (−74 % ở trạng thái thường). Đòn bẩy lớn nhất. Chỉ camera
   `crane` chạy thường trực; `tcode` và `ccode` của một lane được gắn vào `nvstreammux`
   khi `CRANE01` phát `lane_active`, tháo ra khi lane tắt. Nhánh **ghi hình vẫn chạy liên
   tục cho cả 10 camera**, nên evidence không bao giờ mất — chỉ phát hiện trực tiếp bị trễ
   phần đầu. Đổi lại là độ phức tạp: DeepStream runtime source add/remove khá khó ổn định.
3. **Tcode sang sub-stream** — không khuyến nghị. Bộ phân loại số đầu kéo cần đọc được
   chữ số; ở 1280×720 vùng số đã nhỏ, xuống 640×360 gần như chắc chắn hỏng.
4. **Không bao giờ** dùng sub-stream cho `ccode`: ROI đọc mã container ở 2688×1520 xuống
   640×360 chỉ còn ~145×138 px.

Thiết kế config phải cho phép chuyển giữa các phương án **mà không đổi code**:
`rtsp_model` / `rtsp_record` tách riêng từng camera, và `attach_policy: always | on_lane_active`.

### 2.4 ⚠️ Camera không gửi VPS/SPS/PPS đầu kết nối — ảnh hưởng recorder

Khi capture, `ffmpeg` cảnh báo `[hevc] VPS 0 does not exist`, và file MP4 ghi bằng
`-c copy` **không đọc lại được** (`No start code is found. Invalid data found`).

Đây không phải lỗi vặt: thiết kế ghi hình của v2 là `tee → h265parse → splitmuxsink`
passthrough, và **mỗi segment phải tự chứa tham số bộ giải mã** thì `evidenced` mới cắt
clip được.

Việc cần làm trong `ds_app/src/pipeline/recorder.py`:

```
h265parse config-interval=-1     # chèn VPS/SPS/PPS trước MỖI keyframe
splitmuxsink max-size-time=10000000000 send-keyframe-requests=true
```

**Tiêu chí nghiệm thu:** lấy một segment bất kỳ ở giữa chuỗi ghi (không phải segment đầu),
`ffprobe` phải đọc được, và `ffmpeg -ss ... -to ...` phải cắt ra clip xem được.

### 2.5 NVENC — vì sao phải ghi passthrough

GA106 có **1 NVENC**, và card GeForce bị driver giới hạn số phiên encode đồng thời
(3–8 tuỳ phiên bản driver). 10 camera ghi đồng thời ⇒ nếu transcode sẽ vượt trần.

Tách **trước decode** ⇒ nhánh ghi chỉ mux bitstream gốc: **0 phiên encode, 0 phiên decode
thêm, ~0 % CPU**, và giữ nguyên chất lượng gốc để vẽ evidence.

**Tiêu chí nghiệm thu:** `nvidia-smi --query-gpu=encoder.stats.sessionCount --format=csv`
trả về **0** trong khi cả 10 camera đang ghi.

### 2.6 Đĩa cho segment recording

Đo được **21,3 Mbps = 2,66 MB/s = 9,6 GB/giờ** ở 30 fps. Nếu đặt camera về 10 fps thì
còn khoảng **3–5 GB/giờ**.

Cửa sổ evidence xa nhất là `-35 s` (camera đáy) với `delay: 40 s`. Job chạy ở `T+40` và
cần dữ liệu từ `T-35`, tức dữ liệu đã **75 giây tuổi** ngay lúc đọc — đó là sàn cứng
(`EVIDENCE_REACH_SEC` trong `ds_app/src/pipeline/sweeper.py`).

Giữ **6 đoạn 30 giây = 3 phút** mỗi camera ⇒ biên 105 s trên sàn, và **~0,55 GB** cho cả
10 camera (đo 2026-09-01: 55 MB một camera sau 4 phút chạy). Trước đây giữ 30 phút = 5,2 GB.

Đếm **số đoạn** chứ không đếm tuổi: độ dài đoạn dao động theo GOP, nên đếm tuổi cho ra số
đoạn khác nhau mỗi lúc. `SweepPolicy` từ chối mọi `min_age_sec` nằm giữa 0 và 75 s — một
con số như 30 trông đã cân nhắc nhưng vẫn xoá mất bằng chứng.

| Retention | @30 fps | @10 fps |
|---|---:|---:|
| 5 phút (đủ cho evidence) | 0,8 GB | ~0,35 GB |
| 30 phút (để triage) | 4,8 GB | ~2,1 GB |
| 2 giờ | 19 GB | ~8 GB |

Nằm gọn trong yêu cầu 50 GB. Sweeper chạy theo **cả dung lượng lẫn tuổi**, để một camera
bitrate bất thường không làm đầy đĩa — hiện thực ở `ds_app/src/pipeline/sweeper.py`, mặc
định giữ 30 phút / trần 20 GB.

⚠️ Có một **sàn tuổi** (mặc định 5 phút) mà sweeper không vượt qua kể cả khi đã đầy đĩa: đó
là cửa sổ `evidenced` còn cần. Vượt trần mà mọi đoạn đều trẻ hơn sàn ⇒ **không xoá gì** và
báo `over_budget_bytes`. Cách xử lý đúng là hạ fps nguồn hoặc cấp thêm đĩa, **không** phải
nới sàn — nới sàn là phá bằng chứng để cứu dung lượng.

### 2.7 fps xử lý mục tiêu theo vai trò

Suy từ nhịp mà mỗi vai trò thực sự cần, không phải suy đoán:

| Vai trò | Camera GC03 | Chu kỳ cần | fps mục tiêu |
|---|---|---|---:|
| ccode | 1, 4, 6, 7, 8 | `0.2` s | 5 |
| crane | 10 | `0.3` s | 3,3 |
| tcode | 3, 5 | `0.5` s | 2 |
| bottom | 9 | — | **không decode** |
| evidence-only | 2 | — | **không decode** |

Các con số này thực thi ở **decoder**, qua `model_fps` trong `configs/cranes/*.yaml`:
ds_app quy nó ra `drop-frame-interval` (§2.2). Chỉ camera chạy model mới khai.

⚠️ **Nó KHÔNG giảm tải NVDEC** — §2.2 đã chứng minh: nguồn IPPP nên mọi khung vẫn phải
giải mã, cái này chỉ vứt output *sau đó*. Nhưng NVDEC không phải nút thắt duy nhất, và
đây là chỗ **sớm nhất** bỏ được khung:

| Bỏ khung ở | Cắt được |
|---|---|
| **decoder** (`drop-frame-interval`) | nvstreammux, copy buffer, nvinferserver, probe, Kafka |
| probe trên src pad của mux | nvinferserver, Kafka — khung đã gộp batch và copy rồi |
| tầng rule (sau Kafka) | không gì trong ds_app |

Không giảm nhịp ở đâu cả thì tải suy luận là fps **nguồn** (30) chứ không phải fps mục
tiêu (5) — **gấp 6 lần**. Trần đo được 1 137 req/s (§6.1) là trên RTX 5090; trên 3060 nó
thấp hơn nhiều và chưa đo.

Giảm nhịp ở decoder là **trần tĩnh**. Cổng động (chỉ chạy suy luận khi lane có xe, theo
`lane_active` của `CRANE01`) là tầng thứ hai chồng lên, không thay thế — nó tiết kiệm thêm
ở lúc dưới cẩu trống, nhưng không cắt được phần muxing/copy.

---

## 3. VRAM — ngân sách dự kiến

| Hạng mục | Ước tính |
|---|---|
| 7 TensorRT engine FP16 + workspace | 2,5 – 4,0 GB |
| Triton runtime + CUDA context | ~1,0 GB |
| DeepStream: NVDEC surface pool + `nvstreammux` 8 nguồn 2688×1520 | 1,5 – 2,5 GB |
| Dự phòng | ~2,0 GB |
| **Tổng** | **7,0 – 9,5 GB** |

Máy đích là bản **12 GB** ⇒ vừa, còn biên ~2,5–5 GB.

⚠️ Surface pool nay phải tính theo **2688×1520**, không phải 1080p như ước tính đầu — mỗi
surface NV12 là ~6,1 MB thay vì ~3,1 MB. Đây là lý do cận trên tăng từ 9,0 lên 9,5 GB.

Ba thứ dễ ăn hết biên, phải theo dõi ở Spike A/B:

1. `instance_group` của `ccode_dbpost_{h,v}` — mỗi instance là một process Python riêng
   kèm CUDA context của chính nó.
2. `max_batch_size` của hai model `rec` — gom càng nhiều, workspace càng lớn.
3. `buffer-pool-size` của `nvstreammux` với 8 nguồn 2688×1520.

Nếu chạm trần, thứ tự cắt: giảm `instance_group` → giảm `max_batch_size` → nạp model
`h`/`v` theo nhu cầu thay vì giữ cả hai thường trực.

**Vì sao dùng engine chung:** một phiên suy luận riêng cho mỗi ROI tốn ~500 MB; 16 phiên cho MỘT camera là **~8 GB**
(một phiên bị ép `batch_size=1`; GC03 camera 1 khai 8 OCR config ⇒ 16 phiên det+rec). Chuyển sang engine dùng chung trên Triton là **điều kiện cần** để chạy
được trên phần cứng này.

---

## 4. CPU — phân bổ dự kiến (20 luồng)

| Thành phần | Luồng |
|---|---|
| ds_app: GStreamer streaming threads + probe (phải MỎNG) | 4 – 6 |
| ds_app: `kafka_sender` (process riêng) | 1 |
| Triton: `ccode_dbpost_{h,v}` Python backend, `instance_group count: 3` × 2 | ~6 |
| Triton: runtime + gRPC | ~2 |
| 4 × `ruled` + orchestratord + evidenced + syncd + media-api | ~4 (phần lớn chờ I/O) |
| redpanda | ~2 |

`evidenced` chạy ffmpeg là tải bùng phát ⇒ giới hạn bằng `asyncio.Semaphore` **và**
systemd `CPUQuota=`.

**Lý do DB postprocess không được chạy trong probe:** probe chạy trên luồng streaming của
GStreamer. Hậu xử lý DB (`pyclipper` + `shapely`) cho 5
camera ccode sẽ nghẽn cả pipeline. Đưa xuống Triton Python backend với
`instance_group count: N` ⇒ N process thật, không vướng GIL.

---

## 5. Driver — rủi ro chặn tiến độ

DeepStream 7.x yêu cầu **driver ≥ 535** và Ubuntu 22.04. Máy đích đang chạy **470 / CUDA 11.4**
⇒ **phải nâng driver**.

Driver NVIDIA tương thích ngược, nên ứng dụng CUDA 11.4 **vẫn chạy
được** trên driver 535+. **Phải kiểm chứng
thực tế (Spike C) trên máy staging trước khi nâng driver ở production.**

---

## 6. Số đo thực tế

| Ngày | Máy | Hạng mục | Số đo | Nguồn |
|---|---|---|---|---|
| 2026-08-29 | dev | VRAM máy dev | 2 × 32 607 MiB (RTX 5090) | `nvidia-smi` |
| 2026-08-29 | — | VRAM máy đích | **12 GB** | người dùng xác nhận |
| 2026-08-29 | dev | Nguồn RTSP GC03 | **10 × 2688×1520 @ 30 fps HEVC** | `ffmpeg -c copy`, 15 s/cam |
| 2026-08-29 | dev | Tổng bitrate | **21,3 Mbps = 9,6 GB/giờ** | như trên |
| 2026-08-29 | dev | Cấu trúc GOP | **GOP 50, IPPP, 0 khung B** (1 500 khung, 10 cam) | `ffprobe -show_entries frame=pict_type` |
| 2026-08-29 | — | `drop-frame-interval` giảm tải NVDEC? | **KHÔNG** (nhưng cắt 6× tải phía sau decoder) | §2.2 |
| 2026-08-29 | dev | Sub-stream khả dụng | **chỉ 640×360** (`h264/ch1/sub/av_stream`); không có 1280×720 | dò 13 đường dẫn RTSP |
| 2026-08-29 | dev | Tải decode v2 cơ sở (8 luồng main) | **981 Mpixel/s = 3,9× 4K30 = 16× 1080p30** | tính từ số đo |
| 2026-08-29 | dev | VPS/SPS/PPS đầu kết nối | **KHÔNG có** ⇒ cần `config-interval=-1` | `ffprobe` báo lỗi mp4 |
| 2026-08-29 | dev | Giải mã 6 model `.t7` | OK, `onnx.checker` pass hết | script kiểm chứng |
| ☐ | **đích** | **8 luồng main decode có đạt real-time?** | **việc đầu tiên của Spike A** | `nvidia-smi dmon -s u`, dec < 80 % |
| ☐ | đích | Trần phiên NVENC | | Spike A |
| ☐ | **đích** | 10 RTSP ghi passthrough, NVENC session | dev đã đo **0 %** (§6.3); còn phải xác nhận trên 3060 | Spike A |
| 2026-08-30 | dev | **VRAM Triton, 9 model** | **4 146 MiB** (4 model ccode ở FP32) | §6.1 |
| 2026-08-30 | dev | **Đường ống ccode — trần** | **1 137 req/s** (628 lúc đầu, +81 %) | §6.1 |
| 2026-08-30 | dev | **Đường ống ccode — tải thật 100 req/s** | **p50 3,8 ms · p95 5,3 ms** | §6.1 |
| 2026-08-30 | dev | Recognizer SVTR (batch động) | **22 096 mẫu/s** @batch 32 | §6.1 |
| 2026-08-30 | dev | Detector DB (nút thắt) | **684 mẫu/s** (h) · **1 364** (v) | §6.1 |
| 2026-08-30 | dev | TRT **FP16** trên det lẫn rec | **làm sai kết quả** ⇒ cả 4 model ccode dùng FP32 | DN-008, DN-013 |
| 2026-08-30 | dev | **Độ chính xác từng model so với nhãn** | cls 100 % · pico 98,7/97,8 % recall | §6.2 |
| 2026-09-01 | dev | **Ghi hình 10 camera — NVENC** | **0 %** | §6.3 |
| 2026-09-01 | dev | **Ghi hình 10 camera — NVDEC** | **0 %** (src pad không nối) | §6.3 |
| 2026-09-01 | dev | Ghi hình 10 camera — CPU container | **6,5 %** của 6 CPU (đỉnh 8,2 %) | §6.3 |
| 2026-09-01 | dev | Ghi hình 10 camera — RAM container | **313 MiB** / 6 GiB, 131 luồng | §6.3 |
| 2026-09-01 | dev | Ghi hình 10 camera — đĩa | **10,4 GB/giờ** cả cẩu | §6.3 |
| 2026-09-01 | dev | ⚠️ Nối `fakesink` vào src pad | NVDEC **0 % → 11,6 %** (decode 10 luồng để vứt) | §6.3 |
| 2026-09-01 | dev | `model_fps: 5` có thực thi không? | **có** — 5,0 fps vào probe, so với 27,5 fps khi không đặt | đếm `frame_num` |
| 2026-09-01 | dev | Tín hiệu `overrun` có nối đúng? | **có** — 225 lần/5 s khi sink cố ý chậm | RUNBOOK_DS_APP §4.2 |
| 2026-09-01 | dev | Ghi hình 10 camera — mất dữ liệu | **0** overrun, **0** nghi mất khung I | `record --cam all` |
| 2026-09-01 | dev | `kill -9` giữa đoạn — mất bao nhiêu? | **~3 s** cuối; đoạn đang mở vẫn ĐỌC ĐƯỢC | `reserved-moov-update-period-sec: 1` |
| 2026-09-01 | dev | Đoạn dở dang có phân biệt được? | **có** — đuôi `.mp4.part`, đổi tên nguyên tử khi đóng | `splitmuxsink-fragment-closed` |
| 2026-09-01 | dev | Giữ 6 đoạn 30 s | hội tụ đúng **6 + 1 `.part`** · **55 MB**/camera ⇒ ~0,55 GB cả cẩu | chạy 4 phút |
| 2026-09-01 | dev | Đoạn 30 s có ra đúng 30 s? | **30,00 s** (18 GOP chẵn) | `record --segment-sec 30` |
| 2026-09-01 | dev | Nguồn giờ cho đồng hồ clip | tên đoạn **0 s** · birthtime **+2 s** · mtime **+32 s** | DN-015 |
| 2026-09-02 | dev | Nội suy resize cho PicoDet | `CUBIC` vs `LINEAR`: recall **bằng nhau**, hộp lệch **tới 17,2 px** | §6.2 |
| 2026-09-02 | dev | BLS `craneops_crane`/`craneops_tcode` | hộp khớp đường trực tiếp **40/40** · phân loại gộp batch khớp gọi lẻ **59/59** | §6.2 |
| 2026-09-02 | dev | Gửi khung nguyên 12,26 MB qua gRPC | **0 khung trễ tới x8** tải mục tiêu · gãy ở x16 · GPU 9 % | §6.1 |
| 2026-09-02 | dev | Nhánh model chạy thật, crane + tcode | **333/333 khung, 0 bỏ, 0 lỗi** · 98 % nhịp đặt | §6.1 |
| 2026-09-02 | dev | Thứ tự kênh khung DeepStream | RGBA ⇒ phải đảo; sai kênh điểm tụt **0,932 → 0,507** | §6.2 |
| 2026-09-02 | dev | Tiền xử lý crop cho classifier số xe | BGR **100 %** / RGB 87,4 % · nội suy không đổi gì · hình học crop khớp | §6.2 |
| 2026-09-02 | dev | ds_app -> Kafka, 3 camera | **330 gửi / 330 ack, 0 mất** · trục thời gian lệch **0,0 ms** | §6.1 |
| ☐ | đích | Chạy lại toàn bộ §6.1 trên RTX 3060 | 5090 nhanh hơn 3060 khoảng 2–3× | Spike B |

---

### 6.1 Hiệu năng — đo 2026-08-30, trạng thái hiện hành

**⚠️ Đo trên máy dev (RTX 5090), KHÔNG phải máy đích (RTX 3060).** 5090 mạnh hơn khoảng
2–3x ở suy luận. Chia hệ số đó khi ước lượng cho máy đích, và phải đo lại thật ở Spike B.
Dùng để trả lời "có dư địa không", không phải để cam kết SLA.

Công cụ: `tools/bench/triton_bench.py`. Nó **đọc hợp đồng đầu vào từ chính Triton**
(`/v2/models/<tên>/config`) chứ không dùng bảng shape viết tay — hợp đồng đã đổi hai lần
trong dự án (DN-011, DN-012) và một bảng viết tay sẽ lỗi thời trong im lặng.

#### Tải mục tiêu — suy từ cấu hình thật

5 camera có ROI ccode (1, 4, 6, 7, 8), tổng **20 ROI** khai báo. Mỗi lane hoạt động chạy
khoảng một nửa số ROI chạy **song song**, ở ~5 fps sau decimate:

| Tình huống | Tải |
|---|---|
| Một lane hoạt động | **~50 request/giây** |
| Hai lane hoạt động | **~100 request/giây** |

#### Đường ống ccode đầu-cuối

| Phép đo | Kết quả |
|---|---|
| Tải thật **50 req/s** | đạt 49,8 · p50 **4,2 ms** · p95 5,9 · p99 7,9 |
| Tải thật **100 req/s** | đạt 99,4 · p50 **3,8 ms** · p95 5,3 · p99 6,4 |
| Trần (đập hết sức) | **1 137 req/s** (bão hoà ở 8 luồng) |
| **Dư địa** | **11,4x ở 100 req/s · 22,7x ở 50 req/s** |
| VRAM Triton, 9 model | **4 146 MiB** |

VRAM nằm trong khoảng dự kiến ở §3 (2,5–4 GB engine + ~1 GB runtime).

#### Từng model — 8 luồng đồng thời, tensor giả

| Model | Kiểu / hình dạng | batch=1 | Tốt nhất |
|---|---|---:|---:|
| `craneops_ccode_det_h` | UINT8 (640, 672, 3) | 684 mẫu/s | 684 @b1 |
| `craneops_ccode_det_v` | UINT8 (512, 576, 3) | **1 364** | 1 364 @b1 |
| `craneops_ccode_rec_h` | UINT8 (64, 256, 3) | 4 304 | **21 638** @b32 |
| `craneops_ccode_rec_v` | UINT8 (64, 256, 3) | 4 153 | **22 096** @b32 |
| `craneops_truckitems_pico` | FP32 (3, 416, 416) | 1 264 | 1 264 @b1 |
| `craneops_truckhead_pico` | FP32 (3, 416, 416) | 1 318 | 1 318 @b1 |
| `craneops_headcode_cls` | FP32 (3, 224, 224) | 806 | 1 819 @b4 |

**Nút thắt là detector**, không phải recognizer: trần đường ống (1 137 req/s) sát trần của
`det_h` (684 mẫu/s) cộng `det_v` — mỗi request chạy đúng một lần det. Cần thêm dư địa thì
tăng `instance_count` của hai model `det`, không phải của `rec`.

⚠️ **Batch lớn hơn KHÔNG phải lúc nào cũng nhanh hơn.** `det_h` tụt từ 684 xuống 499 mẫu/s
ở batch=2; hai model pico tụt từ ~1 300 xuống ~900. Lý do: `trt_profile` khai `optShapes`
ở batch 1, nên engine được tối ưu đúng điểm đó. Đấy là lựa chọn đúng (BLS gọi det từng ROI
một, DeepStream gọi pico từng khung), nhưng ai muốn gộp batch sau này **phải chỉnh profile
trước** rồi mới đo lại.

#### `dynamic_batching` — có chạy, nhưng chưa cần đến

Đập thẳng vào recognizer bằng nhiều client batch=1:

| Đồng thời | Thông lượng | Batch TB mỗi lần chạy GPU |
|---:|---:|---:|
| 1 | 1 165 req/s | 1,00 |
| 8 | 4 725 req/s | 3,09 |
| 32 | 7 133 req/s | **4,83** |
| 64 | 7 222 req/s | 4,87 |

Gom batch cho **6,1x thông lượng** khi có đủ tải. Nhưng ở đường ống thật, batch trung bình
đo được chỉ **1,00–1,03**: recognizer chỉ chiếm một phần nhỏ mỗi request (phần còn lại là
det + hậu xử lý DB), nên hàng đợi trước nó **không bao giờ hình thành**.

Kết luận: `dynamic_batching` là **dư địa**, không phải thứ đang gánh tải. Giữ bật vì miễn
phí khi rảnh.

#### Tham số thật sự quyết định: `instance_group` của model BLS

| `count` | Thông lượng | p50 |
|---:|---:|---:|
| 1 | 230 req/s | 52,9 ms |
| **3** | **610 req/s** | **11,9 ms** |

**2,7x.** Hậu xử lý DB chạy trên CPU và bị GIL chặn nên phải là nhiều **tiến trình** thật.
Đây chính là thứ bị mất khi `config.pbtxt` không tới được Triton (DN-009).

#### Hai tham số đã cân nhắc và **từ chối / chấp nhận**

| Thay đổi | Được | Mất | Quyết định |
|---|---|---|---|
| `instance_group count=2` cho hai model `det` | det 828 → 905 mẫu/s; đường ống 1 100 → 1 126 req/s (+2,3 %, trong sai số) | VRAM 4 146 → **5 456 MiB (+1,3 GB)** | ❌ **Từ chối.** Máy đích chỉ 12 GB và ngân sách cả hệ là 7–9 GB kể cả DeepStream |
| `model_warmup` cho mọi model TensorRT | phạt lần gọi đầu: rec 5,8 → **1,1x**, pico 2,8 → **1,7x**, det 5,1 → **3,4x** | không | ✅ **Nhận.** Cấu hình thuần, 0 chi phí lúc chạy |

Warmup không xoá hết phạt của `det`: phần dư một phần là chi phí thiết lập kết nối gRPC
lần đầu ở phía client, warmup không chạm tới được. Về giá trị tuyệt đối thì ~7 ms so với
2 ms — trong khung 200 ms là không đáng kể. Lợi ích thật nằm ở chỗ khác: Triton chỉ báo
`READY` sau khi warmup xong, nên `depends_on` của compose mới có nghĩa.

#### Chặng đường tối ưu — từng bước một

| | gốc | + gấp chuẩn hoá | + UINT8 NHWC | + FP32 det |
|---|---:|---:|---:|---:|
| Độ trễ (tuần tự) | 6,82 ms | 6,41 ms | 3,32 ms | **3,4 ms** |
| ├─ GPU (det + rec) | 1,50 | 1,57 | 1,22 | — |
| └─ Python | 4,61 | 3,90 | **1,50** | — |
| Trần | 628 req/s | 715 | 1 064 | **1 137** |
| VRAM | 3 696 MiB | 3 704 | 3 634 | **4 146** |
| Ghi chú | | DN-011 | DN-012 | DN-013 |

**Giảm 50 % độ trễ, tăng 81 % thông lượng** so với điểm xuất phát. Phần lớn đến từ việc
đưa tiền xử lý vào đồ thị model: Python từ 4,61 ms còn 1,50 ms.

---

#### Gửi khung nguyên qua gRPC — nhánh crane/tcode (2026-09-02, máy dev)

Hai model BLS nhận ảnh **thô** ``UINT8 [-1,-1,3]``, nên ds_app đẩy cả khung 2688x1520 =
**12,26 MB** mỗi lần. `enable_cuda_buffer_sharing` không dùng được vì Triton chạy container
riêng. Câu hỏi: phép chép này có phải nút thắt không.

| Phép đo | p50 | payload |
|---|---:|---:|
| Khung nguyên, BLS tự resize | 18,47 ms | 12,26 MB |
| Đã resize sẵn 416x416 | 1,83 ms | 0,52 MB |
| Resize 2688x1520 -> 416x416 (cv2, tại chỗ) | 0,04 ms | — |

Truyền chiếm **~16,6 ms**, tức 90 % thời gian; phép resize gần như miễn phí. Nhưng dưới tải
đồng thời thật (crane 3,3 fps + 2 camera tcode 2,0 fps) thì con số một-luồng đó không đúng:

| Nhịp | req/s | băng thông | p50 | p99 | khung trễ |
|---:|---:|---:|---:|---:|---:|
| **x1 (mục tiêu)** | 7,4 | 91 MB/s | 9,8-11,7 ms | 48-59 ms | **0** |
| x4 | 29,4 | 360 MB/s | 9,6-9,9 ms | 14-17 ms | 0 |
| x8 | 58,5 | 717 MB/s | 10,8-11,0 ms | 18-23 ms | 0 |
| x16 | 117,0 | 1 433 MB/s | 11,3-13,6 ms | 17-19 ms | 9 / 1 057 |
| x24 | 163,2 | 2 000 MB/s | 13,5-15,8 ms | 23-26 ms | **bão hoà** |

**Kết luận: giữ nguyên thiết kế.** Ở tải mục tiêu, Triton dùng 0,63 nhân *ở x8* — tức ~0,08
nhân ở x1 — và GPU chỉ 9 % ngay lúc x24. Biên gãy nằm ở **x16**, và nút thắt là CPU
(serialize gRPC + 2 tiến trình BLS), không phải GPU. Máy đích i7-12700 có nhân chậm hơn
Threadripper 9960X của máy dev cỡ 2 lần, nên vẫn còn khoảng **x8** dư địa.

p99 ở x1 (48-59 ms) **cao hơn** ở x8 (18-23 ms): đó là hiệu ứng cache nguội giữa các khung
thưa, không phải xếp hàng. Đừng đọc nó thành dấu hiệu quá tải.

⚠️ **Đừng "tối ưu" bằng cách đẩy phép resize sang GPU.** `nvinferserver` có
`frame_scaling_filter` = `NvBufSurfTransformInter_Algo1` (GPU-Cubic), nghe như tương đương
`cv2.INTER_CUBIC` nhưng là bản cài đặt khác. §6.2 đo được rằng đổi phép nội suy dịch hộp
**tới 17,2 px** trong khi recall không đổi — tức bảng số tổng sẽ không báo gì. Muốn đổi thì
phải đo lại từng hộp trước.

#### Nhánh model đầu-cuối — crane + tcode (2026-09-02, máy dev)

Ba camera thật qua RTSP, 46 giây, model BLS trên Triton:

| Camera | Role | Khung | Đạt nhịp đặt |
|---|---|---:|---:|
| `..._1517` | crane | 151 | 3,27 / 3,3 fps (98,1 %) |
| `..._1510` | tcode | 91 | 1,97 / 2,0 fps (98,5 %) |
| `..._1512` | tcode | 91 | 1,97 / 2,0 fps (98,5 %) |

**333 gửi, 333 xong, 0 bỏ, 0 lỗi.** Chỉ số khung nhảy đúng bước decimate và dấu thời gian
cách đều, nên `restore_frame_id` + neo PTS hoạt động đúng trên nguồn thật.

⚠️ **Đừng chạy role chưa có model BLS.** `CameraRole.runs_model` nói role đó *rốt cuộc* sẽ
chạy model; `BLS_FOR_ROLE` nói hôm nay đã có model chưa. Trộn hai câu đó lại thì `ccode`
lọt vào `--role all`: đo được **1 503 lỗi trong 60 giây** (5 camera decode để mỗi khung ném
`KeyError`), trong khi hai role kia vẫn chạy đúng nên bảng tổng kết trông gần như bình
thường.

#### ds_app → Kafka (2026-09-02, máy dev)

Ba camera live, 46 giây, Redpanda một node:

```
gửi 330  xong 330  BỎ 0  LỖI 0  LỖI-NHẬN 0
kafka: xếp 330  broker ack 330  BỎ 0  LỖI 0  còn bay 0
```

815 message đọc ngược lại từ topic đều hợp lệ theo `PerceptionMessage`, khoá phân vùng
đúng bằng `camera_code`, và `frame_ts` khớp `start_ts + frame_id / fps` **lệch 0,0 ms**.

Hai chỗ mất message im lặng đã bịt, cả hai đều chỉ lộ ra khi chạy thật:

* **Metadata chưa nạp.** `max_block_ms` thấp biến việc chờ metadata cluster thành lỗi, nên
  **2 message đầu mỗi lần chạy** biến mất. `BusProducer.start()` nạp sẵn metadata cho đúng
  các topic sắp dùng — chặn lúc khởi động thì được, chặn trong `publish()` thì không.
* **Callback ném thì giết worker.** `on_result` nằm ngoài `try` của worker, nên một lỗi ở
  nơi nhận làm thread chết lặng lẽ: `completed` vẫn tăng tới lúc thread cuối chết rồi mọi
  thứ dừng hẳn. Nay nó nằm trong `try`, đếm riêng (`sink_failed`) và **giữ lại thông báo
  lỗi đầu tiên** — một bộ đếm nói rằng có hỏng, nó không nói hỏng ở đâu.

⚠️ `start_ts` là thời điểm của **khung `frame_id == 0`**, không phải gốc PTS của nguồn.
Lấy nhầm mốc làm `start_ts + frame_id/fps` lệch `frame_ts` một khoảng cố định bằng PTS của
khung đầu (đo được **0,475 s**) — nhịp vẫn đúng nên không có gì báo, chỉ là mọi cửa sổ thời
gian trượt đi nửa giây.

### 6.2 Độ chính xác — đo 2026-08-30

Công cụ: `tools/golden/accuracy.py` (so với **nhãn**), `tools/golden/parity_stages.py` và
Đo lại sau mỗi lần đổi model.

#### So với nhãn

| Model | Tập | n | Chỉ số | Kết quả |
|---|---|---:|---|---:|
| `craneops_headcode_cls` | `cls-truckHead/samples` | 451 | top-1 | **100,0 %** |
| `craneops_ccode_rec_h` | `rec-containerNo/samples/h` | 4 | khớp chuỗi | 50,0 % ¹ |
| `craneops_ccode_rec_v` | `rec-containerNo/samples/v` | 3 | khớp chuỗi | **100,0 %** |
| `craneops_truckitems_pico` | `det-truckItems/samples` | 111 | recall / prec @IoU 0,5 | **98,7 % / 98,7 %** |
| `craneops_truckhead_pico` | `det-truckHead/samples` | 100 | recall / prec @IoU 0,5 | **97,8 % / 99,3 %** |
| `craneops_ccode_det_{h,v}` | — | — | — | **không có tập dữ liệu** |

#### Thứ tự kênh khung lấy từ DeepStream (2026-09-02)

`pyds.get_nvds_buf_surface()` trên caps `format=RGBA` trả **RGB** ở ba kênh đầu, nên phải
đảo thành BGR trước khi đưa vào model. Kiểm bằng mắt trên một khung camera cẩu: bản đã đảo
cho cẩu vàng / thân tàu xanh / áo bảo hộ cam; bản không đảo cho cẩu xanh lơ, container tím,
áo bảo hộ xanh.

Cùng khung đó qua `craneops_truckitems_pico`:

| | số vật | điểm cao nhất |
|---|---:|---:|
| Đúng kênh (BGR) | 3 | **0,932** |
| Sai kênh (RGB) | 3 | 0,507 |

Sai kênh **vẫn ra đúng số vật** — nên không có gì báo. Nhưng `Crane01Config.head_thresh`
là 0,6, nên 0,507 bị loại sạch và `CRANE01` không gán được lane nào.

#### Tiền xử lý crop cho `craneops_headcode_cls` (2026-09-02)

Nhánh `tcode` cắt hộp đầu kéo rồi đưa vào classifier, nên crop phải khớp phân bố huấn
luyện ở ba mặt. Đo cả ba trên 451 ảnh có nhãn + hộp thật từ camera live:

| Mặt | Kết quả |
|---|---|
| **Thứ tự kênh** | **BGR 100,00 %** · RGB 87,36 % ⇒ model đã fold phép đảo kênh, nhận BGR |
| **Nội suy** | `LINEAR` / `CUBIC` / `AREA` đều **100 %**, điểm trung bình lệch 0,0001 ⇒ không nhạy |
| **Hình học crop** | hộp live ~405x378 px, tỉ lệ 1,06-1,16 · crop huấn luyện trung vị 384x387, tỉ lệ 1,01 (dải 0,65-1,85) |

Khác PicoDet: ở đó đổi nội suy dịch hộp tới 17,2 px, còn classifier này thì không phân biệt.
Đừng suy luật của model này sang model kia.

⚠️ Comment trong `tools/export_models.py` từng ghi input là **RGB**. Nó mô tả bản ONNX
*trước* khi gấp — `--fold-preprocess` gấp luôn phép đảo kênh, nên bản đang phục vụ nhận
ngược lại. Hai câu mâu thuẫn nằm cách nhau 10 dòng trong cùng một `ModelSpec`.

Với xe vào đúng khung, số xe đọc ra **0,99-1,00** trên cả hai camera tcode — tức ngưỡng
`head_code_thresh` 0,93 là phù hợp, không quá chặt.

Hai model pico đo lại 2026-09-02 với `INTER_CUBIC` (phép nội suy chúng được huấn luyện
với) thay cho `INTER_LINEAR` mà công cụ dùng nhầm trước đó. **Recall không đổi** ở cả hai;
precision đổi hai chiều ngược nhau, mỗi bên đúng một hộp — tức nhiễu.

Điều số tổng KHÔNG cho thấy, và là lý do thật để đổi: cùng một ảnh, hai phép nội suy cho
hộp **lệch nhau tới 17,2 px** (`truckitems`) và 14,5 px (`truckhead`), trên 138 cặp hộp mà
cả hai đều tìm ra cùng số lượng và cùng lớp. `CRANE01` gán lane bằng điểm mốc của chính
hộp đó, nên sai lệch cỡ này lật được phán quyết ở sát biên vùng.

¹ Hai lỗi trên bốn ảnh (`MRKU6934673`, `VKLU2092734`), **cả hai bị `is_container_code` loại**
(mã kiểm tra ISO 6346) nên không lọt xuống nghiệp vụ — chúng thành lượt đọc bị từ chối, và
tầng bình chọn chờ khung tiếp theo. Với n=4 thì con số này không có ý nghĩa thống kê.

#### ORT FP32 ↔ engine TensorRT

| | FP16 | **FP32 (đang dùng)** |
|---|---:|---:|
| bitmap detector, lệch trung vị | 3,4e-02 | **9,7e-03** |
| bitmap detector, lệch lớn nhất | **0,198** | **0,078** |
| lệch đầu-cuối @0,95 | 15/134 | **3/129** |

Ngưỡng quyết định của hậu xử lý DB là `bitmap_threshold=0.1` / `box_threshold=0.2`; FP16
lệch tới 0,198 — đủ để lật một hộp. Ba ca còn lệch đều là chuỗi rác (`'U'`, `'U'`,
`'NAER'`), không phải mã container hợp lệ. Xem DN-013.

---

## 7. Lệnh quan sát

```bash
# VRAM, NVDEC, NVENC theo thời gian thực — NVENC phải bằng 0
nvidia-smi dmon -s pucm

# Số phiên encode đang mở
nvidia-smi --query-gpu=encoder.stats.sessionCount,encoder.stats.averageFps --format=csv -l 2

# Đo lại nguồn RTSP (tái chạy khi đổi cấu hình camera)
uv run python -m tools.bench.probe_sources --config <crane>.yaml

# Hiệu năng Triton (§6.1) — công cụ tự đọc hợp đồng model từ /v2/models/<tên>/config
uv run --with "tritonclient[grpc]" python -m tools.bench.triton_bench --all
uv run --with "tritonclient[grpc]" python -m tools.bench.triton_bench --pipeline --rps 100

# Độ chính xác từng model so với NHÃN (§6.2)
craneops-triton accuracy

```

---

### 6.3 Ghi hình 10 camera (2026-09-01, máy dev)

`craneops-ds record --cam all --duration 90 --segment-sec 10`, GC03, 10 camera
2688×1520@30 HEVC.

| Hạng mục | Số đo |
|---|---|
| Đoạn ghi | 90 đoạn / 90 s, **0 lỗi**, 10/10 camera đều ra file |
| Độ dài đoạn thật | **10,00 s** ở 9/10 camera (học từ hai mốc mở liên tiếp) |
| NVENC | **0 %** |
| NVDEC | **0 %** |
| SM | **0 %** |
| CPU container | **6,5 %** của 6 CPU (đỉnh 8,2 %) |
| RAM container | **313 MiB** / 6 GiB · 131 luồng |
| Đĩa | **10,4 GB/giờ** cả cẩu · 0,31–1,44 GB/giờ mỗi camera |

GPU **hoàn toàn không đụng tới**. Đó là điều kiện sống còn trên máy đích: cả ngân sách
NVDEC ở §2.2 giả định nhánh ghi không tốn gì, và giờ nó được đo chứ không còn là suy đoán.

Đĩa: 10,4 GB/giờ đo được, so với **9,6 GB/giờ** suy từ bitstream nguồn (§2.6) — lệch 8 %
do phần bao mp4. Giữ 30 phút ⇒ ~5,2 GB, nằm gọn trong yêu cầu 50 GB.

#### ⚠️ Đừng nối gì vào src pad của nguồn

`nvurisrcbin` có decoder bên trong. Nhánh ghi tách **trước** nó, nên khi src pad không nối,
luồng sau decode dừng ngay ở buffer đầu (`NOT_LINKED`) và decoder không chạy.

Đo được hai chiều, cùng một bản ghi 10 camera:

| src pad | NVDEC | ghi hình |
|---|---|---|
| **không nối** | **0,0 %** | ✅ 90 đoạn, 0 lỗi |
| nối `fakesink` | **11,6 %** (đỉnh 24 %) | ✅ 120 đoạn, 0 lỗi |

Cả hai đều ghi hình đúng, nên **không có gì báo** khi làm sai. Bản đầu của
`record --cam all` nối một `fakesink` vào mỗi nguồn cho "gọn", và nó decode cả 10 luồng để
vứt đi: 11,6 % NVDEC trên RTX 5090. Trên RTX 3060 một NVDEC gen 5 thì phần đó cộng thẳng
vào 981 Mpixel/s của nhánh model (§2.2) và đẩy tổng lên mức đã tính là **vượt trần**.

Kiểm bằng `nvidia-smi dmon -s u` khi đang ghi: cột `dec` phải bằng 0.
