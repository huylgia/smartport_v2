# Runbook — `ds_app` (DeepStream)

Cách dựng và vận hành tầng thu hình. Mọi lệnh chạy qua Docker; không cần `sudo` cho việc
vận hành hằng ngày.

**Tính năng hiện có: `record` — ghi hình passthrough, KHÔNG suy luận.** Nhánh model
(`nvstreammux` → `nvinferserver` → probe → Kafka) chưa xây; runbook này sẽ mở rộng khi nó
có. `record` không phải chế độ thử: với hai vai trò `bottom` và `evidence_only`, vốn không
bao giờ chạy model, đây là chế độ chạy **thật**.

Triton là service riêng, có runbook riêng: [RUNBOOK_TRITON.md](RUNBOOK_TRITON.md).
`record` **không cần** Triton.

> **Đọc §1 trước.** Hai cấu hình dưới đây, nếu sai, làm pipeline **im lặng không ra dữ
> liệu** — không exception, không log lỗi, không gì cả.

---

## 1. Hai thứ hỏng trong im lặng

### 1.1 GPU phải cấp bằng `count: all`, KHÔNG phải `device_ids`

Pin GPU (`device_ids: ["0"]`, hoặc `--gpus device=0`) cấp CUDA compute device nhưng **không**
cấp node V4L2 của NVDEC (`/dev/nvidia-caps/…`). Hậu quả: `nvv4l2decoder` mở được, pipeline
vào PLAYING, RTSP nhận đủ gói — rồi **đứng vĩnh viễn ở `PREROLLING`**, không một khung nào
chảy, và **không có thông báo lỗi nào**.

Kiểm nhanh, chạy được ở bất kỳ đâu có một file mp4:

```bash
gst-launch-1.0 -e filesrc location=<clip>.mp4 ! qtdemux ! h265parse ! nvv4l2decoder ! fakesink
# --gpus device=0 → treo ở PREROLLING
# --gpus all      → "Got EOS", "Execution ended"
```

`build/docker-compose.ds.yml` đã đặt `count: all`. Cần chia GPU trên máy dùng chung thì
dùng `NVIDIA_VISIBLE_DEVICES`, đừng quay lại `device_ids`.

### 1.2 `USE_NEW_NVSTREAMMUX=no` là bắt buộc

Plugin `nvstreammux` đọc biến này **lúc nạp** để chọn muxer cũ hay mới. DeepStream 8 mặc
định dùng muxer **mới**, vốn **bỏ qua toàn bộ** thuộc tính của muxer cũ mà code đang đặt
(`batch-size`, `width`, `height`, `batched-push-timeout`, `live-source`) — và khi đó nó
**không bao giờ đẩy một batch nào**. Không có metadata, không có lỗi.

Đã đặt ở cả `build/ds_app.Dockerfile` lẫn compose. `make ds-doctor` kiểm lại nó.

---

## 2. Yêu cầu máy đích

| | Cần | Máy dev tham chiếu |
|---|---|---|
| GPU | RTX 3060 **12 GB** | 2× RTX 5090 |
| Driver NVIDIA | ≥ 535; đã kiểm ở **580.173.02** | 580.173.02 |
| Docker | ≥ 24; đã kiểm ở **29.4.3** | 29.4.3 |
| NVIDIA Container Toolkit | bắt buộc | — |
| Đĩa | xem §5.2 — **~9,6 GB/giờ** cho 10 camera | — |
| Mạng | tới được cổng RTSP của camera | — |

Image nền `nvcr.io/nvidia/deepstream:8.0-triton-multiarch` ~21 GB; cộng lớp của dự án thì
cần ~25 GB đĩa cho image.

---

## 3. Dựng image

```bash
make ds-build
```

Image mang `pyds` và các gói Python; **mã nguồn không nằm trong image** — compose mount
`..:/app:ro`, nên sửa code là chạy lại thấy ngay, không phải build lại. Chỉ build lại khi
đổi `build/ds_app.Dockerfile`.

Hai chuyện về `pyds` đã tốn thời gian, ghi lại để khỏi lặp:

* Script cài kèm DeepStream đưa `pyds` vào **venv riêng của nó**, nên `import pyds` vẫn
  hỏng dù nó báo "Successfully installed". Dockerfile lấy chính wheel đó cài lại vào Python
  hệ thống.
* **Không kiểm `import pyds` lúc build.** Lúc `docker build` không có GPU nên
  `libcuda.so.1` chỉ là stub và import chết với `file too short` — dù cài hoàn toàn đúng.
  Thư viện thật do Container Toolkit cấp lúc chạy.

---

## 4. Cấu hình `build/.env.ds`

```bash
cp build/.env.ds.example build/.env.ds
```

File env **riêng** với Triton: hai service khác nhau về ngân sách tài nguyên, tập secret và
vòng đời. Gộp chung thì đổi một biến của Triton lại phải khởi động lại pipeline. File này
bị `.gitignore` chặn.

```bash
CAM01_RTSP=rtsp://<user>:<pass>@113.160.225.15:1508//CH001.sdp
CAM02_RTSP=...                      # đủ 10 camera
CRANEOPS_REC_DIR=/var/lib/craneops/rec
RETAIN_SEC=1800
```

⚠️ **Chỉ URL, không kèm gì khác.** Codec tự nhận (`nvurisrcbin` dò, nhánh ghi lấy đúng họ
parser mà nhánh decode dùng); fps lấy nguyên của nguồn. Nếu URL của bạn đến từ một định
dạng có dấu phân tách, **phải dừng ở dấu phân tách** — config sẽ từ chối URL chứa `|` hoặc
khoảng trắng, vì GStreamer thì không: nó giữ nguyên phần thừa trong path và gửi cho camera,
mà camera có thể bỏ qua, nên lỗi sống sót tới lúc gặp firmware khác.

`CRANEOPS_REC_DIR` tương đối thì tính từ `build/`, không phải gốc repo.

### 4.1 Mã camera suy từ URL, không khai

Định danh là `<mã cẩu>_<ip>_<cổng>`, ví dụ `GC03_113_160_225_15_1508`. Nó là tên thư mục
ghi hình và là trường `camera_code` trong message, nên hai thứ đó không thể trôi khỏi nhau.
`name` trong YAML chỉ là mô tả cho người đọc.

⚠️ **Cổng là phần bắt buộc.** Cả 10 camera GC03 đi qua một gateway NAT ở
`113.160.225.15`, chỉ khác cổng `1508`–`1517`. Mã chỉ gồm IP sẽ giống hệt nhau cho cả 10.
Mã trùng nhau bị từ chối lúc load.

---

## 5. Chạy

### 5.1 Kiểm môi trường trước

```bash
make ds-doctor
```

Chạy cái này **trước khi nghi ngờ bất cứ thứ gì khác**. Nó phân biệt "image hỏng" với "cấu
hình hỏng" với "code hỏng" — ba thứ có triệu chứng giống hệt nhau (không ra dữ liệu, không
lỗi) nhưng cách sửa hoàn toàn khác.

```
  ✅ pyds                 /usr/local/lib/python3.12/dist-packages/pyds.so
  ✅ USE_NEW_NVSTREAMMUX  no ⇒ mux CŨ (đúng)
  ✅ nvurisrcbin          nguồn RTSP kèm tee trước decode
  ✅ nvv4l2decoder        decode phần cứng — thiếu quyền `video` thì treo PREROLLING
  ✅ nvstreammux          gộp nhiều nguồn thành batch, và là thứ TẠO RA metadata
  ✅ nvinferserver        gọi Triton (nhánh model, Phase 3b)
  ✅ splitmuxsink         ghi segment
  ✅ config + URL camera  GC03: 10 camera, 8 vào nhánh model
  ✅ /rec ghi được        /rec
```

### 5.2 Ghi hình

```bash
make record CAM=1 DUR=60                    # chạy 60 giây rồi dừng
make record CAM=1 DUR=0                     # chạy mãi, Ctrl-C để dừng
SEGMENT_SEC=60 make record CAM=1 DUR=0      # mỗi đoạn 60 giây thay vì 10
```

`DUR` là **thời gian chạy**, `SEGMENT_SEC` là **độ dài mỗi đoạn** — hai thứ khác nhau. Đặt
lâu dài thì để `SEGMENT_SEC` trong `build/.env.ds`.

⚠️ **Độ dài đoạn chính là độ trễ tệ nhất của bằng chứng.** `evidenced` phải chờ một đoạn
**đóng** mới cắt được từ nó (mp4 mới có `moov` đầy đủ), nên đoạn 60 s nghĩa là một khoảnh
khắc vừa xảy ra có thể phải chờ tới 60 s. Đoạn ngắn thì ngược lại: nhiều file hơn, nhiều
lần đóng/mở hơn. Cửa sổ evidence xa nhất của hệ là `-35 s`, nên đoạn **10 s là mặc định hợp
lý**; tăng lên chỉ khi thật sự cần ít file.

Đo thật với `SEGMENT_SEC=60`: đoạn cách nhau đúng `+60.00s`, độ dài học được `60.00s`,
0,7 GB/giờ — bằng đúng mức của đoạn 10 s, vì cùng một bitstream chỉ khác cách chia file.

```
cẩu GC03 · camera 1 · vai trò ccode
  mã       GC03_113_160_225_15_1508
  mô tả    Mặt phải trước
  ghi vào  /rec/GC03_113_160_225_15_1508/

[đoạn] 1788187000.mp4   mở lúc 1788187000.799
[đoạn] 1788187010.mp4   mở lúc 1788187010.799  (+10.00s)
[record] 1: GOP nguồn 1.67s, giới hạn 10s ⇒ đoạn dài thật ~10.0s
```

**Đọc kết quả:**

* Tên file là **epoch nguyên giây**, chính là lúc tạo đoạn — một con số để so thẳng với dấu
  thời gian khung, không phải chuỗi giờ buộc phải phân tích ngược (và đoán múi giờ).
* `(+10.00s)` **chính xác** là dấu hiệu trục thời gian đúng: mốc đoạn nằm trên trục PTS, chứ
  không phải đồng hồ tường vốn dao động theo độ trễ hàng đợi. Thấy `+9.7s`, `+10.3s` lệch
  lung tung là dấu hiệu neo thời gian hỏng.
* Dòng `GOP nguồn` báo độ dài đoạn **thật**. `splitmuxsink` chỉ cắt tại keyframe nên độ dài
  thật là bội số của GOP; GOP của camera cảng **không cố định** (đo được cả 1,00 s lẫn
  1,67 s). Ai chờ hết một đoạn phải lấy số này, không lấy `max-size-time` trong config.

  ⚠️ `FragmentIndex.observed_duration()` lấy **giá trị nhỏ nhất** từng thấy. Đo thật trên
  camera 1: phần lớn lần chạy cho đúng `10.00s`, nhưng một lần cho `8.50s` — một gap ngắn
  bất thường (mạng chập, hoặc bắt đầu giữa GOP) làm giá trị nhỏ nhất **dính lại vĩnh viễn**.
  Nó an toàn theo hướng "báo ngắn hơn thực tế", nhưng ai dùng nó để chờ một đoạn đóng sẽ
  chờ thiếu. Khi làm `evidenced`, dùng trung vị hoặc bỏ mẫu đầu tiên.

**Đo được, một camera:** ~3,2 MB mỗi 10 s ≈ **0,6–0,8 GB/giờ**. Mười camera ≈ **9,6 GB/giờ**
(xem [HARDWARE_BUDGET.md](HARDWARE_BUDGET.md) §2.6).

**NVENC phải bằng 0.** Đó là điều kiện sống còn: RTX 3060 chỉ có 1 NVENC và card GeForce
giới hạn số phiên encode. Nhánh ghi cắm vào luồng **chưa decode** nên không encode lại gì.
Kiểm trong lúc đang ghi:

```bash
make gpu-watch      # cột enc PHẢI là 0; dec ~1-2 % cho một camera
```

### 5.3 Xem metadata DeepStream trả về

```bash
make record CAM=1 DUR=60 META=1
```

```
[meta] khung #2
  pad_index      0        (chỉ số nguồn trong nvstreammux)
  frame_num      1        (không decimate ⇒ là chỉ số THẬT)
  buf_pts        574905315  = 0.575s
  → unix (trục chung) 1788185509.878
  khung          2688x1520
  num_obj_meta   0        (chưa có model ⇒ rỗng, đúng như mong đợi)
  đoạn chứa nó   1788185509.mp4   (đã đóng)
```

`num_obj_meta = 0` là **đúng** ở giai đoạn này: chưa có model nào chạy, nên đây là metadata
**khung**, chưa phải metadata **vật thể**. Khi nhánh model có, `obj_meta_list` sẽ mang bbox,
nhãn và độ tin cậy.

Ba thứ ở trên là chính xác những gì `PerceptionMessage` cần: chỉ số khung, dấu thời gian
trên trục chung, và đoạn nào chứa khung đó.

---

## 6. Vận hành

```bash
make ds-doctor                     # kiểm môi trường
make record CAM=<n> DUR=0          # ghi liên tục một camera
make gpu-watch                     # NVENC phải = 0
make record-clean                  # xoá thư mục ghi cục bộ
```

### Dọn đĩa

Sweeper chạy trong **cùng tiến trình** với nhánh ghi, mỗi 60 giây, giữ mặc định 30 phút và
trần 20 GB. Cùng tiến trình là cố ý: sweeper chạy khác user với tiến trình ghi sẽ
`PermissionError` trên **mọi** file và không xoá được gì.

Nó không bao giờ vượt một **sàn tuổi 5 phút**, kể cả khi đã đầy đĩa — đó là cửa sổ mà
`evidenced` còn cần. Vượt trần mà mọi đoạn đều trẻ hơn sàn ⇒ **không xoá gì** và log:

```
[sweep] ⚠️ vượt trần X GB mà mọi đoạn còn trong cửa sổ bằng chứng —
        hạ fps nguồn hoặc cấp thêm đĩa, ĐỪNG nới min_age_sec
```

Nới sàn để "sửa" là phá bằng chứng để cứu dung lượng — đổi một sự cố ồn ào lấy một sự cố im
lặng.

---

## 7. Sự cố thường gặp

### Pipeline vào PLAYING nhưng không có đoạn nào, không lỗi

Gần như chắc chắn là §1.1 hoặc §1.2. Chạy `make ds-doctor` trước.

### `ConfigError: cần biến môi trường CAM0N_RTSP … nhưng nó trống`

Thiếu URL trong `build/.env.ds`. Thông báo nói rõ biến nào — không cần mò từng camera.

### `URL chứa ký tự '|' — gần như chắc chắn là trích thiếu từ một định dạng có phân tách`

URL lấy từ định dạng phân tách mà quên dừng ở dấu phân tách. Xem §4.

### `mã camera trùng nhau: [...]`

Hai camera cùng host **và** cùng cổng, hoặc URL thiếu cổng. Để nguyên thì dữ liệu của camera
này bị gán cho camera kia mà không có gì báo. Xem §4.1.

### `[record] <cam>: bỏ buffer không có PTS`

Nguồn RTSP thỉnh thoảng đẩy buffer không có dấu thời gian; để nó tới `mp4mux` thì muxer
**abort** và mất cả đoạn đang ghi. Log một lần cho mỗi camera. Lặp lại liên tục ⇒ nguồn có
vấn đề, kiểm mạng tới camera.

### `[sweep] ❌ N file xoá không được — PermissionError`

Sweeper chạy khác user với tiến trình ghi. Nếu chỉ chạy qua `make record` thì không xảy ra;
nó xảy ra khi ai đó chạy sweeper riêng hoặc dọn tay bằng user khác.

### Đoạn ghi ra nhưng không mở được

Kiểm `h265parse config-interval=-1` còn nguyên trong `recorder.py`. Không có nó thì tham số
bộ giải mã (VPS/SPS/PPS) không được chèn lại trước mỗi keyframe và từng đoạn không tự đứng
được. Thử:

```bash
gst-launch-1.0 -e filesrc location=<đoạn>.mp4 ! qtdemux ! h265parse ! nvv4l2decoder ! fakesink
```

---

## 8. Gỡ bỏ

```bash
make record-clean                       # xoá segment
docker rmi craneops-ds:dev              # xoá image (~25 GB)
```

---

## Xem thêm

- [RUNBOOK_TRITON.md](RUNBOOK_TRITON.md) — triển khai Triton (service riêng)
- [ARCHITECTURE.md](ARCHITECTURE.md) — vì sao tách ở tầng bitstream, vì sao chỉ 8/10 camera decode
- [HARDWARE_BUDGET.md](HARDWARE_BUDGET.md) — **nguồn sự thật duy nhất cho mọi con số**
- [DESIGN_NOTES.md](DESIGN_NOTES.md) DN-014 — nhánh ghi tách ở tầng bitstream, và bẫy cấp GPU
