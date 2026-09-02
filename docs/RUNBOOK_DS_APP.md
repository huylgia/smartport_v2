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

## 0. Bộ lệnh `craneops`

Mọi thao tác vận hành đi qua ba script trong `deploy/`. Chúng là **shell thuần**: chạy được
trên máy đích, nơi chỉ có Docker — không cần `uv`, không cần venv, không cần cài gì của dự
án.

Đưa lên `PATH` một lần, sau đó gọi từ thư mục bất kỳ:

```bash
./deploy/craneops install    # symlink vào ~/.local/bin — KHÔNG cần sudo
```

```bash
craneops          # hỏi MỌI microservice cùng một câu
craneops-triton   # chỉ Triton
craneops-ds       # chỉ ds_app
```

**Hợp đồng lõi.** Mọi service cài đủ sáu lệnh `build`, `up`, `down`, `status`, `logs`,
`doctor`, nên `craneops <lệnh>` luôn hỏi được cả hệ thống. "Chưa áp dụng" vẫn phải là một
câu trả lời: `craneops-ds up` nói thẳng ds_app chưa có service chạy dài và chỉ sang
`record`, chứ không im lặng bỏ qua — im lặng thì không phân biệt được *chưa xây*, *đang
tắt* và *hỏng*. Lệnh riêng của từng service (`bench`, `accuracy`, `record`, `clean`) nằm
ngoài hợp đồng và không được fan-out; `craneops services` chỉ rõ cái nào là cái nào.
Hợp đồng khai ở `CORE_COMMANDS` trong `deploy/craneops-lib.sh` và có test khoá lại.

Gõ tên lệnh không kèm gì để xem danh sách. `install` tạo **symlink**, không copy — nên
`git pull` là cập nhật xong, không phải cài lại. Muốn dùng chung cho mọi user trên máy thì
`CRANEOPS_BIN=/usr/local/bin ./deploy/craneops install` (chỗ này mới cần quyền ghi);
`craneops uninstall` gỡ ra. Không cài cũng chạy được, chỉ là phải gõ đủ `./deploy/craneops`.

`make` chỉ còn dùng trên máy dev (`make check`, `make schema`, `make config` — chúng cần
`uv`). Các target vận hành cũ vẫn còn nhưng chỉ gọi lại CLI, nên không thể trôi khỏi nhau.

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

Đã đặt ở cả `build/ds_app.Dockerfile` lẫn compose. `craneops-ds doctor` kiểm lại nó.

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
craneops-ds build
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
CRANEOPS_RTSP_CRED=<user>:<pass>    # credential dùng chung cho MỌI camera
CRANEOPS_REC_DIR=/var/lib/craneops/rec
RETAIN_SEC=1800
```

⚠️ **Chỉ URL, không kèm gì khác.** Codec tự nhận (`nvurisrcbin` dò, nhánh ghi lấy đúng họ
parser mà nhánh decode dùng); fps lấy nguyên của nguồn. Nếu URL của bạn đến từ một định
dạng có dấu phân tách, **phải dừng ở dấu phân tách** — config sẽ từ chối URL chứa `|` hoặc
khoảng trắng, vì GStreamer thì không: nó giữ nguyên phần thừa trong path và gửi cho camera,
mà camera có thể bỏ qua, nên lỗi sống sót tới lúc gặp firmware khác.

`CRANEOPS_REC_DIR` tương đối thì tính từ `build/`, không phải gốc repo.

### 4.1 Thêm camera

Camera nhóm theo **chức năng** trong `configs/cranes/<cẩu>.yaml`. Thêm một camera cho chức
năng đang có — camera ccode thứ 6 để tăng độ chính xác, hay một camera tcode dự phòng — là
**một mục YAML**:

Một camera là **một dòng** — đếm được bằng mắt, và một diff đổi camera nào thì thấy ngay
camera đó:

```yaml
cameras:
  ccode:
    - {code: GC03_113_160_225_15_1508, stream: rtsp://113.160.225.15:1508//CH001.sdp, desc: Mặt phải trước}
    # … 4 dòng nữa …
    - {stream: rtsp://113.160.225.15:1518//CH001.sdp, desc: Mặt phải trước - dự phòng}   # ← thêm
```

```bash
make codes      # điền `code`, xếp lại thứ tự trường, đối chiếu với mọi service
```

Hết. Không đặt tên, không cấp phát số, không đụng `.env.ds` lẫn `docker-compose.ds.yml`.

**URL không mang credential.** Host, cổng, path — tất cả định danh luồng — nằm trong config;
chỉ `CRANEOPS_RTSP_CRED` ở env, vì nó là bí mật chứ không phải cấu hình. Nhờ vậy `code` đọc
được từ chính file config: CI xác thực được, và người review một diff biết ngay mã nào ứng
với camera nào.

**`code` sinh ra rồi kiểm lại.** `make codes` ghi nó vào YAML; sửa tay mà lệch `stream` thì
load báo lỗi. Nó hiện ra được nhưng không trôi được — quan trọng vì `code` là chuỗi đi xuyên
cả hệ: ds_app đặt lên `PerceptionMessage`, rule tra config theo nó, evidence đặt tên thư mục
segment theo nó.

`craneops-ds doctor` in bảng đối chiếu để kiểm sau mỗi lần sửa:

```
=== camera ===
  ccode1   GC03_113_160_225_15_1508  rtsp://113.160.225.15:1508//CH001.sdp  Mặt phải trước
  tcode2   GC03_113_160_225_15_1512  rtsp://113.160.225.15:1512//CH001.sdp  Đầu kéo - Lane 1
```

⚠️ **Đổi IP/cổng của một camera là đổi mã của nó.** `make codes` cập nhật YAML và báo ngay
những config rule đang khoá theo mã cũ — đừng bỏ qua cảnh báo đó, vì rule không tìm thấy
config sẽ **im lặng** không xử lý camera ấy.

⚠️ **Mỗi camera chạy model là thêm tải NVDEC.** Ngân sách ở
[HARDWARE_BUDGET.md](HARDWARE_BUDGET.md) §2.2 — vượt trần thì **mọi** camera cùng tụt fps,
không phải camera mới bị bỏ. Camera chỉ để ghi hình thì đặt vai trò `evidence_only`.

### 4.2 Đoạn hoàn tất và đoạn đang ghi

Đoạn **đang ghi** mang đuôi `.mp4.part`; khi `splitmuxsink` báo đóng xong, nó được đổi tên
thành `.mp4`. Đổi tên trong cùng thư mục là thao tác **nguyên tử**, nên ai duyệt `*.mp4`
không bao giờ thấy một file dở dang.

```
1788278470.mp4          ✅ hoàn tất
1788278477.mp4          ✅ hoàn tất
1788278490.mp4.part     ⏳ đang ghi, hoặc mồ côi sau khi tiến trình chết
```

⚠️ **`.part` KHÔNG có nghĩa là không đọc được.** `reserved-moov-update-period-sec: 1` làm
mới `moov` mỗi giây nên đoạn đang ghi vẫn mở được — và `evidenced` thường cần chính nó, vì
cửa sổ bằng chứng hay chạm vào đoạn hiện tại. Đuôi này nói **"chưa chốt"**, không nói
"hỏng". Dùng `Fragment.live_path` để đọc đoạn chưa chốt.

⚠️ **Phải chờ message `splitmuxsink-fragment-closed`, đừng đổi tên lúc đoạn kế mở ra.**
`async-finalize` đóng file ở luồng khác; đo được trên bus là `fragment-closed` của đoạn N
tới **sau** `fragment-opened` của đoạn N+1.

Đo bằng `kill -9` giữa đoạn: ba đoạn trước hoàn tất và đọc được đủ 200 khung, đoạn đang mở
còn `.part` với 31 khung — mất **~3 giây** cuối, đúng bằng khoảng tới lần làm mới `moov`
gần nhất. Không có `reserved-moov-update-period` thì mất **trọn cả đoạn**.

Sweeper duyệt `*.mp4*` nên dọn được cả `.part` mồ côi — nó luôn là file to nhất trong thư
mục vì chưa bị cắt.

### 4.3 Giữ bao nhiêu, và vì sao không ít hơn

Mặc định: **6 đoạn 30 giây mỗi camera = 3 phút**, ~0,55 GB cho cả 10 camera.

```bash
SEGMENT_SEC=30     # 18 GOP chẵn ⇒ đoạn ra đúng 30,00 s
KEEP_SEGMENTS=6    # mỗi camera; 0 = không dọn
```

Đếm **số đoạn** chứ không đếm tuổi: độ dài đoạn dao động theo GOP (đo được 8,33 s và
9,26 s cho cùng cấu hình 10 s), nên đếm tuổi cho ra số đoạn khác nhau mỗi lúc.

⚠️ **Sàn cứng 75 giây.** Ràng buộc thật không phải độ dài clip mà là clip **với tới bao xa
về quá khứ**: cửa sổ xa nhất `-35 s` với `delay: 40 s` ⇒ lúc job chạy, dữ liệu cần đã 75 s
tuổi. `SweepPolicy` từ chối mọi `min_age_sec` nằm giữa 0 và 75 — một con số như 30 trông
đã cân nhắc nhưng vẫn xoá mất bằng chứng, và xoá im lặng.

| Giữ | Biên trên sàn | Đĩa (10 camera) |
|---|---|---|
| 60 s | **−15 s** ❌ mất bằng chứng | 0,17 GB |
| 120 s | 45 s | 0,35 GB |
| **180 s** (mặc định) | **105 s** | **0,52 GB** |
| 300 s | 225 s | 0,87 GB |

### 4.4 Kiểm nhánh ghi có mất dữ liệu không

`record` báo ở cuối mỗi lần chạy và **thoát khác 0** nếu có mất:

```
✅ không mất dữ liệu (0 lần hàng đợi đầy, 0 nghi mất khung I)
```

Hai phép dò đo hai thứ khác nhau, cố ý:

| | Đo gì | Bắt được |
|---|---|---|
| `overrun` của hàng đợi ghi | **nguyên nhân** — hàng đợi đầy nên đã vứt buffer | đúng nguyên nhân đó |
| khoảng cách keyframe > 1,5 lần GOP | **kết quả** — đã mất một khung I | mọi nguyên nhân |

Cần cả hai: mất một khung I là mất cả GOP theo sau (~1,7 s hình), và **file vẫn được tạo,
vẫn mở được** — không có gì báo nếu không đo.

#### Dựng nghẽn để kiểm bộ dò

⚠️ **Đừng bóp băng thông đĩa trên máy dùng chung.** Bóp nhỏ hàng đợi cũng không tạo được
nghẽn — hàng đợi chỉ đầy khi phía **sau** chậm hơn phía trước, mà đĩa thì không chậm.

Dùng pipeline tổng hợp, chạy trong container, không chạm đĩa lẫn camera:

```bash
docker compose --env-file build/.env.ds -f build/docker-compose.ds.yml run --rm \
  --entrypoint python3 doctor -c '
import gi; gi.require_version("Gst","1.0")
from gi.repository import GLib, Gst
Gst.init(None)
p = Gst.parse_launch(
  "videotestsrc is-live=true ! video/x-raw,framerate=60/1 "
  "! queue name=q leaky=1 max-size-buffers=2 max-size-time=0 max-size-bytes=0 "
  "! identity sleep-time=50000 ! fakesink sync=false")
n=[0]; p.get_by_name("q").connect("overrun", lambda _q: n.__setitem__(0,n[0]+1))
p.set_state(Gst.State.PLAYING)
l=GLib.MainLoop(); GLib.timeout_add_seconds(5, lambda:(l.quit(),False)[1]); l.run()
print("overrun:", n[0])'
```

Nguồn nhanh (60 fps) → hàng đợi 2 buffer → sink cố ý ngủ 50 ms mỗi buffer. Đo được
**225 lần overrun trong 5 giây** — tín hiệu nối đúng.

### 4.5 Mã camera suy từ URL, không khai

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
craneops-ds doctor
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
  ✅ nvvideoconvert       đổi sang RGBA — thiếu thì probe không map được khung ra numpy
  ✅ splitmuxsink         ghi segment
  ✅ config + URL camera  GC03: 10 camera (…), 8 khai chạy model, 3 có model hôm nay
  ✅ /rec ghi được        /rec
```

### 5.2 Dò nhịp camera — làm MỘT LẦN khi lắp cẩu mới

```bash
craneops-ds probe                    # ~50 s, ghi passthrough rồi đếm khung trong file
craneops-ds probe --duration 140     # kỹ hơn: nhiều đoạn, lấy trung vị
```

`source_fps` phải khai trong config và **không tự dò được lúc chạy**: `drop-frame-interval`
chỉ đặt được ở NULL/READY (trước khi có khung nào), còn caps của nguồn khai `framerate=0/1`
trên cả 10 camera. `probe` đo bằng cách đếm khung trong chính đoạn ghi passthrough — 0 %
NVDEC, đúng cho cả camera chỉ-ghi — rồi in ra dòng để dán vào config.

Đừng bỏ bước này. Đo trên GC03 thấy **3 trong 10 camera không chạy 30 fps** (18, 27, 24)
trong khi config khai 30 cho tất cả; khai sai làm nhịp model lệch tới 40 % mà không gì báo.
Kết quả **không tự ghi vào config**: camera đang rớt mạng lúc dò sẽ đo ra nhịp thấp, và
chốt con số đó là chốt vĩnh viễn một lỗi.

### 5.3 Chạy thật — ghi hình + suy luận

```bash
craneops-ds run                          # chạy mãi, Ctrl-C để dừng
craneops-ds run --duration 1800          # 30 phút rồi tự dừng
craneops-ds run --segment-sec 30 --keep-segments 6
```

Đây là chế độ production: **một** `nvurisrcbin` cho mỗi camera, nhánh ghi cắm vào tee
trước decode, nhánh model lấy pad đã decode, cả hai dùng chung một `TimeSync`. Cần Triton
và bus chạy trước (`craneops up`).

Chỉ ở chế độ này `segment_hint` mới được điền — chỉ nhánh ghi biết đoạn nào chứa một
khoảnh khắc, chỉ nhánh model gửi message, và `evidenced` cần cả hai để cắt clip.

Bảng tổng kết lúc thoát báo: khung mất ở hàng đợi suy luận, message mất ở bus, nghi mất
khung I ở nhánh ghi, và nhịp nguồn đo được so với nhịp khai.

### 5.4 Ghi hình riêng — chẩn đoán nửa dưới

```bash
craneops-ds record --cam all --duration 60                  # MỌI camera — chỉ nhánh ghi
craneops-ds record --cam ccode1 --duration 60               # một camera
craneops-ds record --cam tcode2 --duration 0                 # chạy mãi, Ctrl-C để dừng
craneops-ds record --cam ccode1 --duration 0 --segment-sec 60   # đoạn 60 giây thay vì 10
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
nvidia-smi dmon -s pucm      # cột enc VÀ dec đều PHẢI là 0 — kể cả với 10 camera
                             # dec > 0 nghĩa là có gì đó nối vào src pad của nguồn;
                             # xem HARDWARE_BUDGET §6.3
```

### 5.5 Xem metadata DeepStream trả về

```bash
craneops-ds record --cam ccode1 --duration 60 --meta
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
craneops-ds doctor                     # kiểm môi trường
craneops-ds record --cam <khoá> --duration 0      # ghi liên tục một camera
nvidia-smi dmon -s pucm                     # NVENC phải = 0
craneops-ds clean                  # xoá thư mục ghi cục bộ
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

Gần như chắc chắn là §1.1 hoặc §1.2. Chạy `craneops-ds doctor` trước.

### `ConfigError: cần biến môi trường CAM0N_RTSP … nhưng nó trống`

Thiếu URL trong `build/.env.ds`. Thông báo nói rõ biến nào — không cần mò từng camera.

### `URL chứa ký tự '|' — gần như chắc chắn là trích thiếu từ một định dạng có phân tách`

URL lấy từ định dạng phân tách mà quên dừng ở dấu phân tách. Xem §4.

### `mã camera trùng nhau: [...]`

Hai camera cùng host **và** cùng cổng, hoặc URL thiếu cổng. Để nguyên thì dữ liệu của camera
này bị gán cho camera kia mà không có gì báo. Xem §4.5.

### `[record] <cam>: bỏ buffer không có PTS`

Nguồn RTSP thỉnh thoảng đẩy buffer không có dấu thời gian; để nó tới `mp4mux` thì muxer
**abort** và mất cả đoạn đang ghi. Log một lần cho mỗi camera. Lặp lại liên tục ⇒ nguồn có
vấn đề, kiểm mạng tới camera.

### `[sweep] ❌ N file xoá không được — PermissionError`

Sweeper chạy khác user với tiến trình ghi. Nếu chỉ chạy qua `craneops-ds record` thì không xảy ra;
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
craneops-ds clean                       # xoá segment
docker rmi craneops-ds:dev              # xoá image (~25 GB)
```

---

## Xem thêm

- [RUNBOOK_TRITON.md](RUNBOOK_TRITON.md) — triển khai Triton (service riêng)
- [ARCHITECTURE.md](ARCHITECTURE.md) — vì sao tách ở tầng bitstream, vì sao chỉ 8/10 camera decode
- [HARDWARE_BUDGET.md](HARDWARE_BUDGET.md) — **nguồn sự thật duy nhất cho mọi con số**
- [DESIGN_NOTES.md](DESIGN_NOTES.md) DN-014 — nhánh ghi tách ở tầng bitstream, và bẫy cấp GPU
