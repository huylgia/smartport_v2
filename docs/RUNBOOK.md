# Runbook — triển khai Triton

Cách dựng và vận hành tầng inference trên **máy đích**. Mọi lệnh ở đây chạy qua Docker;
không cần `sudo` cho việc vận hành hằng ngày (chỉ cần một lần lúc cài Docker).

> **Đọc §1 trước khi làm gì khác.** Có hai ràng buộc khiến "copy nguyên si từ máy dev sang"
> **không** chạy được, và cả hai đều chỉ lộ ra sau khi bạn đã mất công.

---

## 1. Hai điều phải biết trước

### 1.1 Engine TensorRT gắn với **kiến trúc GPU** — phải dựng trên chính máy sẽ chạy

File `.plan` không mang đi được. Engine dựng trên RTX 5090 (`sm_120`) nạp trên RTX 3060
(`sm_86`) sẽ hỏng. Vì vậy máy đích **phải tự dựng engine** — đó là việc của `modelsvc`, và
là lý do lần khởi động đầu tốn vài phút.

Hệ quả: máy đích cần **kho model `.t7` gốc** và **mật khẩu giải mã**, không phải chỉ cần
thư mục engine.

### 1.2 Giấy phép gắn với **vân tay phần cứng** — và vân tay đó phải lấy **trong container**

Vân tay gồm ba nguồn: `dmi_uuid`, `board_serial`, `gpu`. Hai nguồn đầu đọc từ
`/sys/class/dmi/id/*` với quyền `0400` — **chỉ root đọc được**. Container chạy `user: "0:0"`
nên đọc đủ; tài khoản thường trên host thì không.

Đo thật trên máy dev, cùng một máy, hai ngữ cảnh:

| Chạy ở đâu | Nguồn đọc được | digest |
|---|---|---|
| host (user thường) | chỉ `gpu` | `201c6669…` |
| **trong container** | `board_serial`, `dmi_uuid`, `gpu` | `de4a5ace…` |

Hai digest **khác hẳn nhau**. Xin giấy phép bằng digest lấy ở host thì `modelsvc` sẽ từ
chối với *"giấy phép không thuộc thiết bị này"*, và không có gì gợi ý nguyên nhân.

Thêm một cái bẫy: máy dev có 2 GPU. Lệnh chạy ở host bắt được `GPU-0119232b` (index 1),
còn container được cấp `CRANEOPS_GPU=0` nên thấy `GPU-b2bc31c4`. **`CRANEOPS_GPU` lúc lấy
vân tay và lúc chạy phải giống nhau**, nếu không digest lệch dù cùng một máy.

---

## 2. Yêu cầu máy đích

| | Cần | Máy dev tham chiếu |
|---|---|---|
| GPU | RTX 3060 **12 GB** | 2× RTX 5090 |
| CPU | i7-12700, 20 luồng | 48 luồng |
| RAM | ≥ 16 GB (xem §5.1) | — |
| Driver NVIDIA | ≥ 535; đã kiểm ở **580.173.02** | 580.173.02 |
| Docker | ≥ 24; đã kiểm ở **29.4.3** | 29.4.3 |
| Docker Compose | v2+; đã kiểm ở **v5.1.3** | v5.1.3 |
| NVIDIA Container Toolkit | bắt buộc | — |
| Đĩa trống | ~20 GB (image ~15 GB + engine ~2 GB) | — |

Kiểm nhanh Toolkit đã chạy chưa:

```bash
docker run --rm --gpus '"device=0"' nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Thấy bảng `nvidia-smi` là đạt. Lỗi `could not select device driver` nghĩa là chưa cài
Toolkit.

---

## 3. Dựng image

```bash
git clone https://github.com/huylgia/smartport_v2.git
cd smartport_v2
make build-triton          # docker build -f build/triton.Dockerfile
```

Một image dùng cho cả hai service: `modelsvc` (dựng engine) và `triton` (phục vụ). Dùng
chung vì `modelsvc` cần `trtexec`, thứ đi kèm TensorRT trong image này.

Image nền là `nvcr.io/nvidia/tritonserver:25.10-py3` (~15 GB) nên lần kéo đầu lâu. Nếu máy
đích không ra internet được, xuất từ máy có mạng:

```bash
# máy có mạng
docker save craneops-triton:dev | zstd -T0 > craneops-triton.tar.zst
# máy đích
zstd -d < craneops-triton.tar.zst | docker load
```

---

## 4. Cấp giấy phép

Ba bước, hai bên. **Khoá riêng không bao giờ rời khỏi bên cấp phép** — đó là toàn bộ giá
trị của cơ chế.

### Bước 1 — bên cấp phép, một lần duy nhất cho cả sản phẩm

```bash
python -m tools.issue_license --new-keypair --out-private ~/craneops-license.key
```

In ra khoá công khai. Dán vào `EMBEDDED_PUBLIC_KEY` trong
`internal/pkg/security/license.py`, rồi build lại image. Khoá riêng cất offline; mất nó là
không cấp phép cho máy mới được nữa.

### Bước 2 — **trên máy đích**, lấy vân tay

Phải chạy trong container, với đúng `CRANEOPS_GPU` sẽ dùng lúc chạy thật (xem §1.2):

```bash
cp build/.env.triton.example build/.env.triton
# điền CRANEOPS_ASSETS, CRANEOPS_MODEL_PASSWORD, CRANEOPS_GPU trước đã;
# CRANEOPS_LICENSE_KEY để trống ở bước này

docker compose --env-file build/.env.triton -f build/docker-compose.triton.yml \
  run --rm --no-deps modelsvc python3 -m tools.issue_license --fingerprint
```

Kết quả trông như:

```
Vân tay thiết bị:
  board_serial   241247619300…
  dmi_uuid       a53bf5bc-fcb…
  gpu            GPU-b2bc31c4…

digest: de4a5ace66b63ca5d5051b83b45e26e9759afece499fc037a426f78d259bca42
```

✅ Phải thấy **đủ ba dòng**. Nếu có dòng `(thiếu: …)` thì bạn đang chạy sai ngữ cảnh — quay
lại §1.2.

Gửi digest cho bên cấp phép. Digest là băm một chiều, không lộ thông tin máy.

### Bước 3 — bên cấp phép ký

```bash
python -m tools.issue_license --issue de4a5ace… \
  --private ~/craneops-license.key --note GC03 --days 365
```

Trả về chuỗi `CO2.<payload>.<chữ ký>`. Dán vào `CRANEOPS_LICENSE_KEY` trong
`build/.env.triton` trên máy đích.

`--days 0` = vô thời hạn. Với máy tại cảng nên đặt hạn thật (365) — giấy phép hết hạn là
một tín hiệu vận hành, không phải phiền toái.

---

## 5. Cấu hình `build/.env.triton`

Đây là file **duy nhất** chứa bí mật, và nó bị `.gitignore` chặn. Không bao giờ commit.

```bash
CRANEOPS_LICENSE_KEY=CO2.…          # từ §4 bước 3
CRANEOPS_MODEL_PASSWORD=…           # mật khẩu giải mã .t7
CRANEOPS_ASSETS=/đường/dẫn/assets   # thư mục chứa .t7 và char_dict
CRANEOPS_GPU=0                      # PHẢI trùng lúc lấy vân tay
TRITON_IMAGE=craneops-triton:dev
```

> Không có mật khẩu thì `modelsvc` dừng ngay với `MissingPassword`. Đây là **cố ý**: một
> mật khẩu mặc định trong source là mật khẩu công khai.

### 5.1 Giới hạn tài nguyên cho RTX 3060 / i7-12700

Mặc định trong `.env.triton.example` hợp cho máy đích. Ba giá trị cần hiểu:

| Biến | Máy đích | Vì sao |
|---|---|---|
| `MODELSVC_MEMORY` | **12g** | ⚠️ **Không hạ xuống 4g** — `trtexec` từng bị SIGKILL (OOM) ở mức đó khi dựng `ccode_rec_*`. RAM hệ thống, chỉ tốn lúc dựng engine. |
| `MODELSVC_CPUS` | 4 | Chỉ chạy lúc khởi động. |
| `TRITON_CPUS` | 8 | Chừa CPU cho `ds_app` và các service nghiệp vụ. |
| `TRITON_MEMORY` | 8g | RAM hệ thống, không phải VRAM. |
| `TRITON_CUDA_POOL` | `2147483648` (2 GiB) | **Cách duy nhất chặn VRAM** — Docker không giới hạn được. |
| `TRITON_PINNED_POOL` | `268435456` (256 MiB) | |

Ngân sách VRAM đầy đủ: [HARDWARE_BUDGET.md §3](HARDWARE_BUDGET.md). Đo thật khi 9 model
nạp xong: **4 146 MiB**, còn dư nhiều so với 12 GB — biên này dành cho `ds_app` ở Phase 3.

Cổng mặc định 18000-18002 vì 8000-8002 thường đã bị chiếm. Đổi được qua
`TRITON_HTTP_PORT` / `TRITON_GRPC_PORT` / `TRITON_METRICS_PORT`.

---

## 6. Chạy lần đầu

```bash
make up
```

Chuỗi việc xảy ra, theo đúng thứ tự:

```
modelsvc:  kiểm giấy phép  →  giải mã .t7 vào /dev/shm (tmpfs)  →  trtexec dựng .plan
           →  XOÁ bản rõ ONNX  →  thoát 0
                                    ↓  service_completed_successfully
triton:    nạp 9 model từ volume craneops_models  →  phục vụ
```

`triton` chỉ khởi động sau khi `modelsvc` **thoát mã 0**, nên Triton không bao giờ chạy với
model chưa được cấp phép.

Lần đầu dựng 6 engine mất **~9 phút**. Lần sau `modelsvc` dùng lại engine cũ nếu nó mới hơn
file `.t7` nguồn ⇒ khởi động lại chỉ vài giây.

### Kiểm tra

```bash
make status
```

Phải thấy đủ **9/9 READY**:

```
craneops_ccode_det_h        READY     ← DB++ phát hiện chữ, mã ngang
craneops_ccode_det_v        READY     ← mã dọc
craneops_ccode_rec_h        READY     ← SVTR đọc chữ, mã ngang
craneops_ccode_rec_v        READY     ← mã dọc
craneops_ccode_h            READY     ← BLS ghép det→hậu xử lý→cắt→rec
craneops_ccode_v            READY     ← BLS
craneops_headcode_cls       READY     ← phân loại số đầu kéo
craneops_truckhead_pico     READY     ← PicoDet đầu kéo
craneops_truckitems_pico    READY     ← PicoDet container + đầu kéo
```

Kiểm sâu hơn — chạy trên chính máy đích, không tin số của máy dev:

```bash
make accuracy      # so với ảnh có nhãn trong assets/
make bench         # thông lượng, độ trễ, mức gom batch thật
```

Số tham chiếu từ máy dev (2× RTX 5090) — máy đích sẽ **thấp hơn**, đó là bình thường:
trần 1 137 req/s, p50 3,4 ms, VRAM 4 146 MiB. Tải mục tiêu thật chỉ ~50-100 req/s, nên
biên rất rộng. Độ chính xác thì **phải khớp** (`headcode_cls` 100,0 % trên 451 ảnh) — nếu
lệch, engine dựng sai, không phải máy yếu.

---

## 7. Vận hành hằng ngày

```bash
make status        # model nào READY
make logs          # 40 dòng log cuối
make down          # dừng. Engine nằm trong volume craneops_models, KHÔNG mất
make up            # chạy lại, vài giây vì dùng lại engine
make gpu-watch     # theo dõi VRAM / NVDEC / NVENC
```

### Khi nào engine bị dựng lại

`modelsvc` dựng lại một model khi **file `.t7` nguồn mới hơn `.plan`**. Nghĩa là:

* Đổi model → tự dựng lại, không cần làm gì.
* Đổi `config.pbtxt` (qua `tools/export_models.py`) → **không** tự dựng lại engine, nhưng
  Triton nạp config mới khi khởi động lại. Chạy `make config` rồi `make down && make up`.
* Muốn ép dựng lại tất cả:

```bash
docker volume rm craneops_models && make up      # ~9 phút
```

### Nâng cấp mã nguồn

Mã nguồn được mount `:ro` từ host, không nằm trong image. Nên:

```bash
git pull && make down && make up
```

Chỉ cần build lại image khi đổi `build/triton.Dockerfile` hoặc `EMBEDDED_PUBLIC_KEY`.

---

## 8. Sự cố thường gặp

Các lỗi dưới đây đều đã gặp thật trong quá trình phát triển.

### `giấy phép không thuộc thiết bị này`

Vân tay lúc xin phép ≠ vân tay lúc chạy. Theo thứ tự khả năng:

1. Lấy vân tay ở **host** thay vì trong container → thiếu `dmi_uuid`/`board_serial`. §1.2.
2. `CRANEOPS_GPU` đã đổi giữa hai lần → GPU UUID khác.
3. Đổi bo mạch chủ hoặc card. Phải xin giấy phép mới.

Chẩn đoán: lấy lại vân tay (§4 bước 2) và so digest với digest đã gửi đi.

### `giấy phép sai định dạng`

Khoá phải bắt đầu bằng `CO2.`. Chuỗi kiểu `XXXXXXXXXX-YYYYYYYYYY-NOEXP` **không được chấp
nhận** — đó là băm đối xứng tự sinh, không phải giấy phép có chữ ký.

### `trtexec bị giết bởi tín hiệu 9 (SIGKILL = 9, gần như chắc chắn là OOM)`

`MODELSVC_MEMORY` quá thấp. Đặt **12g**. Thông báo lỗi đã nói thẳng cách sửa.

### `trtexec không hỗ trợ kiến trúc GPU của máy này` / `Unsupported SM`

TensorRT trong image cũ hơn GPU. Dùng image Triton mới hơn. Nhắc lại §1.1: engine phải
dựng trên chính máy sẽ chạy.

### `/dev/shm không phải tmpfs — bản rõ model sẽ bị ghi xuống đĩa`

`modelsvc` **từ chối chạy** nếu `/dev/shm` không phải tmpfs, vì khi đó bản rõ ONNX sẽ được
ghi xuống đĩa. Nguyên nhân hầu như luôn là ai đó sửa `docker-compose.triton.yml`, chuyển
`/dev/shm` từ mục `tmpfs:` sang mục `volumes:` — Docker biến nó thành anonymous volume trên
ext4. Kiểm:

```bash
docker compose --env-file build/.env.triton -f build/docker-compose.triton.yml \
  run --rm --no-deps --entrypoint sh modelsvc -c "df -h /dev/shm"
```

Cột `Filesystem` phải là `tmpfs`. Thấy `/dev/nvme…` là sai.

### `MissingPassword: chưa đặt CRANEOPS_MODEL_PASSWORD`

Thiếu mật khẩu trong `.env.triton`. Cố ý không có giá trị dự phòng.

### Model READY nhưng kết quả là chuỗi rác

Gần như luôn là **sai hợp đồng đầu vào**. Mọi model nhận **pixel BGR thô `[0,255]`**, kiểu
`UINT8`; chuẩn hoá đã gấp vào đồ thị ONNX. Đưa RGB hay đưa dữ liệu đã chuẩn hoá thì model
vẫn chạy, vẫn trả về chuỗi — chỉ là chuỗi sai. Đo được: đưa RGB vào `headcode_cls` làm độ
chính xác tụt 100 % → 87,4 %. Chi tiết: [DESIGN_NOTES.md](DESIGN_NOTES.md) DN-012.

### Triton không lên sau `make up`

```bash
make logs
docker logs craneops-modelsvc-1        # modelsvc thất bại thì triton không bao giờ start
docker inspect craneops-modelsvc-1 --format '{{.State.ExitCode}}'   # phải là 0
```

`healthcheck` có `start_period: 300s` để chờ dựng engine lần đầu — trong 5 phút đó trạng
thái `starting` là bình thường.

---

## 9. Gỡ bỏ

```bash
make down                          # dừng, giữ engine
docker volume rm craneops_models   # xoá engine (dựng lại mất ~9 phút)
docker rmi craneops-triton:dev     # xoá image (~15 GB)
```

Không có gì được ghi ra ngoài volume và `/dev/shm`. Bản rõ ONNX bị xoá ngay sau khi dựng
xong engine, và `/dev/shm` là tmpfs nên mất hẳn khi container dừng.

---

## Xem thêm

- [ARCHITECTURE.md](ARCHITECTURE.md) — vì sao hệ thống được chia như vậy
- [HARDWARE_BUDGET.md](HARDWARE_BUDGET.md) — **nguồn sự thật duy nhất cho mọi con số**
- [DESIGN_NOTES.md](DESIGN_NOTES.md) — 13 quyết định đã chốt, kèm lý do
