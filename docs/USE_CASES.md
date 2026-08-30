# Use case — quy trình nghiệp vụ và luồng bắn event

Tài liệu này map **1-1** quy trình nghiệp vụ do phía Cảng cung cấp sang rule, camera,
signal và topic của `smartport_v2`.

* Đặc tả *rule làm gì*: [RULES.md](RULES.md)
* Hợp đồng *message*: [MESSAGE_CONTRACT.md](MESSAGE_CONTRACT.md)
* Quyết định còn mở: [DESIGN_NOTES.md](DESIGN_NOTES.md)

---

## 0. Bối cảnh

**Cẩu QC** (Quay Crane) bốc container giữa **tàu** và **bờ**. Xe đầu kéo đỗ dưới bụng cẩu,
tối đa 3 làn. Mỗi lần bốc một container là một **chu kỳ thao tác**, và mỗi chu kỳ sinh ra
một **event** đẩy sang hệ thống Cảng.

Hai chiều hàng, **quy trình khác nhau về thứ tự**:

| | Hàng nhập (`ix_cd = I`) | Hàng xuất (`ix_cd = X`) |
|---|---|---|
| Hướng | tàu → bờ, cont đặt **lên** xe | xe → tàu, cont bốc **khỏi** xe |
| Container có trên xe lúc xe vào? | **Không** — đến từ tàu | **Có** |
| Nhận dạng số cont **so với** lúc cẩu bốc | **sau** | **trước** |
| Push API | **N lần**, upsert theo `eventId` — xem G-1 | **N lần**, lần đầu sớm hơn |
| Ảnh chụp | 6 mặt một lần | 5 mặt, rồi mặt đáy sau khi cẩu bốc |

### Ánh xạ "6 mặt container" → camera (GC03)

| Mặt | Camera | Tên trong config |
|---|---|---|
| Trước | 6 (lane 1), 4 (lane 2) | `Mặt trước - Lane 1` / `Lane 2` |
| Sau (cửa) | 7 (lane 1), 8 (lane 2) | `Cửa sau - Lane 1` / `Lane 2` |
| Phải | 1 | `Mặt phải trước` |
| Trái | 2 | `Hông trái - Trước` |
| Trên | 10 | `Trần container` |
| **Đáy** | **9** | `Soi đáy` |

Camera 3 và 5 (`Đầu kéo - Lane 1/2`) **không** tham gia chụp mặt container — chúng chỉ
đọc số xe đầu kéo. Chín camera có clip: 1, 2, 4, 6, 7, 8, 9, 10, 11 — không có 3 và 5.

---

## I. Quy trình hàng nhập (`ix_cd = I`)

> Hàng dỡ từ tàu ⇒ nhập lên xe đầu kéo trên bờ

| # | Bước nghiệp vụ | Rule / service | Camera | Signal phát ra |
|---|---|---|---|---|
| 1 | Xe đầu kéo chạy vào vị trí dưới bụng cẩu QC | `CRANE01` gán lane<br>`CRANE02` xác nhận xe đã dừng | 10 | `lane_active`<br>`truck_stable` |
| 2 | Nhận dạng số xe đầu kéo (số dán trên nóc đầu kéo) | `TCODE01` | 3, 5 | `truck_no` |
| 3 | Cẩu QC bốc cont từ tàu lên bờ | `CRANE03` | 10 | `crane_op` ← **mốc neo** |
| 4 | Chụp **6 mặt** container | `evidenced` | 1,2,4,6,7,8,9,10 | — |
| 5 | Nhận dạng số container (2 mã nếu xe chở 2 cont 20′) | `CCODE01`<br>`CRANE04` xác định slot | 1,4,6,7,8<br>10 | `container_no`<br>`cont_position` |
| 6 | Push API sang hệ thống Cảng | `orchestratord` → `syncd` | — | `craneops.events` |

**Vì sao bước 5 nằm sau bước 3:** container chưa có trên xe cho tới khi cẩu đặt xuống.
Không thể đọc mã trước đó.

**Vì sao chụp được cả mặt đáy ở bước 4:** lúc này container đang **treo lơ lửng** phía trên
xe, camera 9 nhìn từ dưới lên thấy được đáy. Sau khi hạ xuống rơ-moóc thì không còn thấy.

**Điểm khó — thứ tự ngược:** hệ thống chỉ biết đây là hàng **nhập** sau khi đã đọc xong số
container và tra manifest Oracle. Nhưng ảnh 6 mặt phải chụp
**trước** thời điểm đó. Một buffer quay vòng trong RAM giải được, nhưng nó giới hạn cửa sổ
nhìn lại đúng bằng dung lượng buffer. **Dùng segment mp4 trên đĩa** thì `evidenced` cắt
được bất kỳ cửa sổ quá khứ nào, và không tốn RAM.

### Payload push (bước 6)

Theo đặc tả nghiệp vụ:

| Tham số | Nguồn |
|---|---|
| Ngày giờ nhận dạng | `anchor_ts` (mốc `crane_op`) |
| Số xe đầu kéo | `truck_no` ← `TCODE01` |
| Số container | `slots[0].container_no` ← `CCODE01` |
| Số container 2 *(nếu có)* | `slots[1].container_no` |
| Số làn xe đầu kéo | `lane` ← `CRANE01` |
| Link ảnh 6 mặt | `slots[*].container_image` ← `evidenced` → MinIO |

---

## II. Quy trình hàng xuất (`ix_cd = X`)

> Hàng xếp từ xe đầu kéo trên bờ ⇒ xuống tàu

| # | Bước nghiệp vụ | Rule / service | Camera | Signal phát ra |
|---|---|---|---|---|
| 1 | Xe chở container chạy vào vị trí dưới bụng cẩu | `CRANE01`, `CRANE02` | 10 | `lane_active`, `truck_stable` |
| 2 | Nhận dạng số xe đầu kéo | `TCODE01` | 3, 5 | `truck_no` |
| 3 | Nhận dạng số container (2 mã nếu 2 cont 20′) | `CCODE01`, `CRANE04` | 1,4,6,7,8 / 10 | `container_no`, `cont_position` |
| 4 | **Push API lần 1** — ngày giờ + số container | `orchestratord` → `syncd` | — | `craneops.events` (revision 1) |
| 5 | Chụp **5 mặt** container (chưa có đáy) | `evidenced` | 1,2,4,6,7,8,10 | — |
| 6 | Cẩu QC bốc container từ xe xuống tàu | `CRANE03` | 10 | `crane_op` ← **mốc neo** |
| 7 | Chụp **mặt đáy** container | `evidenced` (job `slow`, `delay: 40s`) | 9 | — |
| 8 | **Push API lần 2** — đầy đủ | `orchestratord` → `syncd` | — | `craneops.events` (revision N) |

**Vì sao chỉ 5 mặt ở bước 5:** container đang nằm trên rơ-moóc, đáy bị che.

**Vì sao mặt đáy ở bước 7:** cẩu đã nhấc container lên khỏi xe, đáy mới lộ ra. Nên với
hàng **xuất** (`ixCd == "X"`) phải **chờ 30 giây** sau mốc `crane_op` rồi mới chọn khung,
và chọn theo chiều ngược lại. Với hàng nhập thì không chờ.

### Payload push lần 1 (bước 4)

| Tham số | Nguồn |
|---|---|
| Ngày giờ nhận dạng | thời điểm nhận dạng xong mã |
| Số container | `slots[*].container_no` |

### Payload push lần 2 (bước 8)

Giống bảng payload của hàng nhập — đầy đủ 6 trường.

---

## III. Luồng bắn event end-to-end

```
                    HÀNG NHẬP (I)                         HÀNG XUẤT (X)
                    ─────────────                         ─────────────
 t0   xe vào        CRANE01 lane_active                   CRANE01 lane_active
                    CRANE02 truck_stable                  CRANE02 truck_stable
        │                   │                                     │
 t1   số xe         TCODE01 ──► truck_no                  TCODE01 ──► truck_no
        │                   │                                     │
 t2                                                       CCODE01 ──► container_no
                                                          CRANE04 ──► cont_position
                                                                  │
 t3                                                       ══► PUSH API #1
                                                              (giờ + số cont)
        │                                                         │
 t4                                                       evidenced: 5 mặt
        │                                                         │
 t5   cẩu bốc       CRANE03 ──► crane_op ⚓                CRANE03 ──► crane_op ⚓
        │                   │                                     │
 t6                 evidenced: 6 mặt                      evidenced: mặt đáy
                    (cont đang treo,                      (chờ 30 s sau ⚓)
                     thấy cả đáy)
        │                   │                                     │
 t7   số cont       CCODE01 ──► container_no
                    CRANE04 ──► cont_position
        │                   │                                     │
 t8                 ══► PUSH API                          ══► PUSH API #2
                        (đầy đủ)                              (đầy đủ)
```

Điểm khác biệt cốt lõi: **`container_no` đến trước mốc neo ở hàng xuất, sau mốc neo ở hàng
nhập.** Mọi thứ khác — cửa sổ evidence, điều kiện hoàn thành, số lần push — đều suy ra từ
đó.

### Đường đi của message

```
ds_app ──craneops.perception.{crane,tcode,ccode}──► ruled@{crane,tcode,ccode}
                                                                        │
                                                            craneops.signals
                                                                        ▼
                                                              orchestratord
                                                    (chọn nhánh I/X theo ix_cd từ manifest)
                                          ┌─────────────────────────┴──────────────┐
                              craneops.evidence.{fast,slow}                    outbox
                                          ▼                                        ▼
                                     evidenced ──craneops.events──►             syncd
                                     → MinIO                              → API hệ thống Cảng
```

---

## IV. Khoảng trống giữa đặc tả nghiệp vụ và thiết kế hiện tại

Bốn điểm đặc tả này làm lộ ra, cần chốt trước Phase 6.

### G-1 · Đẩy API theo ngữ nghĩa **upsert**, không phải hai mốc cố định

Đặc tả nghiệp vụ nêu hai thời điểm đẩy dữ liệu (bước 4 và bước 8), nhưng mô hình hoá đúng
hai mốc đó là sai — các trường của một sự kiện được điền dần và **không theo thứ tự cố
định**: số container đọc xong trước, ảnh và video xong sau, số xe đầu kéo có thể tới muộn,
container thứ hai của chuyến twin-lift xuất hiện sau nữa.

Nên: **đẩy N lần cho mỗi sự kiện, khoá theo `event_id`, ngữ nghĩa upsert.** Lần đầu ngay
khi đã có `lane` và `container_no`; mỗi lần sau là một trường vừa được điền thêm. Bước 4 và
bước 8 của đặc tả chỉ là hai lần trong chuỗi đó.

Hệ quả thiết kế: `EventMessage` mang thêm `revision` tăng dần; outbox khử trùng theo
`(event_id, revision)` và bảo đảm thứ tự. Không cần khái niệm `push_stage`.

⚠️ Điều kiện đẩy lần đầu **không** đòi phải có ảnh/video. Sự kiện có `lane` + `container_no`
là đã đẩy được; ảnh và video tới sau bằng các lần upsert tiếp theo. Đòi đủ ảnh mới đẩy sẽ
làm dashboard thấy dữ liệu muộn hơn nhiều so với đặc tả.

### G-2 · Composition spec cần rẽ nhánh theo `ix_cd`

`configs/operations/crane_operation.yaml` hiện có **một** `anchor` và **một** bộ cửa sổ
evidence. Nhưng hai chiều hàng khác nhau ở: thứ tự OCR so với mốc neo, tập camera chụp
(6 mặt vs 5+1), độ trễ chụp đáy (0 vs 30 s), và số lần push.

*Hướng:* tách thành hai profile trong cùng file, chọn theo `ix_cd`; phần chung để ở
`defaults`.

### G-3 · `ix_cd` biết muộn ở hàng nhập

`ix_cd` lấy từ manifest Oracle sau khi có `container_no`. Với hàng nhập thì đó là **sau**
khi đã phải chụp 6 mặt.

*Hệ quả:* `evidenced` không thể quyết định "chụp mấy mặt, chờ bao lâu" tại thời điểm
`crane_op`. Hai cách:

1. **Chụp theo kịch bản rộng nhất rồi lọc sau** — luôn cắt đủ 6 mặt; nếu hoá ra là hàng
   xuất thì bỏ ảnh đáy chụp sớm và cắt lại sau 30 s. Tốn thêm I/O, nhưng đơn giản và không
   bao giờ mất dữ liệu.
2. **Trì hoãn job evidence cho tới khi biết `ix_cd`** — chờ `container_no`. Rủi ro: nếu OCR
   thất bại thì không có evidence để triage.

*Nghiêng về (1)*, vì segment mp4 nằm sẵn trên đĩa nên cắt lại là rẻ, và giữ được evidence
cho cả trường hợp OCR hỏng — đúng thứ cần nhất khi đi tìm nguyên nhân.

### G-4 · Payload gửi dashboard rộng hơn đặc tả

Đặc tả nghiệp vụ liệt kê 6 tham số, nhưng dashboard e-port đang nhận thêm `vslCd`,
`callSeq`, `callYear`, `ixCd`, `sztp`, `chassisPosition`, `shortVideo`. **Cần xác nhận
dashboard thực sự dùng những trường nào** — nếu phần thừa không ai đọc thì
`gateway/contract/dashboard.py` bỏ được và `ContainerSlot` gọn đi đáng kể.

---

## V. Chưa rõ — cần hỏi phía Cảng

1. ~~**G-1**: hàng xuất có cần đẩy sớm không?~~ ✅ Đã rõ: đẩy nhiều lần theo ngữ nghĩa
   upsert, lần đầu ngay khi có lane + số container.
2. **G-4**: dashboard dùng những trường nào trong payload? Có trường nào bắt buộc mà đặc tả
   quên liệt kê không?
3. Hàng **nhập** có cần chụp lại mặt đáy sau khi hạ container không, hay ảnh lúc treo là đủ?
4. Với xe chở **2 container 20′**: hai container có được chụp/đẩy trong **một** event, hay
   là hai event riêng? Đặc tả ghi "gửi API có 2 containers" ⇒ một event, khớp với thiết kế
   `slots: list[ContainerSlot]` hiện tại. Cần xác nhận.
5. "Ngày giờ nhận dạng" là thời điểm nào — lúc đọc xong mã container, hay lúc cẩu bốc
   (`crane_op`)? Hai mốc này cách nhau vài chục giây.
