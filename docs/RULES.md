# Đặc tả rule

Mỗi solution nghiệp vụ là một rule đăng ký được qua `@register_rule`. Rule tiêu thụ
`craneops.perception.*`, giữ state của riêng nó, và phát `craneops.signals`.
Tầng điều phối (`orchestratord`) kết hợp các signal đó thành sự kiện thao tác cẩu —
xem `configs/operations/crane_operation.yaml`.

**Rule này chạy vào lúc nào trong một chu kỳ thao tác thật** — và vì sao thứ tự khác nhau
giữa hàng nhập và hàng xuất — xem [USE_CASES.md](USE_CASES.md).

**Rule không biết gì về nhau.** Mọi liên kết đi qua signal, vì
camera giao tiếp bằng cách sửa biến class global dùng chung.

## Bảng tổng hợp

| Code | Tên | Vai trò | Nhóm process | `kind` signal phát ra |
|---|---|---|---|---|
| `CCODE01` | Nhận dạng mã container | ccode | `ccode` | `container_no` |
| `TCODE01` | Nhận dạng số đầu kéo | tcode | `tcode` | `truck_no` |
| `CRANE01` | Gán lane từ đầu kéo | crane | `crane` | `lane_active` |
| `CRANE02` | Xe vào vị trí ổn định | crane | `crane` | `truck_stable` |
| `CRANE03` | Cẩu đang thao tác | crane | `crane` | `crane_op` ← **mốc neo** |
| `CRANE04` | Suy kích thước container | crane | `crane` | `cont_dim`, `cont_position` |
| `BOT01` | Ảnh soi đáy | bottom | `bottom` | `bottom_ready` |

---

## `CRANE01` — Gán lane từ đầu kéo

Camera 10 nhìn xuống khu vực dưới cẩu. PicoDet cho bbox đầu kéo. Lane xác định bằng
**điểm mốc của bbox nằm trong đa giác nào** (`internal/pkg/geometry.py:LaneZones`).

| Config | Mặc định | Ghi chú |
|---|---|---|
| `lane1_zone` … `lane3_zone` | — | mỗi làn **một** đa giác, toạ độ **tương đối `[0..1]`**; để trống = camera không thấy làn đó |

> Config rule nằm ở `configs/rules/<CẨU>/<RULE>/config.json`, **khoá theo `camera_code`**,
> KHÔNG nằm trong `configs/cranes/*.yaml`. Ranh giới ba tầng: xem `common/rule_config.py`.
| `lane_anchor` | `CENTER` | Điểm mốc của bbox — tâm bbox đầu kéo |
| `head_thresh` | 0,6 | ngưỡng tin cậy tối thiểu cho bbox đầu kéo |

Trả về `Lane` hoặc **`None`** nếu nằm ngoài mọi lane. Điều này chỉ diễn đạt được với đa giác: với
hai đường thì mọi điểm trong ảnh đều rơi vào một dải.

⚠️ Cách cũ dùng **hai đường phân chia** và cần một bản `get_head_lane` riêng cho camera crane
lẫn tcode, cùng hình dạng nhưng **ngược dấu** vì hai camera nhìn từ hai hướng. Với đa giác
thì mỗi camera khai vùng của nó, không cần cờ đảo dấu. Xem
[DESIGN_NOTES.md](DESIGN_NOTES.md) DN-002.

---

## `CRANE02` — Xe vào vị trí ổn định

Bbox đầu kéo **không dịch chuyển trong `stable_duration`** ⇒ `truck_stable`.

Hai lựa chọn thiết kế, đều để bỏ phụ thuộc vào thứ trôi được theo thời gian:

| | Cách thường gặp | Ở đây |
|---|---|---|
| Cửa sổ | 3 khung liên tiếp ≈ 0,9 s ở 3,3 fps — **trôi theo fps** | `stable_duration`, mặc định **3,0 s** |
| Ngưỡng dịch chuyển | 3 px tuyệt đối | `stable_move_ratio` — tỉ lệ so với đường chéo bbox |

Đếm theo khung trói định nghĩa "ổn định" vào fps của nguồn: đổi cấu hình camera là vô tình
đổi luôn ngưỡng nghiệp vụ, và không có gì báo.
Ngưỡng 3 px tuyệt đối thì phụ thuộc độ phân giải và khoảng cách xe tới camera — cùng lý do
đã chốt ở [DESIGN_NOTES.md](DESIGN_NOTES.md) DN-002 Q3. `stable_move_ratio: 0.02` là giá
trị tương ứng 3 px trên bbox đầu kéo có đường chéo ~150 px.

⚠️ 3,0 s là ngưỡng **chặt**; OCR vì thế bắt đầu muộn hơn — phải đo xem có
bỏ lỡ chu kỳ nào không.

**`truck_stable` là cổng chặn OCR.** `CCODE01` chỉ chạy khi lane có `truck_stable`. Trước đây
cơ chế này đi qua biến global `BaseCamera.TRUCK_STABLE_POSITION_LANE_DICT`
trong cùng một process; nay đi qua topic `craneops.signals`. Bỏ cổng chặn nghĩa là
5 camera ccode chạy DB detection + SVTR recognition 24/7 kể cả khi dưới cẩu trống.

⚠️ **Không có khái niệm `truck_position`.** Vị trí trước/sau không suy từ cặp đường
`truck_lines` nữa — `CRANE04` xác định slot container trực tiếp bằng khoảng cách
container ↔ đầu xe. Xem [DESIGN_NOTES.md](DESIGN_NOTES.md) DN-001.

| Config | Mặc định | Ghi chú |
|---|---|---|
| `stable_duration` | 3,0 s | Cấu hình được |
| `stable_move_ratio` | 0,02 | Dịch chuyển tối đa, tỉ lệ so với đường chéo bbox |

⚠️ Chưa có timeout dự phòng để báo vị trí *dù chưa ổn định*. Ngưỡng như vậy sẽ phụ
thuộc `ContainerCodeCamera.DATABASE` — biến class global đọc chéo từ package khác
Phần đó gắn với `truck_position` vốn đã bị bỏ (DN-001);
cần quyết định có giữ timeout dự phòng cho `truck_stable` không.

---

## `CRANE03` — Cẩu đang thao tác ⭐ mốc neo

Tâm bbox container nằm trong `near_px` **phía trên** tâm bbox đầu kéo ⇒ đánh dấu
`crane_op` tại `frame_ts`. **Đây là mốc neo cho toàn bộ sự kiện** — mọi cửa sổ ảnh và
clip đều tính từ nó.

| Config | Mặc định | Ghi chú |
|---|---|---|
| `near_px` | 60 (25 với `GC01`) | ⚠️ Phụ thuộc góc camera nên **phải** cấu hình theo từng cẩu, không hardcode |
| `deactivate_timeout` | 25 s | |
| `deactivate_timeout_on_position_change` | 1 s | |
| `container_thresh` | 0,33 | ngưỡng tin cậy tối thiểu cho bbox container |

---

## `CRANE04` — Suy kích thước container

`crane_width / frame_width ≤ width_ratio_20ft` ⇒ `20feet`, ngược lại `40feet`.

Slot xác định bằng **khoảng cách container ↔ đầu xe**: sắp xếp các container theo độ gần
đầu xe, gần nhất là `slot_index = 0`. Rồi `ContainerPosition.for_slot(cont_dim, slot_index)`:

| `cont_dim` | `truck_position` | `cont_position` | mã gửi dashboard |
|---|---|---|---|
| 40feet | — | `40feet` | `""` |
| 20feet | `"1"` | `20feet-1` | `"F"` |
| 20feet | `"2"` | `20feet-2` | `"A"` |

Cả quy tắc dẫn xuất lẫn mã dashboard đều nằm trên `ContainerPosition`
(`ContainerPosition.derive()` và `.chassis_code`). Rule **không** phát signal
`chassis_position` riêng — nó suy 1:1 từ `cont_position`, và để hai giá trị đi riêng
chỉ tạo thêm một đường cho chúng mâu thuẫn. Tách làm hai trường trên `Event`
(`chassisPosition`, `chassisPosition2`) cùng một bảng tra riêng ở
quy ước gửi dashboard.

| Config | Mặc định |
|---|---|
| `width_ratio_20ft` | 0,45 |

---

## `CCODE01` — Nhận dạng mã container

Rule nặng nhất. Chuỗi xử lý (phần det+rec đã đẩy xuống Triton, dạng BLS — DN-007):

1. **Vùng OCR tĩnh** từ config ds_app (`ocr_rois`) → `nvdspreprocess` crop và batch.
   Vùng mang sẵn `lane` và `cont_dim` vì probe phải điền chúng vào `OcrResult`.
2. **DB text detection** → lấy `top_k` vùng theo diện tích.
3. **Cổng độ nét**: điểm FFT magnitude ≥ `sharpness_min` mới cho qua.
4. **SVTR CTC recognition** → chuỗi + confidence.
5. **Phân loại từng chuỗi**:
   - `iso`: dài 4, < 3 ký tự chữ
   - mã đầy đủ: dài ≥ 11
   - `part-1`: dài 4, ≥ 3 ký tự chữ
   - `part-2`: dài ≥ 6
6. **Ghép mảnh** part-1 + part-2 theo khoảng cách tâm ≤ `pair_distance_px`.
7. **Sửa mã**: check digit ISO 6346, bảng nhầm lẫn OCR, bảng sửa owner-prefix (~180 dòng),
   cắt bớt khi dài > 11.
8. **Streak voting**: cùng một mã xuất hiện ≥ `min_streak` frame liên tiếp mới phát signal.

| Config | Mặc định | Ghi chú |
|---|---|---|
| `top_k` | 5 | số hộp giữ lại sau khi xếp theo diện tích |
| `sharpness_min` | 1000 | dưới ngưỡng này thì bỏ crop, không đưa vào OCR |
| `pair_distance_px` | 60 | khoảng cách tâm tối đa để ghép part-1 với part-2 |
| `bitmap_threshold` | 0,1 | ⚠️ hai ngưỡng này quyết định độ chính xác số học |
| `box_threshold` | 0,2 | được phép của detector — xem DESIGN_NOTES DN-013 |
| `character_threshold` | 0,3 | ký tự dưới ngưỡng bị loại trước khi bỏ lặp CTC |
| `iso_threshold` | 0,95 | |
| `ocr_threshold` | 0,95 | ngưỡng chấp nhận chuỗi đọc được — lọc **sau** khi đọc, không ở từng vùng |
| `min_streak` | 3 | số khung liên tiếp cùng một mã mới phát signal |

⚠️ **Tri thức miền.** Bước 5–7 phải port gần nguyên văn và golden-test trước khi refactor.

---

## Chiều ngược — HOÃN

Nghiệp vụ có tình huống mã container xuất hiện ở một lane mà **không** phát hiện đầu kéo
dưới cẩu (xe vào ngược chiều). v1 xử lý bằng một nhánh riêng với ngưỡng chặt hơn.

**Tạm hoãn, không nằm trong phạm vi hiện tại.** Ghi lại ở đây vì hai lý do:

* Nó **không phải** một nhánh code thứ hai — chỉ khác vài ngưỡng
  (`character_threshold` 0,3 → 0,5; `min_streak` 3 → 5; thêm `score_threshold`,
  `min_cont_count`). Khi làm, nó là một *cấu hình* khác của cùng chuỗi xử lý. Nhân đôi
  pipeline nghĩa là mọi sửa lỗi sau này phải nhớ sửa hai chỗ.
* `Direction` vẫn còn trong `common/message.py` và hiện **luôn** là `RIGHT`. Giữ lại vì
  gỡ khỏi hợp đồng message rồi thêm lại là hai lần đổi schema cho cùng một thứ.


## `TCODE01` — Nhận dạng số đầu kéo

Camera 3 và 5. PicoDet phát hiện đầu kéo → FastViT **phân loại** số xe.

> Số đầu kéo là **tập đóng ~130 lớp** (`assets/camera-truckNo/cls-truckHead/samples/`),
> nên đây là classifier chứ không phải OCR. Ánh xạ đúng vào mô hình PGIE→SGIE của
> DeepStream — cùng cơ chế mà DeepStream dùng để gắn thuộc tính lên bbox.

| Config | Mặc định | Ghi chú |
|---|---|---|
| `head_thresh` | 0,8 | ngưỡng tin cậy của PicoDet cho bbox đầu kéo |
| `head_code_thresh` | 0,93 | ngưỡng của classifier — cao hơn vì tập lớp đóng |
| `min_streak` | 3 | số khung liên tiếp cùng một số xe mới phát signal |

Gán lane dùng chung vùng làn như `CRANE01` — mỗi camera khai `lane1_zone`…`lane3_zone`
riêng trong `configs/rules/<cẩu>/TCODE01/config.json`, không còn cờ đảo dấu theo hướng
camera (xem [DESIGN_NOTES.md](DESIGN_NOTES.md) DN-002).

Bỏ phiếu đa số giữa camera 3 và 5 là việc của **combinator** `majority_vote` ở tầng
điều phối, không phải của rule.

---

## `BOT01` — Ảnh soi đáy

Camera 9 nhìn đáy container. **Không chạy model.** Rule chỉ báo khi đủ điều kiện chụp;
việc dựng ảnh (CLAHE + ghép 2×2) là của `evidenced`.

| Config | Mặc định | Nguồn |
|---|---|---|
| `post_op_delay` | 30 s | Chỉ khi `ix_cd == "X"` (hàng xuất) |
| `frames_per_mosaic` | 4 | số khung ghép vào một ảnh mosaic 2×2 |
| `mosaic_count` | 3 | số ảnh mosaic sinh cho mỗi sự kiện |
| `frame_interval` | 20 s | khoảng cách giữa các khung được chọn |

---

## Thêm một rule mới

1. Tạo `internal/rules/<code>.py` với `@register_rule(...)` và `config_model`.
2. Thêm `RuleSpec` vào `internal/rules/configs.py`, rồi `make rules` +
   `python -m tools.rule_configs init <CẨU>` sinh `config.json`, `schema.json`,
   `changelog.md` cho mọi cẩu. Khoá là `camera_code`.
3. Chạy `make schema` để sinh `schema.json` từ `config_model`.
4. Thêm code vào `rule_groups` trong `configs/default.yaml`.
5. Viết test trong `tests/unit/rules/test_<code>.py`.

Không phải copy entrypoint. Không phải sửa file dùng chung nào.

> Không có registry thì thêm một rule nghĩa là copy ~190 dòng `services/<RULE>/main.py`,
> thêm một service vào docker-compose, thêm vào `ENABLED_RULES`, **và** sửa các bảng
> hardcode trong `common/config.py` (`max_time_for_event`, `fps_for_event`,
> `event_type_names`). Nhân với ~30 rule thành ~5 700 dòng gần như giống hệt nhau.
