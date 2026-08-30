# Ghi chú thiết kế — các quyết định còn mở

> **Số đo nằm ở `HARDWARE_BUDGET.md`, không phải ở đây.** File này ghi *quyết định và lý
> do*; file kia là nguồn sự thật duy nhất cho các con số. Chép bảng số sang cả hai nơi thì
> chúng sẽ trôi khỏi nhau, và bản ít được chạy lại hơn sẽ là bản người ta tin.
>
> Mỗi mục ghi một quyết định **đã chốt và đã đo**, kèm lý do đủ để người sau biết đổi nó
> sẽ phá cái gì. Đây không phải nhật ký — nếu một quyết định không còn ràng buộc điều gì
> thì xoá mục đó đi.

Nơi ghi những thay đổi thiết kế đã chốt hướng nhưng chưa chốt chi tiết. Mỗi mục phải nêu:
*đổi cái gì*, *vì sao*, *câu hỏi còn mở*, và *cách kiểm chứng*.

Khi một mục được chốt xong và code xong, chuyển nội dung sang `RULES.md` / `ARCHITECTURE.md`
rồi xoá khỏi đây.

---

## DN-001 · Xác định `ContainerPosition` bằng khoảng cách container ↔ đầu xe

**Trạng thái:** hướng đã chốt · chi tiết còn mở · áp dụng ở Phase 5 (`CRANE04`)
**Ngày:** 2026-08-29

### Đổi cái gì

Bỏ hoàn toàn khái niệm `TruckPosition` (vị trí đỗ trước/sau của đầu xe, suy từ hai đường
dọc cấu hình `truck_lines`). Thay bằng **vị trí tương đối giữa thùng container và đầu xe**,
dùng ngưỡng khoảng cách.

```
Cách cũ: tâm bbox đầu kéo → thuộc dải dọc nào (truck_lines) → truck_position "1"|"2"
     → kết hợp cont_dim → cont_position

v2:  đo khoảng cách container ↔ đầu xe → xếp hạng slot theo độ gần
     → kết hợp cont_dim → cont_position
```

### Vì sao tốt hơn

1. **Bỏ được một bộ hiệu chỉnh tuyệt đối theo pixel.** `truck_lines` là toạ độ tuyệt đối
   trong khung hình (GC03 cam 10: `284-175-213-713_506-174-483-713`). Camera bị xê dịch,
   đổi zoom, hay thay thiết bị là phải hiệu chỉnh lại. Khoảng cách *tương đối* giữa hai
   vật thể trong cùng khung hình thì miễn nhiễm với việc đó.
2. **Đo đúng thứ ta quan tâm.** `truck_position` là *proxy*: nó nói xe đỗ ở đâu, rồi ta
   suy ra container nào đang được làm hàng. Khoảng cách container ↔ đầu xe đo **trực tiếp**
   container nào. Bớt một bước suy diễn là bớt một chỗ sai.
3. **Không phụ thuộc chiều xe vào.** Nếu dùng trị tuyệt đối của khoảng cách, xe vào từ
   trái hay phải đều cho cùng kết quả. Cách dùng đường thì phải biết hướng.
4. **Twin-lift 20 ft trở nên tự nhiên.** Hai container cùng lúc ⇒ sắp xếp theo khoảng cách
   tới đầu xe: gần nhất là slot 1, xa hơn là slot 2. Cách cũ cần mỗi container rơi đúng
   dải dọc của nó.

### Hình học thực tế (GC03, camera 10 "Trần container")

Đọc từ config thật:

* `lane_lines = 1-351-1271-363_0-530-1271-557` → hai đường **gần ngang** (y ≈ 351–363 và
  y ≈ 530–557, x trải 0→1271) ⇒ **lane xếp chồng theo chiều dọc ảnh**.
* `truck_lines = 284-175-213-713_506-174-483-713` → hai đường **gần dọc** (x ≈ 284→213 và
  x ≈ 506→483, y trải 174→713) ⇒ vị trí trước/sau đo **theo trục ngang ảnh**.

Suy ra: **xe chạy theo phương ngang của ảnh**, lane là các dải ngang. Vậy khoảng cách
container ↔ đầu xe nên đo **theo trục x**.

⚠️ Toạ độ trên là trong khung **1280×720** (camera 10 được downscale về `720p` —
xem `HARDWARE_BUDGET.md` §2.3). Nếu v2 chạy camera này ở độ phân giải khác thì mọi ngưỡng
pixel phải quy đổi.

### Câu hỏi còn mở

**Q1. Đo khoảng cách giữa hai điểm nào?**

| Phương án | Ưu | Nhược |
|---|---|---|
| `|cx_container − cx_head|` (tâm–tâm) | đơn giản nhất | phụ thuộc chiều dài container ⇒ 20 ft và 40 ft cho giá trị rất khác nhau |
| Khe hở giữa mép sau đầu xe và mép gần của container | bám sát hình học thật | nhạy với bbox bị cắt cụt ở biên khung |
| Chiếu tâm container lên trục dọc thân xe | đúng nhất khi xe chéo | cần ước lượng trục thân xe |

*Nghiêng về:* tâm–tâm theo trục x, vì camera nhìn từ trên và xe chạy ngang; nhưng phải
**chuẩn hoá** (xem Q2).

**Q2. Ngưỡng theo pixel tuyệt đối hay chuẩn hoá?**

Ngưỡng pixel phụ thuộc độ cao lắp và zoom. Đề xuất chuẩn hoá theo bề rộng bbox container
hoặc bề rộng đầu xe:

```
d_norm = |cx_container − cx_head| / w_container
```

Như vậy một ngưỡng dùng chung được cho mọi cẩu. Cần đo trên dữ liệu thật để biết hai cụm
(slot 1 / slot 2) có tách rời rõ không.

**Q3. Không phát hiện được đầu xe thì sao?**

Không phát signal `cont_position`, và **không đoán**. Câu hỏi còn mở: có cần timeout dự
phòng (bậc 3-7 s) để phát một giá trị suy đoán khi đầu xe khuất quá lâu, hay im lặng là
đúng? Im lặng thì orchestrator chờ, và event có thể hết hạn.

**Q4. Vai trò B của `truck_position` thay bằng gì?** ⚠️ **Đây là phần chưa có lời giải.**

Khái niệm `truck_position` trước đây còn được **camera ccode** dùng, qua biến toàn cục
`BaseCamera.TRUCK_STABLE_POSITION_LANE_DICT`, cho hai việc:

* **Cổng chặn**: `truck_position` rỗng ⇒ **bỏ qua OCR hoàn
  toàn**. Đây là cơ chế tiết kiệm tính toán đáng kể — và trên RTX 3060 thì càng đáng giữ.
* **Gán slot** (`:744-756` `get_cont_position`): ánh xạ `(truck_position, cont_dim)` →
  `cont_position` cho *kết quả OCR*, để biết mã vừa đọc thuộc container nào.

Ba hướng thay thế:

| Hướng | Cách làm | Đánh đổi |
|---|---|---|
| **(a) Orchestrator ghép** | `CCODE01` chỉ phát `container_no` kèm `cont_dim` của ROI; `CRANE04` phát `cont_position`; orchestrator ghép theo thời gian gần nhau | Sạch nhất về kiến trúc, rule không biết nhau. Nhưng **mất cổng chặn** ⇒ OCR chạy liên tục ⇒ tốn GPU |
| **(b) `ruled@ccode` nghe thêm `craneops.signals`** | Đăng ký `needs=["craneops.signals"]`, lọc lấy `truck_stable` / `cont_position` của `CRANE02`/`CRANE04` | Giữ được cổng chặn. Coupling qua **hợp đồng đã công bố**, không phải biến global — chấp nhận được |
| **(c) Orchestrator phát "trạng thái lane"** | Orchestrator tổng hợp rồi phát một topic trạng thái mà rule nghe | Rõ ràng nhất, nhưng thêm một vòng và một topic |

*Nghiêng về (b)* vì ngân sách GPU: 5 camera ccode chạy OCR liên tục là lãng phí lớn nhất
có thể tránh được. Trường `needs` trong `@register_rule` đã dự trù sẵn cho việc này.

Định lượng giá trị của cổng chặn: 5 camera ccode × 5 fps = **25 khung/giây** đi qua OCR,
mỗi khung 1–2 ROI. Không có cổng chặn thì chừng ấy DB detection + SVTR recognition chạy
24/7 kể cả khi dưới cẩu trống. Có cổng chặn thì chỉ chạy trong lúc thực sự có xe đỗ.

**Q5. Cổng chặn nên dựa vào tín hiệu nào?** ✅ **ĐÃ CHỐT: `truck_stable`.**

Cổng chặn là `truck_stable` từ `CRANE02`, và **định nghĩa "ổn định" đổi từ đếm khung sang
đếm thời gian**:

| | Cách thường gặp | Ở đây |
|---|---|---|
| Điều kiện | tâm bbox dịch ≤ 3 px trong ≥ 3 khung liên tiếp | bbox không dịch chuyển trong `stable_duration` |
| Cửa sổ | 3 khung ≈ 0,9 s ở 3,3 fps — **trôi theo fps** | **3,0 s**, cấu hình được |
| Ngưỡng dịch chuyển | 3 px tuyệt đối | tương đối theo kích thước bbox |

Hai lý do đổi:

* **Đếm khung trôi theo fps.** "3 khung" nghĩa là 0,9 s với camera crane (3,3 fps).
  Đổi `drop_frame_interval` là vô tình đổi luôn định nghĩa ổn định. Đếm theo giây thì không.
* **3 px tuyệt đối phụ thuộc độ phân giải và khoảng cách.** Cùng lý do đã chốt ở DN-002 Q3.
  Chuẩn hoá theo đường chéo bbox: 3 px trên một bbox đầu xe đường chéo ~150 px ≈ **2 %**,
  nên `stable_move_ratio: 0.02` là giá trị mặc định.

Lưu ý khi kiểm chứng: 3,0 s là ngưỡng chặt, nên OCR sẽ bắt đầu muộn hơn. Cần đo
xem có bỏ lỡ chu kỳ nào không trong golden test.

### Ảnh hưởng tới những gì đã có

| Hạng mục | Thay đổi |
|---|---|
| `common/enum.py` | ✅ Bỏ `TruckPosition`; `ContainerPosition.derive()` → `for_slot(cont_dim, slot_index)` |
| `common/enum.py` | ✅ Bỏ `SignalKind.TRUCK_POSITION` |
| `configs/cranes/*.yaml` | Bỏ `truck_lines` khỏi camera crane; thêm ngưỡng khoảng cách |
| `RULES.md` `CRANE02` | Chỉ còn phát `truck_stable`, không còn `truck_position` |
| `RULES.md` `CRANE04` | Đầu vào đổi từ `truck_position` sang hình học container ↔ đầu xe |
| `RULES.md` `CCODE01` | Cần mục mới cho cổng chặn + nguồn `cont_position` (chờ Q4) |

### Cách kiểm chứng

Đây là **thay đổi hành vi**, không phải refactor. Golden test sẽ khác biệt một cách hợp lệ,
nên cần một bước đối chiếu riêng:

1. Trên bộ video golden, chạy song song hai cách trên **cùng** đầu vào detection:
   phân dải theo đường dọc và xếp hạng theo khoảng cách.
2. Lập bảng chéo `cont_position` giữa hai cách. Mục tiêu: khớp ≥ ngưỡng đã thống nhất
   trên các chu kỳ có nhãn.
3. Với mọi trường hợp lệch, xem lại thủ công để biết **cách nào đúng**. Mục tiêu là đúng
   hơn, không phải giống hơn — nên một ca lệch mà cách mới đúng là kết quả tốt.
4. Bắt buộc phủ: 40 ft, 20 ft slot 1, 20 ft slot 2, twin-lift hai container cùng lúc,
   và trường hợp không phát hiện được đầu xe.
5. Dữ liệu từ bước 2 cũng chính là dữ liệu để **chọn ngưỡng** ở Q2 — vẽ phân bố `d_norm`
   và kiểm tra hai cụm có tách rời không.


---

## DN-002 · Lane là **đa giác**, không phải hai đường phân chia

**Trạng thái:** đã chốt hướng · `internal/pkg/geometry.py` đã hiện thực · còn phần config + converter
**Ngày:** 2026-08-29

### Đổi cái gì

```
Cách cũ: lane_position_config = "1-351-1271-363_0-530-1271-557"   (hai đường)
     → point_side_of_line(tâm bbox) → lane "1" | "2" | "3"

v2:  lane_zones = {"1": [[x,y],...], "2": [...], "3": [...]}   (mỗi lane một đa giác)
     → point-in-polygon → Lane | None
```

### Vì sao tốt hơn

1. **Diễn đạt được hình thang.** Camera nhìn chéo từ trên xuống nên lane bị méo phối cảnh.
   Hai đường chỉ tạo được các dải song song.
2. **Diễn đạt được "ngoài khu vực".** Với hai đường thì **mọi** điểm trong ảnh đều rơi vào
   một dải — xe chạy ngang phía xa vẫn bị gán lane 1 hoặc 3. Đa giác trả `None`.
3. **Ranh giới tường minh.** Với hai đường, lane 2 là "khoảng ở giữa" — không viết ra được.
4. **Bỏ được cờ đảo dấu.** Cách cũ cần hai bản `get_head_lane` cho
   cho camera tcode, cùng hình dạng nhưng **ngược dấu** vì hai camera nhìn từ
   hai hướng. Với đa giác thì mỗi camera khai vùng của nó, không cần biết hướng.

### Kiểu lỗi mới mà cách này tạo ra

Đường thì không thể chồng lấn; đa giác thì có. `LaneZones.overlapping_lanes()` phát hiện
việc đó và **phải được gọi lúc validate config**, không để lộ ra lúc chạy.

### Quyết định: từ chối đa giác hỏng, không tự sửa

Một cách làm phổ biến là luôn áp `buffer(0)` rồi log cảnh báo và
chạy tiếp. v2 **mặc định ném lỗi**, chỉ sửa khi truyền `sanitize=True`.

Lý do, đã đo bằng shapely 2.1: hình nơ `[[0,0],[100,100],[100,0],[0,100]]` gồm hai tam
giác 2500 + 2500. `buffer(0)` trả về **một** Polygon diện tích **2500** — mất hẳn một nửa,
im lặng. Với vùng lane thì đó là mất nửa làn xe. Vùng lane là config do người vẽ và sửa
được, nên fail-fast để người vận hành vẽ lại là đúng hơn.

Cũng đã đo: đỉnh thẳng hàng và hình tự cắt **đều** cho `Polygon.area == 0`, nên không phân
biệt được bằng diện tích. Phân biệt bằng `buffer(0).is_empty` — thẳng hàng cho hình rỗng,
tự cắt thì không.

### Câu hỏi còn mở

**Q1. Dùng điểm mốc nào?** `Anchor.CENTER` (tâm bbox đầu kéo) là mặc định.
Với camera 10 nhìn từ trên thì tâm là hợp lý. Nhưng `Anchor.CENTER | Anchor.BOTTOM` chặt
hơn và trả `None` khi bbox vắt qua hai lane — có nên bật cho camera nào không?

**Q2. Converter từ config cũ.** Hai đường + biên khung ảnh **suy ra được** ba đa giác:
lane 1 = phía trên đường 1, lane 2 = giữa, lane 3 = phía dưới đường 2, cắt theo khung ảnh.
Nghĩa là `tools/convert_legacy_config.py` sinh được lane_zones **tự động**, golden test
chứng minh tương đương, rồi người vận hành mới tinh chỉnh lại theo phối cảnh thật.
Cần chốt: có làm converter tự động không, hay vẽ lại tay ngay từ đầu?

**Q3. Toạ độ theo độ phân giải nào?** ✅ **ĐÃ CHỐT: chuẩn hoá về `[0..1]`.**

Cấu hình cũ dùng toạ độ tuyệt đối trong khung 1280x720 (camera 10 bị downscale về
`720p`), nên đổi độ phân giải xử lý là phải hiệu chỉnh lại toàn bộ. Điều này rất dễ xảy ra
ở v2: ngân sách NVDEC có thể buộc camera crane chuyển sang sub-stream 640x360
(`HARDWARE_BUDGET.md` §2.3, phương án 2), và lúc đó mọi toạ độ pixel đều sai.

Hiện thực:

* Config lưu toạ độ **tương đối**, validate nằm trong `[0, 1]`.
* `LaneZones.from_config(raw, frame_size=(w, h))` là chỗ **duy nhất** quy đổi sang pixel.
* Sau khi dựng, mọi truy vấn đều nhận pixel — cùng hệ với bbox model trả về, không có chỗ
  nào phải nhớ đổi đơn vị.
* Toạ độ ngoài `[0, 1]` bị từ chối kèm thông báo nêu thẳng nguyên nhân hay gặp nhất:
  dán nhầm toạ độ pixel tuyệt đối.
* Test `test_same_config_at_two_resolutions_is_geometrically_equivalent` chốt lại: cùng một
  config cho cùng kết quả ở 1280x720 và 2688x1520.

Hệ quả cho DN-001: ngưỡng khoảng cách container ↔ đầu xe cũng **phải** chuẩn hoá (Q2 của
DN-001 đã nghiêng về `d_norm = |Δcx| / w_container`) — cùng lý do.

### Ảnh hưởng

| Hạng mục | Thay đổi |
|---|---|
| `internal/pkg/geometry.py` | ✅ đã có `PolygonZone`, `LaneZones`, `Anchor`, `denormalize` |
| `configs/cranes/*.yaml` | `lane_lines` → `lane_zones`, toạ độ **tương đối [0..1]** (camera 3, 5, 10 của GC03) |
| `tools/validate_config.py` | phải gọi `overlapping_lanes()` và fail nếu có |
| `tools/convert_legacy_config.py` | sinh đa giác từ hai đường cũ **rồi chia cho 1280x720** — xem Q2 |
| `RULES.md` `CRANE01`, `TCODE01` | đổi mô tả cách gán lane |


---

## DN-003 · Model phân loại số đầu kéo phải reparameterize trước khi xuất

**Trạng thái:** ✅ **HOÀN TẤT** · engine đã dựng, Triton phục vụ, độ chính xác xác minh 100 %
**Ngày:** 2026-08-29

### Cập nhật 2026-08-29 17:12 — file đã được thêm, và nó có ba lỗi export

`truckHeadCls_150125.t7` (17,3 MB) đã được đưa vào `assets/`. Giải mã được bằng đúng
mật khẩu cũ, `onnx.checker` pass. Nhưng khi soi cấu trúc thì:

```
tham số      : 3.31 M      (fastvit_t8 — đúng)
tổng node    : 6924
  Conv       : 3314        <- fastvit_t8 xuất đúng chỉ cần ~60-90
  Reshape    : 3264
  Sub        : 10          <- RepMixer CHƯA hợp nhất
Conv group=1 : 3292/3314   <- depthwise BỊ BUNG thành conv thường
input        : [1,224,224,3]  <- NHWC, batch cố định
producer     : tf2onnx 1.15.0, opset 12
```

**Lỗi 1 — chưa reparameterize.** 10 node `Sub`, tên node còn chứa `REPARAM` và `mixer_bn`.
Đúng thứ đã dự đoán từ `best_acc.h5`.

**Lỗi 2 — depthwise conv bị bung, nghiêm trọng hơn.** 1824 node `Conv` có kernel
`(1, 1, 7, 7)` — tức là **một kênh một node**. Cộng 720 node `(2,1,3,3)`, 384 node
`(2,1,1,1)`, 336 node `(2,1,7,7)`. Chỉ 22/3314 node có `group > 1`, tức là chỉ 22 depthwise
conv được chuyển đúng; số còn lại bị `tf2onnx` tách thành từng kênh rồi ghép lại bằng 3264
node `Reshape`.

Hệ quả: thay vì ~22 lần gọi kernel grouped-conv, GPU phải chạy hơn 3000 lần gọi kernel tí
hon. Model vẫn cho **kết quả đúng** — nên không có gì báo động — nhưng chậm hơn nhiều lần.

**Lỗi 3 — NHWC + batch cố định.** Khác 6 model kia (NCHW). Cần vá batch động; layout thì
nên xuất lại bằng `--inputs-as-nchw` cho đồng bộ.

### Đã chặn tự động

`tools/export_models.py:check_health()` nay từ chối dựng engine từ ONNX như vậy. Hai kiểm
tra, đều nhắm vào loại lỗi *chạy đúng nhưng chậm* — thứ không có gì báo động:

| Kiểm tra | Ngưỡng | Model này |
|---|---|---|
| Conv trên mỗi triệu tham số | ≤ 40 | **1002** ❌ |
| Còn khối reparameterize chưa hợp nhất | không được có | **10 node Sub + tên REPARAM** ❌ |

Sáu model kia đều đạt.

### ✅ Kết quả cuối

Xuất lại bằng `tools/export_headcode_cls.py --nchw --fold-preprocess`, mã hoá thành
`truckHeadCls_reparam.t7` (KHÔNG ghi đè `truckHeadCls_150125.t7` của bạn), rồi cho đi
chung đường với 6 model kia:

| | file `.t7` cũ | bản xuất lại |
|---|---:|---:|
| node | 6 924 | **183** |
| Conv | 3 314 | **52** |
| Sub (chưa hợp nhất) | 10 | **0** |
| Conv / M tham số | 1 002 | **16** |
| `check_health` | ❌ | **✅ ĐẠT** |
| Dựng engine | — | **83 s** |
| Top-1 qua Triton, TensorRT **FP16** | — | **100,0 %** (451 ảnh) |

FP16 **không** làm giảm độ chính xác — khớp đúng ONNX FP32. Rủi ro "FP16 lệch kết quả ở
biên" nêu trong kế hoạch đã được loại cho model này.

Tên tensor cũng đã dọn: tf2onnx sinh ra `input:0` / `Identity:0` (dấu `:0` là rác của
TensorFlow, và `trtexec` không nhận dấu hai chấm trong `--minShapes`). Đổi thành
`input` / `head`; bước đổi tên đã đưa vào `export_headcode_cls.py` để lần sau tự sạch.

### Cách xuất lại (đã dùng)

```bash
uv run --with tensorflow --with keras-cv-attention-models --with tf2onnx \
    python -m tools.export_headcode_cls --h5 <best_acc.h5> --out <out.onnx> --opset 13
```

Ba thứ phải khác so với lần xuất đã tạo ra file hiện tại:

1. **`switch_to_deploy()` trước khi xuất** — script đã ép làm và kiểm chứng đầu ra không đổi.
2. **opset ≥ 13** thay vì 12 — xử lý depthwise của `tf2onnx` tốt hơn ở opset mới.
3. **`--inputs-as-nchw input`** — cho đồng bộ với 6 model kia.

Sau khi xuất lại, `triton/modelsvc` sẽ tự chạy `check_health` trước khi dựng engine.

### Vấn đề gốc (giữ lại làm bối cảnh)

Cấu hình trước đây tham chiếu
`cls-truckHead/truckHeadCls_150125.t7`, nhưng trước 2026-08-29 file đó **không tồn tại** trong `assets/`.
Thứ thực sự có: `best_acc.h5` (41 MB, Keras HDF5) + `label.txt` (54 lớp). `notebook/` cũng
không có notebook export cho model này, chỉ có cho 3 model kia.

### Đã xác minh bằng h5py (không cần TensorFlow)

| Hạng mục | Giá trị |
|---|---|
| Kiến trúc | `fastvit_t8` |
| Keras | 2.15.0 |
| Số layer | 261 |
| Input | `[None, 224, 224, 3]` — **NHWC** |
| Output | `Dense(54, softmax)`, khớp `label.txt` |
| Thư viện dựng model | `keras_cv_attention_models` (layer `resmlp>ChannelAffine`) |

Phân bố layer: 70 BatchNorm · 46 Conv2D · 40 Add · 29 ZeroPadding2D · 22 DepthwiseConv2D ·
20 ChannelAffine · **10 Subtract**.

### ⚠️ Model đang ở dạng TRAIN-TIME, chưa hợp nhất

10 layer `Subtract` có tên nói thẳng ra điều đó:

```
stack1_block1_REPARAM_TWICE_out  <-  [stack1_block1_mixer_REPARAM_out, stack1_block1_mixer_bn]
stack1_block2_REPARAM_TWICE_out  <-  [...]
...  (10 khối, phân bố 2+2+4+2 đúng cấu hình fastvit_t8)
```

FastViT — và họ MobileOne nói chung — huấn luyện với **nhiều nhánh song song** rồi **hợp
nhất toán học** thành một convolution duy nhất khi suy luận. Hai dạng cho **cùng kết quả**,
nhưng dạng đã hợp nhất chạy nhanh hơn hẳn.

Xuất thẳng dạng train-time nghĩa là mang theo toàn bộ nhánh thừa: 10 khối RepMixer ×
(depthwise conv + 2 BN + Subtract + Add + ChannelAffine) thay vì 10 depthwise conv.
TensorRT fuse được một phần (Conv+BN), nhưng **không thể** tự làm phép hợp nhất của
reparameterization — nó không biết hai nhánh đó tương đương một conv.

### Cách làm

`tools/export_headcode_cls.py`, chạy trong môi trường tạm:

```bash
uv run --with tensorflow --with keras-cv-attention-models --with tf2onnx \
    python -m tools.export_headcode_cls --h5 <best_acc.h5> --out <out.onnx>
```

Script **luôn** kiểm chứng, không cho bỏ qua lặng lẽ:

1. Đối chiếu số lớp output với `label.txt`.
2. Đếm layer trước/sau hợp nhất — **không giảm thì dừng** (nghĩa là hợp nhất không chạy).
3. So đầu ra trước/sau trên dữ liệu ngẫu nhiên, `atol=1e-4`; lệch quá thì **không xuất**.
4. Kiểm `argmax` giữ nguyên.

TensorFlow **cố ý không** nằm trong dependency chính: quá nặng, và chỉ dùng một lần.

### Câu hỏi còn mở

**Q1.** ✅ Đã có file, nhưng không dùng được như hiện tại — xem phần cập nhật ở trên.

**Q2. NHWC vs NCHW.** ✅ Đã xác minh: file thật là **NHWC** `[1,224,224,3]`. `SPECS` đã
sửa cho khớp. Khi xuất lại nên dùng `--inputs-as-nchw` để đồng bộ với 6 model kia, rồi
sửa `SPECS` lần nữa.

**Q3. Tiền xử lý.** ✅ **ĐÃ CHỐT bằng thực nghiệm.**

Chạy ONNX đã xuất trên **451 ảnh có nhãn** ở `cls-truckHead/samples/`, thử 5 chế độ:

| chế độ | màu | top-1 | top-3 |
|---|---|---:|---:|
| **`torch` (`/255` rồi ImageNet mean/std)** | **RGB** | **100,0 %** | **100,0 %** |
| `torch` | BGR | 87,6 % | 97,1 % |
| `tf` (`/127.5 − 1`) | RGB | 99,6 % | 100,0 % |
| `inception` (`(x/255 − 0.5)/0.5`) | RGB | 99,6 % | 100,0 % |
| `raw01` (chỉ `/255`) | RGB | 96,0 % | 98,4 % |
| `raw` (không chuẩn hoá) | RGB | 3,1 % | 19,1 % |

⇒ Đúng phép tiền xử lý mà model được huấn luyện với:
`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, `color_mode=1` (RGB), `/255`.

⚠️ 100 % gần như chắc chắn vì đây là tập huấn luyện — con số này xác nhận **quy ước tiền
xử lý**, KHÔNG phải độ chính xác thực địa. Nhưng đó đúng là thứ cần biết.

Đáng chú ý: dùng nhầm BGR chỉ tụt xuống 87,6 % — đủ cao để *trông như đang chạy được*.
Đây là loại lỗi sống sót qua kiểm thử thủ công.

### Đã gấp chuẩn hoá vào model

DeepStream `nvinfer`/`nvinferserver` chuẩn hoá theo `y = net-scale-factor × (x − offset)` —
**một** hệ số vô hướng cho cả ba kênh. Nhưng std của ImageNet khác nhau theo kênh
(0,229 / 0,224 / 0,225), nên **không biểu diễn được** bằng cấu hình đó. Ép dùng std trung
bình thì sai lệch nhỏ nhưng âm thầm.

Nên `--fold-preprocess` chèn một layer `Rescaling` vào đầu model:
`scale = 1/(255·std)`, `offset = −mean/std` theo từng kênh.

Kết quả kiểm chứng:

| Đưa vào model | top-1 |
|---|---:|
| RGB **thô [0,255]**, không tiền xử lý gì | **100,0 %** ✅ |
| lỡ chia 255 trước | 1,6 % |

Cấu hình DeepStream vì thế chỉ cần `net-scale-factor=1.0`, không offset, `model-color-format`
= RGB. Không còn chỗ nào để cấu hình sai.


---

## DN-004 · Triển khai bằng Docker Compose, không phải systemd

**Trạng thái:** đã chốt · `build/docker-compose.triton.yml` đã có
**Ngày:** 2026-08-29

### Đổi cái gì

Kế hoạch ban đầu chọn **PyInstaller + systemd** cho các service Python. Không dùng được:
**môi trường triển khai không có quyền sudo**, mà tạo/bật systemd unit thì bắt buộc phải có.
Mọi thứ chạy qua Docker Compose.

Điều này thực ra làm nhẹ đi mâu thuẫn đã nêu từ đầu (§D.1 của kế hoạch): `ds_app` và Triton
vốn đã buộc phải là container. Giờ mọi service đi chung một cách đóng gói.

Hệ quả cần theo dõi: mô hình bảo vệ bằng license + model mã hoá vốn được thiết kế cho binary
PyInstaller. Trong container, mã nguồn Python nằm dạng rõ (được mount `:ro`). Nếu điều đó
không chấp nhận được thì phải bàn lại — nhưng xem mục dưới, ràng buộc phần cứng trên máy này
vốn đã yếu sẵn.

### ⚠️ Serial BIOS đọc ra KHÁC NHAU giữa host và container

Đo thật trên máy dev:

| Ngữ cảnh | `/sys/class/dmi/id/product_serial` |
|---|---|
| host, không phải root | `""` (không đọc được, file 0400 root) |
| container Docker (root) | `"System Serial Number"` |

Hai giá trị khác nhau ⇒ **hai khoá bản quyền khác nhau**. Khoá cấp lúc chạy trên host
sẽ **không bao giờ** validate trong container, và thông báo lỗi
("Hardware configuration mismatch") không hề gợi ý nguyên nhân.

⇒ **Khoá phải được sinh từ chính container sẽ chạy.** Lệnh có trong `build/.env.example`.

### ⚠️ Và serial đó là chuỗi giữ chỗ

`"System Serial Number"` là giá trị OEM để trống — **giống nhau trên mọi máy cùng đời bo
mạch**. Nghĩa là "ràng buộc phần cứng" trên máy này thực chất chỉ còn ràng buộc theo
hostname + kiến trúc CPU. Khoá cấp cho máy A sẽ chạy được trên máy B cùng model nếu đặt
trùng hostname.

`internal/pkg/security/license.py` **không** tự loại giá trị này (làm vậy sẽ đổi khoá và phá tương
thích với khoá đã cấp), nhưng bật cờ `bios_serial_is_placeholder` và đưa vào thông báo lỗi.

Cần quyết định: có chấp nhận mức bảo vệ này không? Nếu không, hai hướng:

1. Thêm định danh mạnh hơn vào khoá — UUID GPU là ứng viên tốt (`_nvidia_identifiers()` đã
   có sẵn), nhưng phải cấp lại toàn bộ khoá.
2. Chấp nhận và ghi rõ rằng license chỉ là rào cản hành chính, không phải kỹ thuật.

### Cách sắp xếp

`modelsvc` là service khởi tạo, `triton` phụ thuộc nó qua
`condition: service_completed_successfully`. Compose sẽ không khởi động Triton nếu modelsvc
thoát khác 0 — nên Triton **không bao giờ** chạy với model chưa được cấp phép.

Volume `models` là **tmpfs**: bản rõ ONNX không chạm đĩa, và mất sạch khi `compose down`.


---

## DN-005 · Giấy phép ký Ed25519 thay cho vân tay tự sinh

**Trạng thái:** ✅ đã hiện thực và chạy thử end-to-end trong container
**Ngày:** 2026-08-29

### Vì sao vân tay tự sinh KHÔNG phải là giấy phép

```
Sơ đồ ngây thơ: key = SHA512(product_key | hostname | machine | processor | serial)[:20]
```

Hàm sinh khoá **nằm ngay trong phần mềm được giao**, nên bất kỳ ai có phần mềm cũng chạy
được nó trên máy mới và tự cấp cho mình một khoá hợp lệ.

Đây là điểm mấu chốt: **đổi sang định danh phần cứng mạnh hơn không sửa được gì cả.** Kẻ
sao chép chỉ việc sinh vân tay mới trên máy mới rồi tự băm ra khoá. Định danh mạnh chỉ ngăn
được việc *chép nguyên khoá cũ*, không ngăn được việc *tự cấp khoá mới*.

Ngoài ra bốn định danh dùng ở đó đều yếu:

| Định danh | Vấn đề |
|---|---|
| `hostname` | đổi bằng một lệnh |
| `machine` | `x86_64` trên mọi máy |
| `processor` | rỗng trên nhiều bản Linux; giống nhau trên mọi máy cùng đời CPU |
| `product_serial` | máy dev trả `"System Serial Number"` — chuỗi giữ chỗ OEM |

### Cách làm mới

**Mật mã bất đối xứng.** Bên cấp phép giữ khoá riêng Ed25519, không bao giờ giao đi. Phần
mềm nhúng khoá công khai, chỉ dùng để *xác minh*. Giấy phép là chữ ký lên (vân tay + hạn dùng).

```
CO2.<payload base64url>.<chữ ký base64url>
payload = {"v":2, "dev":"<sha256 vân tay>", "exp":<epoch|null>, "iat":..., "note":"GC03"}
```

Khách hàng **không thể** tự cấp giấy phép cho máy mới — không có khoá riêng.

### Vân tay mới, đo thật trên máy (2026-08-29)

| Nguồn | Giá trị | Đặc tính |
|---|---|---|
| `dmi_uuid` | `a53bf5bc-fcbc-17e7-06a7-bcfce71706a6` | UUID thật trong DMI |
| `board_serial` | `241247619300243` | Serial bo mạch thật |
| `gpu` | `GPU-b2bc31c4-3d57-6c15-1dba-a4e8b4fe9cc5` | Khắc trong card, bất biến |

Cả ba đọc được trong container **không cần mount gì thêm** (`/sys` chia sẻ sẵn, tiến trình
trong container là root). Giá trị giữ chỗ của OEM bị loại qua `PLACEHOLDER_VALUES`.

`is_strong` đòi ít nhất một trong `dmi_uuid` / `gpu` — hai thứ duy nhất không sửa được bằng
phần mềm. Thiếu cả hai thì `validate()` **từ chối**, vì ràng buộc thiết bị sẽ vô nghĩa.

### Đã chạy thử end-to-end

Trong container `--gpus device=0 --cpus=2 --memory=2g`:

| Kịch bản | Kết quả |
|---|---|
| Giấy phép đúng máy | ✅ hợp lệ (GC03, hạn 2027-08-29) |
| Chép sang máy khác | ✅ từ chối — "không thuộc thiết bị này" |
| Khách tự ký bằng khoá riêng của mình | ✅ từ chối — "chữ ký không hợp lệ" |

### ⚠️ Khoá xác minh **chỉ** đến từ hằng số trong mã nguồn

Từng có `CRANEOPS_LICENSE_PUBLIC_KEY` cho phép ghi đè `EMBEDDED_PUBLIC_KEY`, với lý do "cho
test và staging". Nó **vô hiệu hoá toàn bộ cơ chế cấp phép**, và đã kiểm chứng bằng cách
khai thác thật:

| | Kết quả |
|---|---|
| Không đặt biến, khoá nhúng xác minh | ✅ từ chối giấy phép tự ký |
| Đặt biến = khoá công khai của chính kẻ tự ký | ❌ **chấp nhận** cho một máy không được cấp phép |

Tự sinh cặp khoá → đặt phần công khai vào biến môi trường → tự ký giấy phép cho bất kỳ máy
nào. Một biến môi trường là đủ.

**Nguyên tắc rút ra:**

> Ranh giới tin cậy phải nằm đúng chỗ quyền hạn thường ngày của người vận hành kết thúc.

Sửa file cấu hình và biến môi trường **là việc của người vận hành** — họ được phép, và họ
làm hằng ngày. Còn sửa một hằng số trong mã nguồn rồi dựng lại image thì không; ai làm được
tới đó đã có quyền thực thi mã tuỳ ý, lúc ấy cấp phép không còn là hàng rào nữa. Nên khoá
xác minh phải nằm ở phía **artifact**, không phải phía **config**.

Hệ quả cho ai đọc sau: **đừng thêm lại bất kỳ đường ghi đè nào qua env**, kể cả cho staging.
Staging cần khoá khác thì dựng image riêng với `EMBEDDED_PUBLIC_KEY` của nó. Test ghi đè
bằng `monkeypatch.setattr` — cần chạy mã trong tiến trình, không phải chỉ đặt một biến — và
`test_no_environment_variable_can_override_the_embedded_key` khoá điều này lại.

### Vận hành: đổi khoá nhúng làm **mọi giấy phép cũ hết hiệu lực**

`EMBEDDED_PUBLIC_KEY` gắn với đúng một khoá riêng. Sinh cặp khoá mới rồi dán vào nghĩa là
mọi giấy phép đã cấp bằng khoá riêng cũ sẽ bị từ chối với *"chữ ký không hợp lệ"*. Trước khi
đổi, phải chắc đã cấp lại giấy phép cho **mọi** máy đang chạy.

Khoá riêng cất ngoài repo: mất là không cấp được giấy phép mới, lộ là ai cũng cấp được.

### Giới hạn còn lại — cần nói thẳng

Mã nguồn Python nằm dạng rõ trong container (mount `:ro`). Ai có quyền trên máy đều đọc
được `internal/pkg/security/license.py` và **sửa** nó để bỏ qua bước kiểm tra. Chữ ký chặn được việc
*tự cấp giấy phép*, không chặn được việc *gỡ bỏ đoạn kiểm tra*.

Đó là giới hạn cố hữu của phần mềm chạy trên máy khách. Giấy phép nên coi là **rào cản
hành chính có bằng chứng**, không phải bảo vệ kỹ thuật tuyệt đối.

Muốn siết thêm thì phải obfuscate hoặc biên dịch. Nhưng trước khi làm, hãy chắc là đang
siết đúng chỗ: bí mật **duy nhất** cần bảo vệ là mật khẩu giải mã model, và nó **không nằm
trong mã nguồn** — chỉ đọc từ `CRANEOPS_MODEL_PASSWORD` lúc chạy, thiếu thì
`cipher._password()` ném `MissingPassword` chứ không có giá trị dự phòng. Nên `strings`
trên một binary PyInstaller **không moi ra được gì**. Biên dịch chỉ làm khó việc *gỡ đoạn
kiểm tra*, không giấu thêm bí mật nào.


---

## DN-006 · Engine TensorRT phải dựng trên chính máy sẽ chạy

**Trạng thái:** đã xác minh bằng thực nghiệm
**Ngày:** 2026-08-29

### Điều đã đo

Chạy `modelsvc` trong container Triton 24.08 trên máy dev (RTX 5090):

```
✅ giấy phép hợp lệ  (dev-workstation, hạn: 2027-08-29)
   trtexec: /usr/src/tensorrt/bin/trtexec
❌ Error Code 1: Internal Error (Unsupported SM: 0xc00)
```

`SM 0xc00` = compute capability 12.0 = Blackwell. TensorRT trong Triton 24.08 chỉ hỗ trợ
tới Hopper (sm_90).

### Hai kết luận

**1. Máy dev cần image Triton mới hơn.** Không phải lỗi code — toàn bộ đường đi trước đó đã
chạy đúng: xác minh giấy phép, tìm `trtexec`, giải mã `.t7`, kiểm tra sức khoẻ ONNX, vá
batch động, gọi `trtexec` với đúng tham số. Chỉ mỗi bước dựng engine là bất khả thi vì
GPU quá mới so với image.

Máy đích là **RTX 3060 (sm_86)** — Triton 24.08 hỗ trợ đầy đủ. Máy dev cần 25.x trở lên;
đã chuyển `build/triton.Dockerfile` sang 25.10 để dùng chung một phiên bản cho cả hai.

**2. Điều quan trọng hơn: engine gắn với kiến trúc GPU.** Kể cả khi máy dev dựng được,
file `.plan` đó **cũng không dùng được** trên RTX 3060. TensorRT sinh mã máy cho đúng
compute capability.

⇒ Bước dựng engine **luôn** phải chạy trên máy đích. Đó chính là lý do `modelsvc` là một
service khởi tạo chạy ngay trước Triton, chứ không phải một bước trong CI hay trong quá
trình build image.

Cũng vì thế mà cơ chế dùng lại engine (`_plan_is_fresh`) đáng giá: lần khởi động đầu mất
vài phút cho 7 model, những lần sau chỉ vài giây.

### Đã chạy được trên máy dev với Triton 25.10

Đây là lý do phải chốt phiên bản image theo GPU đích. Máy này đã
có sẵn `tritonserver:25.10-py3` nên tôi dùng bản đó. Kết quả: **6/6 engine dựng thành công**.

| Model | Thời gian dựng |
|---|---:|
| craneops_truckitems_pico | 94 s |
| craneops_truckhead_pico | 94 s |
| craneops_ccode_det_h | 137 s |
| craneops_ccode_det_v | 117 s |
| craneops_ccode_rec_h | 62 s |
| craneops_ccode_rec_v | 63 s |
| **Tổng** | **~9,5 phút** |

Kết luận "engine gắn với kiến trúc GPU" vẫn đúng: engine dựng ở đây (sm_120) **không** dùng
được trên RTX 3060 (sm_86). Bước dựng vẫn phải chạy trên máy đích.

### ⚠️ Volume phải là loại THƯỜNG, không phải tmpfs

Ban đầu tôi để volume `models` là tmpfs với lý do "bản rõ không chạm đĩa". Sai lầm: tmpfs bị
gỡ khi container cuối cùng dừng, nên **mỗi lần `compose down` là mất sạch engine** và phải
dựng lại 9,5 phút. Cơ chế dùng lại engine (`_plan_is_fresh`) không bao giờ có tác dụng.

Tách hai mối quan tâm:

* **Engine `.plan`** → volume thường, sống qua restart
* **Bản rõ ONNX** → `/dev/shm` (tmpfs), xoá ngay sau khi dựng

Đo được sau khi sửa:

| | Trước | Sau |
|---|---:|---:|
| Thời gian restart | ~570 s (dựng lại) | **9 s** (dùng lại) |
| File ONNX bản rõ sót lại | 0 | **0** |

Lưu ý vận hành: Docker **không** áp định nghĩa volume mới lên volume đã tồn tại. Đổi từ
tmpfs sang thường phải `docker volume rm craneops_models` thì mới ăn.

### Bug tìm được nhờ chạy thật

Log `trtexec` để lộ một lỗi mà unit test không bắt được:

```
[W] Dynamic dimensions required for input: image, but no shapes were provided.
    Automatically overriding shape to: 1x3x416x416
```

Hai model PicoDet có `needs_batch_patch=True` nhưng **thiếu `trt_profile`**. Không có
profile min/opt/max thì `trtexec` tự chốt batch về 1 — chỉ **cảnh báo**, không lỗi — khiến
việc vá batch thành động trở nên vô nghĩa. Đã bổ sung profile cho cả hai.

Đây là loại lỗi mà chỉ chạy thật mới thấy: mọi thứ vẫn "thành công", chỉ là kết quả không
như mong đợi.

---

## DN-007 · Nhánh mã container là **BLS**, không phải `ensemble`

**Trạng thái:** ✅ **ĐÃ CHỐT** · đã dựng, đã chạy, đã đo

Kế hoạch ban đầu (Phần C.3) ghi nhánh ccode là một `ensemble` của Triton:

```
craneops_ccode_h (ensemble)
  ├─ craneops_ccode_det_h      TensorRT
  ├─ craneops_ccode_dbpost_h   Python backend
  └─ craneops_ccode_rec_h      TensorRT
```

Nhưng chuỗi xử lý thật **không diễn đạt được bằng
ensemble**. Bước giữa det và rec không phải một phép cắt, mà là năm bước có rẽ nhánh theo
dữ liệu:

| # | Bước | Vì sao ensemble không làm được |
|---|---|---|
| 1 | Chọn **top-5 hộp theo diện tích** | số hộp vào thay đổi từng khung |
| 2 | Cắt theo hộp | — |
| 3 | Nắn phối cảnh theo **4 đỉnh của từng hộp** (mã ngang) | ma trận biến đổi khác nhau cho từng phần tử |
| 4 | Cân sáng CLAHE | — |
| 5 | **Cổng nét ảnh loại bỏ crop nhoè** | **số crop ra ≠ số hộp vào** |

`ensemble` là một đồ thị **tĩnh**: tensor chảy qua các bước theo lược đồ khai báo sẵn.
Bước (5) làm số phần tử thay đổi theo nội dung ảnh, và bước (1) cũng vậy. Không có cách
nào khai báo "bỏ bớt phần tử" trong lược đồ ensemble.

→ Dùng **BLS** (Business Logic Scripting): một model Python backend duy nhất tự gọi det
và rec qua `pb_utils.InferenceRequest`.

### Không mất `dynamic_batching`

Đây là điều đáng lo nhất khi bỏ ensemble, vì gom batch là tối ưu số 2 của toàn dự án.
Thực tế **được nhiều hơn**:

* Mỗi lần chạy gom **toàn bộ crop còn sống của một ROI thành MỘT lời gọi rec** thay vì
  gọi từng crop một. Riêng chỗ này đã là 1→5 lần.
* Ba tiến trình BLS (`instance_group count: 3`, `KIND_CPU`) chạy song song, nên lời gọi
  rec từ các camera khác nhau chồng lấn trong hàng đợi và được bộ gom batch của Triton
  nhập tiếp.

Test `test_pipeline_doc_duoc_va_gom_crop_thanh_mot_lan_goi` khoá tính chất này lại: nó
kiểm `rec_batch_sizes == [2]` chứ không phải `[1, 1]`.

### Hai chi tiết bắt buộc, đều không có trong tài liệu

1. **`preferred_memory`.** Model TensorRT trả tensor nằm trong **VRAM**. Gọi `.as_numpy()`
   trên đó ném `Tensor is stored in GPU and cannot be converted to NumPy`. Phải khai
   `pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0)` trong
   `InferenceRequest`. Hậu xử lý DB chạy trên CPU nên dù sao cũng phải chép về host.

2. **`config.pbtxt` phải được chép vào model repository.** Xem DN-009.

**Mã liên quan:** `internal/pkg/vision/*.py` (thuần,
test được không cần GPU) · `triton/bls/ccode.py` (chỉ chuyển tiếp) ·
`triton/repo/craneops/craneops_ccode_{h,v}/`

---

## DN-008 · Recognizer OCR chạy **FP32**, không phải FP16

**Trạng thái:** ✅ **ĐÃ ĐO** · rủi ro nêu ở Phần G của kế hoạch là có thật

Kế hoạch xếp "FP16 làm lệch kết quả OCR ở biên" ở mức rủi ro **trung bình**, kèm biện
pháp "nếu FP16 lệch thì giữ FP32 cho rec". Đã đo trên ảnh có nhãn
`assets/samples/QC3/Cam01/DRYU2874604-1731336343-01.jpg`:

| Độ chính xác | Chuỗi đọc | Điểm |
|---|---|---|
| **FP32** (đang dùng) | `DRVU2874604` | 0,9283 |
| FP16 | `DRVU284604` | **0,9685** |


FP16 **mất ký tự `7`** — và trả về độ tin cậy **cao hơn** bản đúng. Đây là kiểu hỏng tệ
nhất có thể: tầng bình chọn phía sau (`StreakVoter`, `manifest_crosscheck`) cân nhắc theo
điểm số, nên một kết quả sai mà tự tin sẽ **thắng** kết quả đúng.

Detector thì không sao — bitmap ORT ↔ TensorRT FP16 lệch tối đa 0.024 và cho ra đúng cùng
một hộp. Nên chỉ tắt FP16 cho hai model `rec`:

```python
ModelSpec(name="craneops_ccode_rec_h", ..., fp16=False)
ModelSpec(name="craneops_ccode_rec_v", ..., fp16=False)
```

Chi phí: engine rec FP32 dựng mất 24 s (FP16 cũng cỡ đó) và tốn thêm VRAM. SVTR nhỏ nên
không đáng kể so với ngân sách ở `HARDWARE_BUDGET.md`.

---

## DN-009 · `config.pbtxt` phải được chép vào model repository

**Trạng thái:** ✅ **ĐÃ SỬA** · lỗi im lặng, đã có test khoá lại

Triton chạy với `--strict-model-config=false`. Thư mục model nào **không có**
`config.pbtxt` thì nó **tự suy** config từ file `.plan`. Ban đầu `modelsvc` chỉ ghi
`.plan` vào `/models/<tên>/1/`, nên toàn bộ `config.pbtxt` sinh từ `tools/export_models.py`
chưa bao giờ tới được Triton.

Triệu chứng: **không có triệu chứng nào**. Mọi model vẫn `READY`, kết quả vẫn đúng. Chỉ
hai thứ âm thầm bị vứt:

| Khai trong config.pbtxt | Triton tự suy | Hệ quả **đã đo** |
|---|---:|---|
| `instance_group.count: 3` | **1** | **230 req/s thay vì 610 — chậm 2,7×**, p50 52,9 ms thay vì 11,9 ms |
| `max_queue_delay_microseconds: 5000` | **0** | **không khác biệt đo được ở tải hiện tại** — xem đính chính bên dưới |

### ✏️ Đính chính: `max_queue_delay` không phải lý do

Bản đầu của ghi chú này khẳng định mất `max_queue_delay` sẽ làm "crop từ các camera khác
nhau gần như không bao giờ được gộp — mất phần lớn lợi ích của `dynamic_batching`".
**Đo A/B thì không đúng.** Chạy đường ống với `max_queue_delay` = 0 và = 5000 cho kết quả
trùng nhau trong sai số: 637,7 vs 628,5 req/s, batch trung bình 1,01 vs 1,00, p50 11,6 vs
11,7 ms. Nếu delay 5 ms thực sự được áp cho mỗi lời gọi thì p50 đã phải tăng thêm 5 ms.

Nguyên nhân thật: ở đường ống, recognizer chỉ chiếm ~1 ms trong 12 ms mỗi request (phần
còn lại là detector và hậu xử lý DB), nên **hàng đợi trước recognizer không bao giờ hình
thành** — không có gì để gom, bất kể cấu hình delay ra sao. Gom batch có hoạt động thật
(6,1× thông lượng khi đập thẳng vào recognizer ở mức đồng thời 32), nhưng đó là **dư địa
cho tương lai**, không phải thứ đang gánh tải.

Lỗi thiếu `config.pbtxt` vẫn đáng sửa — nhưng vì `instance_group`, không phải vì delay.
Số đo đầy đủ: `HARDWARE_BUDGET.md` §6.1.

### ✏️ Và rồi chính lỗi này tái diễn — ngay trong ghi chú về nó

Lúc A/B thử ngưỡng gom batch, tôi tạm đặt `max_queue_delay_microseconds` về 0 bằng `sed`,
rồi khôi phục bằng:

```bash
sed -i 's/max_queue_delay_microseconds: 0$/...5000/'
```

Dòng trong `tools/export_models.py` là `"  max_queue_delay_microseconds: 0",` — kết thúc
bằng `",` chứ không phải `0`. Neo `$` không khớp, **việc khôi phục im lặng thất bại**, và
cấu hình chạy với `delay=0` suốt nhiều ngày sau đó, trái với chú thích ngay phía trên nó.

Hệ quả thật thì nhỏ: A/B đã chứng minh delay không ảnh hưởng ở tải hiện tại, nên mọi số
trong §6.1 vẫn đúng. Nhưng bài học thì không nhỏ:

* **Đừng sửa cấu hình bằng `sed` rồi khôi phục bằng `sed`.** Sinh lại từ nguồn sự thật
  (`make config`) hoặc sửa nguồn rồi sinh lại — đừng vá hai chiều trên file đã sinh.
* **Chốt chặn chỉ chặn được thứ nó nhìn thấy.** `make config-check` so file đã sinh với
  `SPECS`; nó không phát hiện gì vì cả hai đều đã trôi cùng nhau. Thứ bắt được lỗi là đọc
  cấu hình **Triton đang thực sự chạy** qua `/v2/models/<tên>/config`.
* Đây là lần thứ ba trong dự án một phép sửa bằng regex làm hỏng thứ khác (hai lần kia:
  cắt nhầm hàm theo chỉ số dòng, và đổi tên `logits` đụng biến cục bộ).

Kiểm chứng sau khi sửa (`/v2/models/craneops_ccode_rec_h/config`):

```
max_batch_size  : 32
queue_delay(us) : 5000      ← trước đó là 0
instance_group  : [(1, 'KIND_GPU')]
```

Khoá lại bằng `test_config_pbtxt_is_copied_into_repo` và
`test_config_pbtxt_is_synced_even_when_engine_is_reused` — cái sau quan trọng hơn: đổi
ngưỡng batching **không** cần dựng lại engine, nên việc đồng bộ config phải xảy ra
**trước** kiểm tra "engine còn mới".

### Kèm theo: dấu build cho engine

Cùng loại lỗi, phát hiện khi tắt FP16: đổi `fp16` hay `trt_profile` **không** làm file
`.t7` nguồn mới hơn, nên `_plan_is_fresh` (vốn chỉ so mtime) sẽ dùng lại engine FP16 cũ
**vĩnh viễn** và việc tắt FP16 âm thầm vô tác dụng.

`prepare_model` giờ ghi `build.json` cạnh engine, chứa đúng danh sách cờ đã truyền cho
`trtexec`, và `_plan_is_fresh` so cả chữ ký đó. Engine không có dấu build (bản cũ) bị coi
là lỗi thời và dựng lại — an toàn hơn là tin vào thứ không rõ nguồn gốc.

---

## DN-010 · Hậu xử lý nên viết bằng C++ trong `nvdsinfer` không?

**Trạng thái:** ✅ **ĐÃ ĐO — câu trả lời là KHÔNG cho nhánh ccode, CÓ cho nhánh crane/tcode**

Một cách làm phổ biến với DeepStream là đặt hậu xử lý trong plugin `nvdsinfer` viết bằng
C++/CUDA. Câu hỏi hợp lý: nhánh ccode có nên làm vậy không, và có nhanh hơn không?

### Bóc tách 6,8 ms mỗi request (đo 2026-08-30, từ metrics của Triton)

| Tầng | Thời gian | Tỉ lệ |
|---|---:|---:|
| gRPC + truyền ảnh + hàng đợi | 0,71 ms | 10 % |
| **GPU** — `craneops_ccode_det_v` | 0,75 ms | 11 % |
| **GPU** — `craneops_ccode_rec_v` | 0,75 ms | 11 % |
| **Python** — tiền + hậu xử lý | **4,61 ms** | **68 %** |

Chỉ 22 % thời gian là GPU. Nhưng 4,61 ms Python đó **không phải hậu xử lý**:

| Hàm | Thời gian | Bản chất |
|---|---:|---|
| `preprocess.to_tensor` — **chuẩn hoá** | **1,86 ms** | ⭐ TIỀN xử lý |
| `textcrop.prepare_crop` (CLAHE + xoay + đo nét) | 0,41 ms | giữa hai model |
| ├─ `sharpness` (FFT 2 chiều) | 0,21 ms | |
| └─ `equalize_brightness` (CLAHE) | 0,13 ms | |
| `preprocess.batch_to_tensor` (crop → 64×256) | 0,12 ms | tiền xử lý |
| **`dbpost.decode`** | **0,151 ms** | ⭐ HẬU xử lý |
| **`ctc.decode`** | **0,012 ms** | ⭐ HẬU xử lý |

**Hậu xử lý đúng nghĩa chiếm 0,163 ms — 2,4 % của một request.** Viết lại bằng C++,
kể cả nhanh gấp 10 lần, tiết kiệm 0,15 ms trên 6,8 ms. Không đáng.

### API của `nvdsinfer` không diễn đạt được chuỗi này

Ngay cả khi muốn, chữ ký hàm không cho phép:

```cpp
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                         NvDsInferNetworkInfo const&,
                         NvDsInferParseDetectionParams const&,
                         std::vector<NvDsInferObjectDetectionInfo>& objectList)
```

Vào là **chỉ tensor đầu ra**; ra là **chỉ bbox hoặc nhãn**. Ba thứ chuỗi ccode cần mà API
không có:

1. **Ảnh nguồn** — `prepare_crop` phải nắn phối cảnh và chạy CLAHE trên ROI gốc. Parser
   không thấy ảnh.
2. **Gọi model thứ hai** — sau hậu xử lý DB còn phải chạy recognizer. Parser xử lý output
   của đúng một model.
3. **Trả về chuỗi ký tự** — `NvDsInferObjectDetectionInfo` chỉ có bbox + lớp + độ tin cậy.

Bước `bitmap → hộp` (0,151 ms) thì vừa khuôn `NvDsInferParseCustomXXX`. Nhưng tách riêng
mỗi bước đó ra C++ nghĩa là chuỗi bị cắt làm đôi giữa hai tiến trình, đổi lấy 0,15 ms.

Và parser chạy **trong luồng của nvinfer**: đẩy 4,6 ms/request × 50–100 req/s = 23–46 %
một lõi vào luồng streaming — đúng thứ đã cố ý chuyển ra ngoài (DN-009).

### Chỗ đáng tối ưu thật, theo thứ tự

1. ✅ **Gấp phép chuẩn hoá vào chính model** — **ĐÃ LÀM**, xem phần "Kết quả thật" bên dưới.
   Đúng kỹ thuật đã dùng cho `craneops_headcode_cls` (DN-003): chèn `Mul`/`Sub` vào đồ thị
   ONNX để GPU làm. Công cụ: `tools/fold_preprocess.py`.
2. **Bỏ FFT trong cổng nét** (0,21 ms) — thay bằng phương sai Laplacian. ⚠️ Đổi hành vi so
   hiện tại, phải chạy lại golden set trước khi nhận.
3. Viết lại `dbpost` bằng C++ — **0,151 ms**. Đứng cuối danh sách, đúng vị trí của nó.

### Nhưng C++ **đúng chỗ** cho nhánh crane/tcode (Phase 3)

| | ccode | crane / tcode |
|---|---|---|
| Hậu xử lý | DB unclip + CTC | NMS của PicoDet |
| Có gọi model thứ hai giữa chừng? | **có** | không |
| Cần ảnh nguồn? | **có** (warp, CLAHE) | không |
| Trả về gì | chuỗi ký tự | bbox — **đúng khuôn API** |
| Chạy ở đâu | Triton Python backend (BLS) | **`nvdsinfer` C++, trong DeepStream** |

Với PicoDet, C++ còn tránh được cả vòng gRPC sang Triton, vì kết quả đi thẳng vào
`NvDsObjectMeta`. Nguyên tắc: **hậu xử lý đi theo hình dạng của chuỗi xử lý, không theo
ngôn ngữ.** Một model một parser thì C++ đúng; một chuỗi có rẽ nhánh theo dữ liệu thì không.


---

## DN-011 · Gấp chuẩn hoá vào model ccode — kết quả thật, và một dự đoán sai

**Trạng thái:** ✅ **ĐÃ LÀM VÀ ĐÃ KIỂM CHỨNG** · lãi ít hơn dự đoán 4 lần

Thực hiện mục 1 của DN-010: chèn `Mul(A)` + `Sub(B)` vào đầu đồ thị của bốn model ccode
để `(x/255 - mean)/std` chạy trên GPU thay vì trên CPU mỗi request.

```
TRƯỚC:  input x (đã chuẩn hoá) ──► Conv ──► …          Python làm: 1,90 ms
SAU:    input x (pixel thô)    ──► Mul(A) ──► Sub(B) ──► Conv ──► …
```

Hằng số (BGR), sinh bởi `tools/fold_preprocess.py`:

| Model | `A = 1/(255·std)` | `B = mean/std` |
|---|---|---|
| `ccode_det_h` | `[0.00392157]×3` | `[0.481094, 0.457525, 0.407871]` |
| `ccode_det_v` | `[0.01712475, 0.017507, 0.01742919]` | `[2.117904, 2.035714, 1.804444]` |
| `ccode_rec_h` / `_v` | `[0.00784314]×3` | `[1.0, 1.0, 1.0]` |

### Kết quả — đúng, nhưng nhỏ hơn dự đoán

**Kết quả nghiệp vụ không đổi** (`DRVU2874604` @0,9283 vs @0,9284) — điều kiện tiên quyết,
đã đạt. Số hiệu năng: `HARDWARE_BUDGET.md` §6.1, cột "+ gấp chuẩn hoá". Độ trễ giảm 6 %,
thông lượng tăng 14 % — **nhỏ hơn dự đoán bốn lần**, và đó mới là phần đáng ghi lại.

### ✏️ Đính chính: DN-010 dự đoán 24 %, thực tế 6 %

Con số 24 % trong DN-010 đến từ một micro-benchmark đo `astype + transpose` được
**0,261 ms**, từ đó suy ra `to_tensor` sẽ giảm 1,90 → 0,26 ms. Đo lại trong chính đường
chạy: `to_tensor` với `norm=None` tốn **1,187 ms**, không phải 0,261 ms. Tiết kiệm thật
là 0,711 ms, cộng 0,085 ms ở `batch_to_tensor` — tổng 0,796 ms ở tầng hàm, khớp với
0,710 ms đo đầu-cuối.

Hai bài học:

1. **Micro-benchmark chạy trên máy rảnh không dự đoán được máy đang tải.** Lần đo 0,261 ms
   diễn ra khi Triton đang nhàn; lần sau Triton đang phục vụ. Máy này dùng chung.
2. **Ngoại suy từ một hàm ra cả hệ thống là sai lầm.** Phần Python còn lại (3,90 ms) phần
   lớn KHÔNG phải chuẩn hoá mà là chi phí cố định: đóng gói tensor `pb_utils`, phân tích
   JSON tham số, dựng 6 tensor đầu ra. Những thứ đó không đổi khi gấp chuẩn hoá.

### Chi phí còn lại và bước tiếp theo

`to_tensor` còn 1,187 ms, trong đó `cv2.resize` chỉ 0,044 ms. Phần còn lại là
`astype(float32)` rồi `ascontiguousarray(transpose(2,0,1))` — một phép sao chép **có bước
nhảy** trên mảng float32 3,5 MB. Hai cách bỏ nốt, cùng một kỹ thuật (thêm node vào đồ thị):

| Bước | Cách | Ước tính |
|---|---|---|
| Input `UINT8` thay vì `FP32` | thêm `Cast` vào đồ thị | mảng còn 0,9 MB ⇒ sao chép rẻ 4 lần, payload gRPC cũng giảm 4 lần |
| Input `NHWC` thay vì `NCHW` | thêm `Transpose` vào đồ thị | bỏ HẲN phép sao chép có bước nhảy; Python chỉ còn `resize` |

⚠️ Ước tính ở trên **chưa đo** — và bài học vừa rồi nói rõ đừng tin ước tính chưa đo.

### Cách kiểm chứng (bắt buộc, đã chạy)

Sai một hằng số thì model vẫn `READY`, vẫn trả về chuỗi, chỉ là chuỗi rác. Ba lớp chặn:

1. `tools/fold_preprocess.py` chạy **cả hai đồ thị** trên cùng dữ liệu (đồ thị cũ nhận
   ảnh đã chuẩn hoá, đồ thị mới nhận ảnh thô) và từ chối ghi file nếu lệch quá `1e-4`.
   Đo được: lệch tối đa `7,6e-05`.
2. `tests/unit/test_ccode_pipeline.py` khoá việc Python phải gửi **pixel thô `[0,255]`**
   khi `folded_preprocess=True`.

⚠️ Về lớp 3: chạy với ngưỡng điểm thật (0,95) thì **cả ba đường đều trả về rỗng**, và
"rỗng bằng rỗng" là phép so đúng một cách vô nghĩa — nó không phân biệt được port đúng
với port hỏng hoàn toàn. Đã thêm cờ `--score-threshold 0` và một dòng cảnh báo in ra khi
cả ba cùng rỗng, để lần sau không ai đọc nhầm dấu ✅ đó là bằng chứng.


---

## DN-012 · Một quy tắc cho cả bảy model: **nhận pixel BGR thô**

**Trạng thái:** ✅ **ĐÃ LÀM VÀ ĐÃ KIỂM CHỨNG** · độ trễ giảm 51 %, thông lượng tăng 69 %

Trước khi làm, mỗi model một luật tiền xử lý riêng, và luật đó chỉ tồn tại trong đầu người
viết code gọi nó:

| Model | Thang | Thứ tự kênh | mean/std |
|---|---|---|---|
| ccode det_h | `/255` | BGR | riêng, std=1 |
| ccode det_v | `/255` | BGR | kiểu ImageNet |
| ccode rec (2) | `/255` | BGR | 0,5 / 0,5 |
| pico (2) | `/255` | **RGB** | không có |
| headcode_cls | thô | **RGB** | đã gấp sẵn (DN-003) |

Sai một ô trong bảng là model vẫn chạy, vẫn trả kết quả, chỉ là kết quả rác. Đo được mức
độ: đưa RGB vào `headcode_cls` (thay vì BGR) làm độ chính xác tụt **100 % → 87,4 %** —
sai rõ ràng nhưng vẫn "trông như đang hoạt động".

Nay chỉ còn một luật: **mọi model nhận pixel BGR thô `[0,255]`**. Mọi phép đổi thang và
đảo kênh nằm trong đồ thị, chạy trên GPU, TensorRT hợp nhất vào conv đầu tiên.

### Hai mức, vì hai đường tiêu thụ khác nhau

| | ccode (4 model) | crane/tcode (3 model) |
|---|---|---|
| Ai gọi | **Python** (BLS trong Triton) | **DeepStream** `nvinferserver` |
| Gấp `/255`, mean/std, đảo kênh | ✅ | ✅ |
| Đầu vào `UINT8` `NHWC` | ✅ | ❌ **cố ý không** |

**Vì sao không đưa UINT8/NHWC cho ba model kia**, dù người dùng có lý khi muốn đồng nhất:
`nvinfer`/`nvinferserver` tự tiền xử lý trên GPU và sinh ra tensor **float**. Đổi ba model
đó sang UINT8 sẽ trói tay cấu hình DeepStream ở Phase 3 để đổi lấy một khoản tiết kiệm
**không tồn tại** — đường đó không có Python nào để mà tiết kiệm. Phần đồng nhất có ích
(pixel thô, BGR) thì đã áp cho cả bảy.

Với ba model đó, cấu hình DeepStream trở thành: `net-scale-factor=1.0`,
`model-color-format=1` (BGR), không offset — giống hệt nhau cho cả ba.

### Kết quả đo

Số đầy đủ: `HARDWARE_BUDGET.md` §6.1, cột "+ UINT8 NHWC".

Một chi tiết đáng chú ý: **GPU cũng nhanh lên**, không chỉ Python. Tensor đầu vào nhỏ đi 4
lần nên phần chép host→device rẻ hơn hẳn — khoản này nằm trong `compute_input_duration`
của Triton, và nó là lý do `det_v` tăng từ 561 lên 1 364 mẫu/s *dù* đồng thời chuyển từ
FP16 sang FP32.

### Kiểm chứng — và ba lần tôi làm sai chính phép kiểm chứng

Kết quả nghiệp vụ **không đổi**: mã container vẫn đọc ra `DRVU2874604`; `headcode_cls` giữ
**100,0 %** trên 451 ảnh.

Nhưng phép kiểm chứng phải sửa ba lần trước khi nó thật sự kiểm được gì:

1. **Ngưỡng tuyệt đối** — từ chối hai model PicoDet vì lệch 9,2e-04, trong khi đầu ra của
   chúng là toạ độ bbox tới 623 px, tức 1,5e-06 tương đối. Thuần nhiễu float32.
2. **Ngưỡng tương đối** — lại từ chối `truckhead_pico` vì điểm số trên ảnh không có xe có
   giá trị lớn nhất 0,054, nên một sai lệch tuyệt đối vô nghĩa hoá thành 9,4e-04 tương
   đối. → Phải dùng `ATOL + RTOL·|kỳ vọng|` (ngữ nghĩa `np.allclose`), với `1e-3` chọn
   theo **độ phân giải quyết định của nghiệp vụ** (nhỏ hơn mọi ngưỡng tầng sau ≥ 100 lần),
   không phải chọn cho vừa số đo.
3. **Đối chứng đúng vô nghĩa** — thêm phép "đưa sai thứ tự kênh phải cho kết quả khác
   hẳn", nhưng điều kiện lại là `control > relative × 100`. Khi `relative ≈ 0` thì vế phải
   cũng ≈ 0 nên **mọi** giá trị đối chứng đều đạt, kể cả 0,5 (tức đưa sai kênh mà kết quả
   không đổi). `headcode_cls` đã báo ✅ đúng theo kiểu đó. → Thêm sàn tuyệt đối `1.0`.

Đây là lần thứ hai trong dự án một phép so "đúng một cách vô nghĩa" suýt lọt (lần trước:
parity ccode ở ngưỡng 0,95 cho cả ba đường đều rỗng — DN-011). Bài học lặp lại đủ để ghi
thành nguyên tắc: **mọi phép so bằng đều phải kèm một đối chứng chứng minh nó phân biệt
được.**

Cũng cần đầu vào **thật**, không phải nhiễu ngẫu nhiên: nhiễu là đầu vào bệnh lý cho
detector (không thấy vật gì ⇒ mọi điểm về 0 ⇒ sai lệch tương đối phồng lên vô nghĩa).

### Một lỗi mà chỉ chạy thật mới thấy

`triton/bls/ccode.py` ép cứng `np.ascontiguousarray(tensor, np.float32)` khi gọi model.
Toàn bộ 264 test đơn vị vẫn xanh vì chúng dùng model giả. Chỉ khi chạy thật Triton mới
báo: *"inference input 'x' data-type is 'FP32', but model expects 'UINT8'"*. Kiểu dữ liệu
là **hợp đồng của model**, không phải lựa chọn của lớp chuyển tiếp — đã sửa và ghi rõ lý
do tại chỗ.

---

## DN-013 · Detector DB cũng chạy **FP32** — ngưỡng hậu xử lý quyết định điều đó

**Trạng thái:** ✅ **ĐÃ ĐO** · cả bốn model ccode dùng FP32

DN-008 chốt FP32 cho recognizer vì FP16 đọc **sai mã** với độ tin cậy **cao hơn**. Câu hỏi
còn lại: detector có được dùng FP16 không? Nó chỉ sinh một bitmap xác suất, và sai số ở đó
"trông có vẻ" vô hại.

Không vô hại. Con số cần so không phải là sai số tuyệt đối của bitmap, mà là **khoảng cách
từ sai số đó tới ngưỡng quyết định của tầng sau nó**:

| | FP16 | **FP32 (đang dùng)** |
|---|---:|---:|
| bitmap, lệch trung vị | 3,4e-02 | 9,7e-03 |
| bitmap, lệch **lớn nhất** | **0,198** | 0,078 |

Hậu xử lý DB nhị phân hoá bitmap ở `bitmap_threshold = 0.1` rồi lọc hộp ở
`box_threshold = 0.2` (`internal/pkg/vision/dbpost.py`). Sai số lớn nhất của FP16 là
**0,198** — lớn hơn ngưỡng thứ nhất gấp đôi và gần bằng ngưỡng thứ hai. Một pixel ở rìa
vùng chữ đủ để lật từ "có chữ" sang "không", tức mất hẳn một hộp, tức mất hẳn một dòng của
mã container.

### Nguyên tắc rút ra

> Sai số số học chỉ vô hại khi **nhỏ so với ngưỡng quyết định đứng sau nó**, không phải khi
> nhỏ so với thang giá trị của chính nó.

Một bitmap lệch 0,198 trên thang `[0,1]` nghe như 2 %. Đặt cạnh ngưỡng 0,1 thì nó là 198 %.
Khi cân nhắc hạ độ chính xác ở bất kỳ model nào sau này, hãy tìm ngưỡng gần nhất ở phía
sau nó trước — đó mới là thước đo.

### Cái giá, và vì sao chấp nhận được

FP32 cho cả det lẫn rec tốn thêm **512 MiB VRAM** (3 634 → 4 146 MiB) và **không** làm chậm
đáng kể (trần còn tăng: 1 064 → 1 137 req/s, do dứt được các lần chuyển đổi kiểu). Ngân sách
VRAM ở §3 của `HARDWARE_BUDGET.md` còn dư nhiều lần, nên đây là đánh đổi rẻ. Số đầy đủ:
`HARDWARE_BUDGET.md` §6.1 và §6.2.

`fp16=False` được ghi ngay tại chỗ khai báo model trong `tools/export_models.py`, kèm trỏ về
mục này — đổi cờ đó mà không đọc mục này là cách dễ nhất để tái tạo lại lỗi.
