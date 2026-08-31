# Hợp đồng message

Nguồn sự thật là `common/message.py` (pydantic v2). Tài liệu này giải thích *vì sao*;
định nghĩa trường lấy từ code.

**Quy tắc bắt buộc**

1. Mọi payload có `schema_version`. Đổi trường bắt buộc ⇒ tăng major.
2. **Validate ở cả producer và consumer.** Một JSON Schema chỉ dùng làm tài liệu
   nhưng không ai enforce lúc runtime — gõ sai key rơi im lặng vào `.get(key, default)`.
   v2 fail-fast.
3. Thời gian là **trục suy từ frame**: `frame_ts = start_ts + frame_id / fps`. Không dùng
   `time.time()` tại điểm xử lý — wall-clock bị nhiễu theo độ trễ queue/inference.
4. Key quyết định partition ⇒ quyết định thứ tự. Chọn key sao cho mọi message cần xử lý
   tuần tự nằm chung một partition.

---

## Bảng topic

| Topic | Key | Producer | Consumer | Retention |
|---|---|---|---|---|
| `craneops.perception.ccode` | `camera_code` | ds_app | ruled@ccode | 1 h |
| `craneops.perception.tcode` | `camera_code` | ds_app | ruled@tcode | 1 h |
| `craneops.perception.crane` | `camera_code` | ds_app | ruled@crane | 1 h |
| `craneops.perception.bottom` | `camera_code` | ds_app | ruled@bottom | 1 h |
| `craneops.signals` | `lane` | ruled | orchestratord | 6 h |
| `craneops.manifest` | `crane_id` | syncd | ruled@ccode, ruled@crane | **compacted** |
| `craneops.evidence.fast` | `event_id` | orchestratord | evidenced | 24 h |
| `craneops.evidence.slow` | `event_id` | orchestratord | evidenced | 24 h |
| `craneops.events` | `event_id` | evidenced | syncd | 7 ngày |
| `craneops.control` | `crane_id` | syncd / craneopsctl | tất cả | 1 h |

**Vì sao key như vậy**

- `perception.*` key theo `camera_code`: một camera là một dòng thời gian, phải giữ thứ tự frame.

`camera_code` có dạng `<mã cẩu>_<ip>_<cổng>` (`GC03_113_160_225_15_1508`) và **suy từ URL
trong config**, không khai tay. Cùng chuỗi đó là tên thư mục ghi hình, nên `segment_hint`
và `camera_code` không thể trôi khỏi nhau.

`crane_id` vẫn là trường riêng dù mã camera đã chứa nó: mã cẩu có thể chứa gạch dưới, và
khi đó tách ngược từ `camera_code` là đoán.
- `signals` key theo `lane`: orchestrator dựng state machine theo lane, mọi signal của
  cùng một lane phải tuần tự. Signal từ nhiều camera khác nhau vẫn về cùng partition.
- `manifest` **compacted**: chỉ cần bản mới nhất. Thay `ContainerCodeCamera.DATABASE`
  — thay cho biến toàn cục dùng chung giữa các camera, vốn chỉ có tác dụng trong một process.
- `evidence` tách **fast/slow**: ảnh chụp gần như tức thì; clip phải chờ `delay: 20–40 s`
  sau thao tác cẩu. Trộn chung sẽ để job nhanh kẹt sau job chậm. Mượn từ
  hai lane nhanh/chậm.

---

---

## Ghi chú từng topic

### `craneops.perception.*`

`segment_hint` trỏ tới file segment chứa khung này — đó là cách `evidenced` biết cắt clip
ở đâu. Producer tra được nó chính xác, nên nó nói ra; tìm lại bằng cách quét thư mục theo
dấu thời gian là suy đoán, và suy đoán sai ở ranh giới giữa hai segment.

⚠️ Hai điều kiện để giá trị này đúng, cả hai đều dễ mất mà không có triệu chứng:

1. **Mốc mở đoạn và dấu thời gian khung phải trên CÙNG một trục.** Nhánh ghi và nhánh
   model đóng dấu bằng hai đồng hồ khác nhau thì chúng trôi khỏi nhau và cửa sổ cắt clip
   lệch dần. Cả hai đi qua `ds_app/src/pipeline/timesync.py` — neo `PTS → unix` một lần
   cho mỗi camera. Đo trên camera thật: khoảng cách giữa các mốc đoạn là **10,00 s chính
   xác**, thứ chỉ có được trên trục suy từ media.
2. **Phải tra theo cửa sổ, không phải lấy đoạn mới nhất.** Nhánh ghi có thể chậm hơn nhánh
   model cả một đoạn, nên đoạn mới nhất thường không phải đoạn chứa khung đang xét.

`fps` là fps của **nguồn** (30 với smartport), không phải fps sau decimate. `frame_id` là
chỉ số khung **gốc**, đã khôi phục qua `restore_frame_id`. Dùng `frame_ts` có sẵn, đừng tự
tính lại.

`attrs` trong mỗi `Detection` gắn phẳng thuộc tính từ SGIE (ví dụ điểm phân loại số đầu
kéo) — cùng cơ chế mà DeepStream dùng để gắn thuộc tính lên bbox.

Chỉ `role: ccode` được mang mảng `ocr` — model từ chối nếu vai trò khác.

### `craneops.signals`

`kind` là khoá mà composition spec tham chiếu trong `configs/operations/*.yaml`. Nó là
enum (`SignalKind`) chứ không phải chuỗi tự do: gõ sai sẽ khiến một trường của sự kiện âm
thầm không bao giờ được điền, và không có gì báo lỗi.

### `craneops.manifest`

`containers` rỗng **không** phải lỗi — nghĩa là chưa có tàu tại bến, một trạng thái vận
hành bình thường xảy ra hàng ngày. Hệ chuyển sang chế độ không-đối-chiếu (combinator
`fuzzy_dedup`) và chờ; không dừng, không báo động, và tuyệt đối không khởi động lại máy.

### `craneops.events`

Đây là payload mà `syncd` **dàn phẳng** rồi POST lên dashboard e-port
(`/admin/berth/support/detection`). Nội bộ v2 dùng `slots: list[ContainerSlot]`; việc dàn
phẳng sang tên trường mà dashboard e-port yêu cầu (`containerNo`/`containerNo2`, `ixCd`/`ixCd_2`,
`shortVideo`/`shortVideo2`…) chỉ xảy ra ở `gateway/contract/dashboard.py` — biên tương
thích ngược, không rò rỉ vào trong. Tối đa 2 slot (twin-lift 20 ft).

### `craneops.control`

Thay `BaseCamera.PAUSE_HANDLE_FRAME_SIGNAL` — một `threading.Event` cấp class
vốn chỉ có tác dụng trong một process.

---

## Ví dụ payload

<!-- BEGIN GENERATED EXAMPLES -->

*Sinh tự động bởi `tools/gen_message_examples.py` — đừng sửa tay.*

### `craneops.perception.*`

```json
{
  "schema_version": "1.0",
  "crane_id": "GC03",
  "camera_code": "GC03_113_160_225_15_1508",
  "role": "ccode",
  "frame_id": 300,
  "start_ts": 1756312827.4,
  "fps": 30.0,
  "frame_ts": 1756312837.4,
  "segment_hint": "/var/lib/craneops/rec/GC03_113_160_225_15_1508/1756312830.mp4",
  "detections": [
    {
      "bbox": {
        "x1": 505.0,
        "y1": 81.0,
        "x2": 1115.0,
        "y2": 662.0
      },
      "class_name": "container",
      "confidence": 0.93,
      "attrs": {}
    }
  ],
  "ocr": [
    {
      "roi_index": 0,
      "shape": "vertical",
      "lane": "1",
      "cont_dim": "40feet",
      "bbox": {
        "x1": 612.0,
        "y1": 190.0,
        "x2": 704.0,
        "y2": 540.0
      },
      "text": "MSKU",
      "confidence": 0.97
    }
  ]
}
```

### `craneops.signals`

```json
{
  "schema_version": "1.0",
  "rule_code": "CCODE01",
  "crane_id": "GC03",
  "camera_code": "GC03_113_160_225_15_1508",
  "lane": "1",
  "direction": "RIGHT",
  "kind": "container_no",
  "frame_ts": 1756312837.4,
  "confidence": 0.96,
  "payload": {
    "container_no": "MSKU1234567",
    "iso": "45G1",
    "streak": 4
  }
}
```

### `craneops.manifest`

```json
{
  "schema_version": "1.0",
  "crane_id": "GC03",
  "berth_no": "TS03",
  "synced_at": 1756312837.4,
  "vsl_cd": "VSL01",
  "call_seq": "001",
  "call_year": "2026",
  "containers": [
    {
      "container_no": "MSKU1234567",
      "ix_cd": "I",
      "cont_dim": "40feet",
      "sztp": "45G1"
    }
  ]
}
```

### `craneops.evidence.fast / craneops.evidence.slow`

```json
{
  "schema_version": "1.0",
  "event_id": "GC03-1756312837-1",
  "crane_id": "GC03",
  "lane": "1",
  "anchor_ts": 1756312837.4,
  "delay": 20.0,
  "jobs": [
    {
      "kind": "clip",
      "camera_code": "GC03_113_160_225_15_1508",
      "window": [
        -20.0,
        15.0
      ],
      "grid": null,
      "count": 1
    },
    {
      "kind": "mosaic",
      "camera_code": "GC03_113_160_225_15_1516",
      "window": [
        -35.0,
        10.0
      ],
      "grid": [
        2,
        2
      ],
      "count": 3
    }
  ]
}
```

### `craneops.events`

```json
{
  "schema_version": "1.0",
  "event_id": "GC03-1756312837-1",
  "crane_id": "GC03",
  "lane": "1",
  "direction": "RIGHT",
  "anchor_ts": 1756312837.4,
  "berth_no": "TS03",
  "vsl_cd": "VSL01",
  "call_seq": "001",
  "call_year": "2026",
  "truck_no": "45",
  "slots": [
    {
      "container_no": "MSKU1234567",
      "ix_cd": "I",
      "sztp": "45G1",
      "cont_position": "40feet",
      "container_image": "https://eport.../snapshots/...jpg",
      "short_video": "https://eport.../videoclips/...mp4",
      "confidence": 0.96
    }
  ]
}
```

### `craneops.control`

```json
{
  "schema_version": "1.0",
  "crane_id": "GC03",
  "action": "reload_rule",
  "rule_code": "CCODE01",
  "issued_at": 1756312837.4
}
```

<!-- END GENERATED EXAMPLES -->
