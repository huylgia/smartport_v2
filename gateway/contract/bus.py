"""Đẩy message lên Kafka — **cổng I/O duy nhất** ra bus.

Ở tầng ``gateway/`` chứ không nằm trong ``ds_app/``: mọi service đều publish, và một bản
producer cho mỗi service là bốn chỗ để cấu hình trôi khỏi nhau.

Ba quyết định đáng nói, cả ba đều là chỗ v1 đã hỏng:

**Gửi là bất đồng bộ, nhưng "đã gửi" nghĩa là broker đã ack.** v1 bắn
``threading.Thread`` + ``requests.post(timeout=2)`` và không bao giờ biết kết quả — event
mất im lặng. Ở đây ``send()`` trả ngay (librdkafka tự đệm) còn callback đếm ack/lỗi, nên
:meth:`BusProducer.stats` phân biệt được "đã xếp" với "broker đã nhận".

**Hàng đợi đầy thì BỎ, không chặn.** ``on_result`` chạy trên thread suy luận; chặn nó là
dồn ngược lên hàng đợi khung rồi tới probe. Phần bỏ được đếm.

**Validate ở producer.** Schema chỉ là tài liệu nếu không ai kiểm — pydantic dựng message
nên một trường sai là lỗi tại chỗ, không phải rác nằm trên topic tới lúc consumer nổ.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from common.message import Topic

if TYPE_CHECKING:
    from common.message import _Msg

__all__ = ["BusProducer", "BusStats"]


@dataclass
class BusStats:
    """Đếm để phần mất mát không im lặng.

    ``queued`` là đã đưa cho client; ``acked`` là **broker đã xác nhận**. Hai số này lệch
    nhau lâu dài nghĩa là broker không theo kịp hoặc mạng hỏng — và đó là thứ v1 không có
    cách nào nhìn thấy.
    """

    queued: int = 0
    acked: int = 0
    failed: int = 0
    dropped: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str, by: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + by)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "queued": self.queued,
                "acked": self.acked,
                "failed": self.failed,
                "dropped": self.dropped,
                "in_flight": self.queued - self.acked - self.failed,
            }

    @property
    def clean(self) -> bool:
        with self._lock:
            return self.failed == 0 and self.dropped == 0


class BusProducer:
    """Producer Kafka an toàn luồng, dùng chung cho mọi service.

    Args:
        servers: ``host:port``, phân tách bằng dấu phẩy.
        client_id: nhận dạng trong log của broker — đặt tên service.
        producer_factory: chỗ tiêm bản giả trong test. Không có Kafka thì không import.
    """

    def __init__(
        self,
        servers: str,
        *,
        client_id: str = "craneops",
        producer_factory: Any | None = None,
    ) -> None:
        self._servers = servers
        self._client_id = client_id
        self._factory = producer_factory or self._default_producer
        self._producer: Any = None
        self.stats = BusStats()

    def _default_producer(self) -> Any:
        from kafka import KafkaProducer

        return KafkaProducer(
            bootstrap_servers=self._servers.split(","),
            client_id=self._client_id,
            # Gom message trong 30 ms rồi gửi một lô. Ở 7,4 message/s thì nó gần như không
            # gom được gì, nhưng nó cũng gần như không tốn gì — và khi nhánh ccode vào
            # (thêm ~25/s) thì nó bắt đầu có tác dụng mà không phải sửa gì.
            linger_ms=30,
            compression_type="lz4",
            # `acks=1`: leader đã ghi là đủ. `all` chờ mọi replica, mà cụm một node thì
            # không có replica nào để chờ — nó chỉ thêm độ trễ đổi lấy một bảo đảm không
            # tồn tại.
            acks=1,
            # Chặn bộ đệm. Đầy thì `send()` ném và ta ĐẾM rồi bỏ. 32 MiB ≈ vài phút
            # message perception ở tải thật.
            buffer_memory=33_554_432,
            # ⚠️ 1 giây, KHÔNG phải 0. Đây là một đánh đổi, không phải con số tuỳ tiện.
            #
            # `max_block_ms` chi phối HAI việc trong `send()`: chờ metadata của topic chưa
            # biết, và chờ chỗ trống trong bộ đệm. Đặt 0 thì `send()` không bao giờ chặn —
            # nhưng nó cũng làm hỏng `partitions_for()` lúc khởi động (hết giờ tức thì),
            # và vài message ĐẦU TIÊN mất trong lúc client hỏi cluster. Đo được: 2 message
            # mất mỗi lần chạy.
            #
            # Với topic đã nạp sẵn metadata (xem `start`) và bộ đệm còn chỗ, `send()` trả
            # về ngay — 1 giây kia chỉ đụng tới khi broker thật sự hỏng. Lúc đó chặn 1 giây
            # trên thread suy luận là chấp nhận được: hàng đợi khung có `leaky` nên phần
            # dồn lại rơi ở đó, không dội ngược tới probe.
            max_block_ms=1000,
            retries=3,
        )

    def start(self, topics: Iterable[Topic] = ()) -> None:
        """Dựng producer, và **nạp sẵn metadata** cho các topic sắp dùng.

        ⚠️ Bước nạp sẵn không phải tối ưu hoá — nó là điều kiện đúng đắn. ``max_block_ms=0``
        làm ``send()`` ném ngay thay vì chờ, nên vài lời gọi ĐẦU TIÊN thất bại trong lúc
        client còn đang hỏi metadata cluster. Đo được: **2 message mất** ở mỗi lần chạy,
        luôn là hai cái đầu, và không có gì báo ngoài bộ đếm.

        Chặn ở đây thì được — đây là lúc khởi động. Chặn trong ``publish()`` thì không:
        nơi gọi là thread suy luận.
        """
        self._producer = self._factory()
        for topic in topics:
            # Gọi này chặn tới khi có metadata, và với Redpanda bật auto-create thì nó
            # tạo luôn topic.
            self._producer.partitions_for(str(topic))

    def publish(self, message: _Msg, key: str | None = None, topic: Topic | None = None) -> bool:
        """Đẩy một message pydantic. Trả ``False`` nếu đã bỏ.

        **Không bao giờ chặn** — ``max_block_ms=0`` biến bộ đệm đầy thành lỗi tại chỗ.

        Args:
            message: message đã dựng bằng pydantic, nên đã hợp lệ.
            key: khoá phân vùng. Cùng khoá ⇒ cùng phân vùng ⇒ **giữ thứ tự**; với
                perception thì khoá là ``camera_code``, để message của một camera không
                bao giờ tới consumer sai thứ tự.
            topic: bắt buộc với :class:`~common.message.PerceptionMessage`, vì topic của
                nó phụ thuộc **role** chứ không phụ thuộc lớp — dùng
                :func:`~common.message.perception_topic`. Các message khác lấy từ
                ``message.topic``.
        """
        if self._producer is None:
            raise RuntimeError("BusProducer chưa start()")
        payload = message.model_dump_json().encode("utf-8")
        try:
            future = self._producer.send(
                str(topic or message.topic),
                value=payload,
                key=key.encode("utf-8") if key else None,
            )
        except Exception:
            # Bộ đệm đầy, hoặc broker không có. Không được ném lên: nơi gọi là thread suy
            # luận, và một message mất không đáng để mất luôn nhánh xử lý.
            self.stats.bump("dropped")
            return False
        self.stats.bump("queued")
        future.add_callback(lambda _meta: self.stats.bump("acked"))
        future.add_errback(lambda _err: self.stats.bump("failed"))
        return True

    def flush(self, timeout: float = 10.0) -> None:
        """Chờ mọi message đã xếp được ack. Gọi trước khi thoát.

        Không có bước này thì phần còn trong bộ đệm mất lúc process kết thúc, và bảng
        thống kê sẽ báo "đã gửi" cho những message chưa từng rời máy.
        """
        if self._producer is not None:
            self._producer.flush(timeout=timeout)

    def close(self, timeout: float = 10.0) -> None:
        if self._producer is not None:
            self._producer.close(timeout=timeout)
            self._producer = None
