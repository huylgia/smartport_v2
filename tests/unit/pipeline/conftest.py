"""``Gst`` giả — đủ để kiểm LOGIC dựng graph trên máy không có DeepStream.

Việc dựng graph thật cần DeepStream và phải khói-test trên máy có GPU. Nhưng phần lớn thứ
dễ sai lại **không** cần GPU để kiểm: camera nào được nối vào muxer, ``drop-frame-interval``
đặt giá trị nào, hàng đợi nào được chặn ra sao, EOS có bị nuốt không. Fake này ghi lại mọi
thao tác để những câu hỏi đó trả lời được bằng ``pytest``.

Fake **không** mô phỏng luồng dữ liệu — nó không nói được pipeline có chạy hay không.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest


class FakePad:
    def __init__(self, name: str, owner: FakeElement) -> None:
        self.name = name
        self.owner = owner
        self.peer: FakePad | None = None
        self.target: FakePad | None = None
        self.probes: list[tuple[Any, Any]] = []

    def link(self, other: FakePad) -> Any:
        self.peer = other
        other.peer = self
        return FakeGst.PadLinkReturn.OK

    def unlink(self, other: FakePad) -> None:
        self.peer = None
        other.peer = None

    def get_peer(self) -> FakePad | None:
        return self.peer

    def set_target(self, pad: FakePad) -> bool:
        self.target = pad
        return True

    def get_target(self) -> FakePad | None:
        return self.target

    def add_probe(self, kind: Any, callback: Any, *args: Any) -> int:
        self.probes.append((kind, callback))
        return len(self.probes)


class FakeElement:
    def __init__(self, factory: str, name: str) -> None:
        self.factory = factory
        self.name = name
        self.props: dict[str, Any] = {}
        self.children: list[FakeElement] = []
        self.parent: FakeElement | None = None
        self.signals: dict[str, list[Any]] = {}
        self.state: Any = None
        self._pads: dict[str, FakePad] = {}
        self.linked_to: FakeElement | None = None

    # --- thuộc tính
    def set_property(self, key: str, value: Any) -> None:
        self.props[key] = value

    def get_property(self, key: str) -> Any:
        return self.props.get(key)

    def find_property(self, key: str) -> bool:
        return key in {"use-robust-muxing", "muxer-factory", "muxer-properties"}

    # --- danh tính
    def get_name(self) -> str:
        return self.name

    def get_factory(self) -> Any:
        # Gst thật: get_factory().get_name() trả tên FACTORY ("h265parse"), không phải tên
        # element ("parser"). Trả `self` là sai và làm test tin nhầm.
        return type("F", (), {"get_name": staticmethod(lambda: self.factory)})()

    def get_parent(self) -> FakeElement | None:
        return self.parent

    # --- pad
    def get_static_pad(self, name: str) -> FakePad:
        return self._pads.setdefault(name, FakePad(name, self))

    def get_request_pad(self, template: str) -> FakePad:
        name = template.replace("%u", str(len(self._pads)))
        return self._pads.setdefault(name, FakePad(name, self))

    def request_pad_simple(self, name: str) -> FakePad:
        return self._pads.setdefault(name, FakePad(name, self))

    def add_pad(self, pad: FakePad) -> bool:
        self._pads[pad.name] = pad
        return True

    # --- cây
    def add(self, child: FakeElement) -> bool:
        self.children.append(child)
        child.parent = self
        return True

    def remove(self, child: FakeElement) -> bool:
        self.children.remove(child)
        return True

    def get_by_name(self, name: str) -> FakeElement | None:
        for child in self.children:
            if child.name == name:
                return child
            found = child.get_by_name(name)
            if found is not None:
                return found
        return None

    def iterate_elements(self) -> Any:
        return FakeIterator(list(self.children))

    # --- vòng đời
    def link(self, other: FakeElement) -> bool:
        self.linked_to = other
        return True

    def connect(self, signal: str, callback: Any, *args: Any) -> int:
        self.signals.setdefault(signal, []).append((callback, args))
        return 1

    def emit(self, signal: str, *emit_args: Any) -> None:
        """Kích hoạt tay — dùng để mô phỏng ``nvurisrcbin`` dựng xong phần bên trong."""
        for callback, bound in self.signals.get(signal, []):
            callback(self, *emit_args, *bound)

    def set_state(self, state: Any) -> Any:
        self.state = state
        return FakeGst.StateChangeReturn.SUCCESS

    def get_state(self, _timeout: Any) -> Any:
        return FakeGst.StateChangeReturn.SUCCESS

    def sync_state_with_parent(self) -> bool:
        return True


class FakeIterator:
    def __init__(self, items: list[FakeElement]) -> None:
        self._items = list(items)

    def next(self) -> tuple[Any, FakeElement | None]:
        if not self._items:
            return FakeGst.IteratorResult.DONE, None
        return FakeGst.IteratorResult.OK, self._items.pop(0)


class FakeBin(FakeElement):
    """``Gst.Bin`` — cùng bề mặt với element, thêm hàm dựng ``new``."""

    @staticmethod
    def new(name: str) -> FakeBin:
        return FakeBin("bin", name)


class FakeGst:
    """Bề mặt ``Gst`` tối thiểu mà code dựng pipeline dùng tới."""

    SECOND = 1_000_000_000
    Bin = FakeBin
    PadLinkReturn = SimpleNamespace(OK="OK", REFUSED="REFUSED")
    StateChangeReturn = SimpleNamespace(SUCCESS="SUCCESS", FAILURE="FAILURE")
    IteratorResult = SimpleNamespace(OK="OK", DONE="DONE")
    PadDirection = SimpleNamespace(SRC="SRC", SINK="SINK")
    PadProbeType = SimpleNamespace(BUFFER="BUFFER", EVENT_DOWNSTREAM="EVENT_DOWNSTREAM")
    PadProbeReturn = SimpleNamespace(OK="OK", DROP="DROP")
    EventType = SimpleNamespace(EOS="EOS")
    State = SimpleNamespace(NULL="NULL", PLAYING="PLAYING", PAUSED="PAUSED")
    BufferFlags = SimpleNamespace(DELTA_UNIT=1)
    CLOCK_TIME_NONE = -1

    created: ClassVar[list[FakeElement]] = []

    class ElementFactory:
        @staticmethod
        def make(factory: str, name: str) -> FakeElement | None:
            if factory == "khong-ton-tai":
                return None
            element = FakeElement(factory, name)
            FakeGst.created.append(element)
            return element

    class GhostPad:
        @staticmethod
        def new_no_target(name: str, _direction: Any) -> FakePad:
            return FakePad(name, None)  # type: ignore[arg-type]

    class Structure:
        @staticmethod
        def new_from_string(text: str) -> str:
            return text


@pytest.fixture
def gst() -> type[FakeGst]:
    FakeGst.created = []
    return FakeGst
