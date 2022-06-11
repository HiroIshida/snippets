# under construction
from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, Callable, Type, Generic, TypeVar

_T = TypeVar("_T")

@dataclass
class Serializer:
    serializers: Dict[Type, Callable[[Any], Dict]] = {}
    deserializers: Dict[Type, Callable[[Dict], Any]] = {}

    @classmethod
    def default(cls) -> "Serializer":
        primitive_types = (int, float, str)
        serializers = {}
        deserializers = {}
        for t in primitive_types:
            serializers[t] = lambda x: x
            deserializers[t] = lambda x: x
        return cls(serializers, deserializers)

    def register_serializer(self, t: _T, f: Callable[[_T], Dict]) -> None:
        self.serializers[t] = f

    def register_deserializer(self, t: _T, f: Callable[[Dict], _T]) -> None:
        self.deserializers[t] = f

    def serialize(self, data):
        if is_dataclass(data):
            pass
