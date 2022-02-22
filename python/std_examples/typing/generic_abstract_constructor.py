from abc import abstractmethod, abstractclassmethod
import copy
from typing import TypeVar, Type

ElementT = TypeVar('ElementT', bound='ElementBase')

class ElementBase:
    data: int
    def __init__(self, data): self.data

    @abstractmethod
    def get_plus_one(self: ElementT) -> ElementT:
        pass

    @abstractclassmethod
    def from_data(cls: Type[ElementT], data: int) -> ElementT:
        pass

class Concrete(ElementBase):

    def get_plus_one(self) -> Concrete:
        out = copy.deepcopy(self)
        out.data = self.data + 1
        return out

    @classmethod
    def from_data(cls, data: int) -> 'Concrete':
        return cls(data)
