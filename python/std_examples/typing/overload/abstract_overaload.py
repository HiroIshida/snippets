from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic, List, Tuple, overload


class Parent(ABC):

    @classmethod
    @abstractmethod
    @overload
    def method1(cls, a: None) -> Optional[int]:
        pass

    @classmethod
    @abstractmethod
    @overload
    def method1(cls, a: int) -> int:
        pass

    @classmethod
    @abstractmethod
    def method1(cls, a: Optional[int] = None) -> Optional[int]:
        pass


class Child(Parent):

    @classmethod
    @overload
    def method1(cls, a: None) -> Optional[int]:
        pass

    @classmethod
    @overload
    def method1(cls, a: int) -> int:
        pass

    @classmethod
    def method1(cls, a: Optional[int] = None) -> Optional[int]:
        return a


a: int = Child.method1(1)
b: Optional[int] = Child.method1(None)
assert b is not None
