# see discussions
# https://github.com/python/mypy/issues/1212
 
from typing import TypeVar
from typing import Type
T = TypeVar('T', bound='Parent')

class Parent:
    def __init__(self, data : int):
        self.data = data
    @classmethod 
    def from_float(cls: Type[T], data : float) -> T:
        return cls(int(data))

class Derived(Parent): pass
def somefunc(obj : Derived): return

somefunc(Derived.from_float(1.0))
