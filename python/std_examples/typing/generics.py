import typing
from typing import Generic
from typing import TypeVar

T = TypeVar('T')
class Sample(Generic[T]):
    data: T
    def __init__(self, data: T):
        self.data = data

    def show(self):
        # cause pyright error, but ok with mypy
        print(typing.get_args(self.__orig_class__)[0].__name__) 

s = Sample[int](2)
s.show()

