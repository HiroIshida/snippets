from typing import overload, Optional
from typing_extensions import reveal_type

# https://peps.python.org/pep-0484/#arbitrary-argument-lists-and-default-argument-values

class Example: 

    @overload
    @classmethod
    def temp(cls, x: int, y: int=...) -> Optional[int]:
        # you must add ellipsis to show that there exists a default augment
        ...

    @overload
    @classmethod
    def temp(cls, x: None, y: int=...) -> None:
        ...

    @classmethod
    def temp(cls, x: Optional[int] = None, y: int = 12):
        if x is None:
            return None
        return x + y


Example.temp(1, 1)
Example.temp(None, 1)
Example.temp(None)
