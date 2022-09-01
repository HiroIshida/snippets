from dataclasses import dataclass
from typing import Optional

@dataclass 
class Info:
    n: int = 20

class NormalClass:
    info: Optional[Info] = None

@dataclass
class Human(NormalClass):
    height: float
    width: float

    @classmethod
    def from_normal_class(cls, info: Info, **kwargs):
        obj = cls(**kwargs) 
        obj.info = info
        return obj

h = Human.from_normal_class(Info(), height=10, width=10)

