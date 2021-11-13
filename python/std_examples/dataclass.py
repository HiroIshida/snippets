from dataclasses import dataclass

@dataclass
class Info:
    n_state: int

@dataclass
class Config:
    n_state: int
    n_hidden: int = 200
    n_layer: int = 2

    @classmethod
    def from_info(cls, info, **kwargs):
        return cls(info.n_state, **kwargs)

info = Info(100)
c = Config.from_info(info, n_layer=100)
print(c)

