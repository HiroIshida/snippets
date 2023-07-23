from dataclasses import dataclass

@dataclass
class Example:
    a: int
    b: float
    c: str

    def __setstate__(self, state):
        if "c" not in state:
            state["c"] = None
        self.__dict__.update(state)
