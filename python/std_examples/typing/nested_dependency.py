from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Protocol, Type


ConfigT = TypeVar("ConfigT", bound="ConfigProtocol")
SolverT = TypeVar("SolverT", bound="SolverBase")

class ConfigProtocol(Protocol):
    n_iter: int


class SolverBase(ABC, Generic[ConfigT]):
    
    @classmethod
    def setup(cls: Type[SolverT], conf: ConfigT) -> SolverT:
        return cls()

    @abstractmethod
    def solve(self) -> int:
        ...


def func(solver_t: Type[SolverBase[ConfigT]], config: ConfigT) -> int:
    solver = solver_t.setup(config)
    return solver.solve()


if __name__ == "__main__":

    @dataclass
    class MyConfig:
        a: float

    @dataclass
    class AnotherConfig:
        n_iter: int
        a: float


    class MySolver(SolverBase[MyConfig]):

        def solve(self) -> int:
            return 0

    func(MySolver, MyConfig(10, 0))
    func(MySolver, AnotherConfig(10, 0))
