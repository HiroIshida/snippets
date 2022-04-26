# 2022/04/26 Found that actually no problem with mypy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Type

class Fruit: pass
class Apple(Fruit): pass
class Orange(Fruit): pass
FruitT = TypeVar('FruitT', bound=Fruit)

JuiceFactoryT = TypeVar('JuiceFactoryT', bound='JuiceFactory')

class JuiceFactory(ABC, Generic[FruitT]):

    @abstractmethod
    def __call__(self, inp: FruitT) -> int: pass


class AppleJuiceFactorty(JuiceFactory[Apple]):
    type_in = Apple

    def __call__(self, inp: Orange) -> int:
        return 1
