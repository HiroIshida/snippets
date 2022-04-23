from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type

class Fruit: pass
class Apple(Fruit): pass
class Orange(Fruit): pass
FruitT = TypeVar('FruitT', bound=Fruit)

class FruitJuice: pass
class AppleJuice(FruitJuice): pass
class OrangeJuice(FruitJuice): pass
FruitJuiceT = TypeVar('FruitJuiceT', bound=FruitJuice)

JuiceFactoryT = TypeVar('JuiceFactoryT', bound='JuiceFactory')
class JuiceFactory(ABC, Generic[FruitT, FruitJuiceT]):
    type_in: Type[FruitT]
    tyoe_out: Type[FruitJuiceT]

    @abstractmethod
    def __cal__(self, inp: FruitT) -> FruitJuiceT: pass


class AppleJuiceFactorty(JuiceFactory[Apple, AppleJuice]):
    type_in = Apple
    type_out = OrangeJuice

    def __call__(self, inp: Orange) -> OrangeJuice:
        return OrangeJuice()
