from abc import ABC, abstractmethod


class MixinBase(ABC):

    @abstractmethod
    def f(self, a: int) -> int: ...
        # the main function that we want to mix-in

    @abstractmethod
    def g(self, a: int) -> int: ...
        # a method that we know that is used in f()

class Mixin1(MixinBase):
    def f(self, a: int) -> int: return self.g(a) ** 2

class Mixin2(MixinBase):

    def f(self, a: int) -> int: return self.g(a) + 2


## implicit example
class ImplicitExample:
    def g(self, a: int): return a

class ImplicitExampleWithMixin1(ImplicitExample, Mixin1): ...
class ImplicitExampleWithMixin2(ImplicitExample, Mixin2): ...


## explicit example
class ExplicitExample(MixinBase):
    def g(self, a: int): return a

class ExplicitExampleWithMixin1(ExplicitExample, Mixin1): ...
class ExplicitExampleWithMixin2(ExplicitExample, Mixin2): ...
