from abc import ABC, abstractclassmethod

# holy trait

class DogStyle(ABC):
    @abstractclassmethod
    def fooable(cls) -> bool: ...

class IsDog(DogStyle):
    @classmethod
    def fooable(cls): return True

class IsNotDog(DogStyle):
    @classmethod
    def fooable(cls): return False

class A(IsDog): ...
class B(A): ...
class C(B): ...
class D(IsNotDog, B): ...
class E(IsNotDog, A): ...

def g1(obj: IsDog): print("{} is dog".format(obj.__class__.__name__))
def g2(obj: IsNotDog): print("{} is not dog".format(obj.__class__.__name__))
def f(obj: DogStyle):
    if obj.fooable(): g1(obj) # type: ignore
    else: g2(obj) # type: ignore

for cls in [A, B, C, D, E]: 
    f(cls())

# If we don't have DogStyle I have to write like
"""
def f(obj):
    if isinstance(obj, (D, E)): g1(obj)
    else g2(obj)
"""
# the problem of this code is that if user want to add another typt F
# which is IsNotDog, we have to modify `f` inside library
