from typing import Tuple
from typing import List
from typing import Generic
from typing import TypeVar

# tuple is covariant type
# https://blog.daftcode.pl/covariance-contravariance-and-invariance-the-ultimate-python-guide-8fabc0c24278

class Parent: ...
class Son(Parent): ...
class Daughter(Parent): ...

def f(a: Tuple[Parent, ...]): return
f((Son(), Daughter()))
f((Parent(), Daughter()))

def g(a: List[Parent]): return
g([Son(), Daughter()])
g([Parent(), Daughter()])

T = TypeVar("T", bound=Tuple[Parent, ...])
class A(Generic[T]): pass
class B1(A[Tuple[Parent]]): pass
class B2(A[Tuple[Son, Daughter, Parent]]): pass
