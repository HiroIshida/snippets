from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class A: pass

class B(A, ABC):
    @abstractmethod
    def fuga(): pass

class C(B): pass

class D(B):
    def fuga(): return

try:
    C()
except TypeError as e:
    print(e) 
D()
print("created D")
