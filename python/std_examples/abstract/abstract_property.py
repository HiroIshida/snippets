from abc import ABC, abstractmethod, abstractproperty


class Person(ABC): 
    # DO NOT USE abstractproperty, it's deprecated
    @property
    @abstractmethod
    def age(self) -> int:
        pass

class Ishida(Person):  # NG
    pass

class Yamaguchi(Person):  # OK
    @property
    def age(self) -> int:
        return 10

class Murooka(Person):  # OK
    age: int = 20

# abstract property can either be property method or normal property
y = Yamaguchi()
m = Murooka()
i = Ishida()
