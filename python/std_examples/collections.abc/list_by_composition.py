from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar, List

T = TypeVar('T')

class ListByComposition(Sequence, Generic[T]):

    @abstractmethod
    def get_composed_list(self) -> List[T]:
        pass

    def __iter__(self):
        return self.get_composed_list().__iter__()

    def __getitem__(self, indices_like):
        return self.get_composed_list()[indices_like]

    def __len__(self):
        return len(self.get_composed_list())


from dataclasses import dataclass

@dataclass
class Example(ListByComposition):
    lst: List

    def get_composed_list(self):
        return self.lst

e = Example([1, 2, 3])
