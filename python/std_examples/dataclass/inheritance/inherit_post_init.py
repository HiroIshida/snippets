from dataclasses import dataclass

@dataclass
class A:
    def __post_init__(self):
        print("class A post init")


@dataclass
class B(A):

    def __post_init__(self):
        super().__post_init__()
        print("class B post init")


B()
