import copy

class Hoge():
    def __init__(self, val): self.val = val

a = [Hoge(1), Hoge(2)]
b = copy.copy(a)
b.append(Hoge(3))

assert len(a) == 2
assert len(b) == 3
b[0].val = 777
assert a[0].val == 777
