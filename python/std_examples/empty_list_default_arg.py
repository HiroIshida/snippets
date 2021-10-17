class Strange:
    def __init__(self, a=[]): self.a = a
    def push(self, b): self.a.append(b)

class Sane:
    def __init__(self): self.a = []
    def push(self, b): self.a.append(b)

for cls in [Strange, Sane]:
    print("testing class: {}".format(cls.__name__))
    u = cls()
    u.push({})
    print(u.a)

    u2 = cls()
    print(u2.a)
