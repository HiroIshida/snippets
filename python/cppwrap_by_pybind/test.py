#!/usr/bin/env python
import example
momo = example.Pet("momo")
vec = momo.getvec()
print(vec)
momo.pushvec(1)
momo.pushvec(2)
momo.pushvec(4)
momo.pushvec(8)
vec = momo.getvec()
print(vec)
