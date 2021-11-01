import os
import re

fname1 = "/tmp/hogehoge_unkounko"
fname2 = "/tmp/hogehoge_rr3rj38j"
for fname in [fname1, fname2]:
    open(fname, 'a').close()

fnames = os.listdir("/tmp")
for fname in fnames:
    m = re.match(r'hogehoge*.', fname)
    if m is not None:
        print(fname)

