#!/usr/bin/env python
import sys

filename_original = sys.argv[1]

with open(filename_original, 'r') as f:
    string = f.read()

strategy = ''
lst = []
while True:
    try:
        before, middle, string = string.split("$", 2)
    except ValueError:
        lst.append(string)
        break
    if strategy == 'strip':
        middle_new = "\blue"
    elif strategy == 'nothing':
        middle_new = "$" + middle + "$"
    else:
        middle_new = "\\textcolor{blue}{$" + middle + "$}" 
    lst.extend([before, middle_new])
string_new = ''.join(lst)

with open(filename_original, 'w') as f:
    f.write(string_new)
