import traceback

def f():
    g()

def g():
    for line in traceback.format_stack():
        print(line.strip())

f()

