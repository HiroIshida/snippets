class A(object):
    def __init__(self):
        pass

    @classmethod
    def say(cls):
        print(cls.__name__)

class B(A):
    pass

b = B()
b.say()
