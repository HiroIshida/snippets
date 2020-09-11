# see: 
# https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
class Hoge(object):
    def __init__(self, num):
        self.num = num

    @classmethod
    def from_string(cls, string):
        return cls(len(string))

    @classmethod
    def from_lst(cls, lst):
        return cls(len(lst))

if __name__=='__main__':
    h1 = Hoge.from_string("ahoaho")
    h2 = Hoge.from_lst([1, 2, 3])
