from typing import Dict
from typing import NewType

MyDict = NewType('MyDict', Dict[str, int])

def hoge(m : MyDict):
    pass

a = MyDict({"hoge": 1})
hoge(a) # permitted
hoge({"aho": 2}) # not permitted 

# on the other hand, if you try to do wihtout NewType..

YourDict = Dict[str, int]
def hage(m : YourDict):
    pass
hage({"aho": 2}) # this is permitted

# which means, by using NewType, you can clearly enforce the type 
