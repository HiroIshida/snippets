from cachetools.func import lru_cache

class Hoge:
    def __init__(self, N):
        self.var = 0
    def incf(self):
        self.var += 1

@lru_cache(maxsize=1000)
def cache_created_hoge(N):
    return Hoge(N)

hoge = cache_created_hoge(0)
hoge.incf()
print(hoge.var)

hoge2 = cache_created_hoge(0)
print(hoge2.var)


        
