class Hoge:
    def __init__(self, d):
        self.d = d

class Sample:
    def __init__(self):
        self.d = {'data': None}

    def create(self):
        return Hoge(self.d)

    def set_data(self, d):
        self.d['data'] = d
        

s = Sample()
h = s.create()
print(h.d['data'])
s.set_data(3.0)
print(h.d['data'])
