class MyQueue:
    def __init__(self, N):
        self.N = N
        self.data = [np.zeros(3) for n in range(N)]

    def push(self, elem):
        tmp = self.data[1:self.N]
        tmp.append(elem)
        self.data = tmp

    def mean(self):
        s_est_lst = [np.mean(np.array([s[i] for s in self.data])) for i in range(3)]
        return np.array(s_est_lst)

