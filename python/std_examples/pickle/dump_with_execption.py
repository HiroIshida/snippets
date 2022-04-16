import pickle
import numpy as np

a = np.random.randn(1000, 300000)

print("dumping...")
f = open('hoge.pkl', 'wb')
try:
    pickle.dump(a, f)
    f.close()
except:
    print("keyboard intetruppetd but let me dump again")
    f.close()
    with open('hoge.pkl', 'wb') as f:
        pickle.dump(a, f)
