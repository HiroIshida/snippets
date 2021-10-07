import numpy as np
import reeds_shepp
import json
q0 = [0.0, 0.0, 0.]
test_cases = []
for i in range(10000):
    print(i)
    q1 = np.random.randn(3) * 2
    q_seq = np.array(reeds_shepp.path_sample(q0, q1, 0.2, 0.01))
    test_cases.append([q1.tolist(), q_seq.tolist()])
with open('/tmp/reeds_shepp_test_cases.json', 'w') as f:
    data = {'cases': test_cases}
    json.dump(test_cases, f)
