import re
import time

string = 'tmpu7d7puwo._std_msgs__String'
ts = time.time()
m = re.match(r"(\w+)._(\w+)__(\w+)", string)
print(time.time() - ts)
print(m.group(2))
print(m.group(3))
