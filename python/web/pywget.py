import sys
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
addr = 'https://raw.githubusercontent.com/HiroIshida/tinyfk/master/data/pr2.urdf'
data = urlretrieve(addr, '/tmp/pr2.urdf')
print(data)
