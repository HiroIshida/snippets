import tempfile

with open('tmp', 'wb') as f:
    for i in range(100):
        f.write(int(i).to_bytes(8, 'big'))

with open('tmp', 'rb') as f:
    f.seek(87 * 8)
    a = int.from_bytes(f.read(8), 'big')
    print(a)

