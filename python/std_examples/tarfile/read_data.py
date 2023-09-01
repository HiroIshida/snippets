from io import StringIO, BytesIO
import numpy as np
import tarfile
import pickle

# Read the tarfile
# https://stackoverflow.com/questions/2018512/reading-tar-file-contents-without-untarring-it-in-python-script

with tarfile.open("tempdir.tar", mode="r") as tar:
    for member in tar.getmembers():
        f = tar.extractfile(member)
        cont = pickle.loads(f.read())

# Note: mode must be "a" instead of "w" !!
with tarfile.open("tempdir.tar", mode="a") as tar:
    cont_add = np.random.randn(100)
    cont_add_byte = pickle.dumps(cont_add)
    file_like = BytesIO(cont_add_byte)
    tar_info = tarfile.TarInfo(name="cont_add.pkl")
    tar_info.size = len(cont_add_byte)
    tar.addfile(tarinfo=tar_info, fileobj=file_like)
