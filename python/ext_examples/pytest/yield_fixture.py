import pytest
import os
import shutil
from pathlib import Path

@pytest.fixture(autouse=True)
def hoge() -> Path:
    p = Path("./tmp").expanduser()
    p.mkdir(exist_ok=False)
    print("created directory")
    yield p
    shutil.rmtree(p)
    print("deleted directory")
    assert not p.exists()


def test_hoge(hoge):
    p = hoge
    print(p)
    assert False
    # note that even if test failed with assert, termiantion process is called
