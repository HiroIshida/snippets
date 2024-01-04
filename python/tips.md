##  pip install -e . 時にsys.pathがおかしくなる
pyproject.tomlがcurrent directory以下にあると, pathが消されてしまう. desired behaviorなのかどうかは不明. temp解決策はpyproject.tomlを削除

## debugging with gdb
```
sudo apt install python3-dbg
gdb --args python3 your_script.py
```
