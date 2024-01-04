##  pip install -e . 時にsys.pathがおかしくなる
pyproject.tomlがcurrent directory以下にあると, pathが消されてしまう. desired behaviorなのかどうかは不明. temp解決策はpyproject.tomlを削除

## debugging with gdb
```
sudo apt install python3-dbg
gdb --args python3 your_script.py
```

## deadlock when multiprocessing
最も簡単なのはspawnすること. spawnすると不要なglobal変数(e.g. lock)とかがコピーされない. 
spawnするとかなり遅いので, forkを使わないといけない場合は次のことを試す.
- loggerのlockを強制的にリリース
- pytorch (numpyも??)を内部で使っている場合, threadpoolctlでblasのthreadを1に制限

