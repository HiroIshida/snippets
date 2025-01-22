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

## pyproject baseの editable install時のpath解決の仕組み
### まずpthファイルとは

site package内に`__editable__.mujoco_xml_editor-0.0.0.pth`が作成される. その内部は以下のようになっている.
```python
import __editable___mujoco_xml_editor_0_0_0_finder; __editable___mujoco_xml_editor_0_0_0_finder.install()
```
さらに, ...instlal()が呼ばれて, pathが追加される.
この方法でのeditable installはほとんどのIDE/LSPは対応していない.
https://github.com/microsoft/pyright/issues/3846
