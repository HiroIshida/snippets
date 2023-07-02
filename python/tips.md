##  pip install -e . 時にsys.pathがおかしくなる
pyproject.tomlがcurrent directory以下にあると, pathが消されてしまう. desired behaviorなのかどうかは不明. temp解決策はpyproject.tomlを削除
