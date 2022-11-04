使い方は非常に簡単で次のコマンドをcmakelistに加えるだけ.
```cmake
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)
```

### pythonのwheelをasanとともにbuildしたときにおきる問題
```
h-ishida@stone-jsk:~/python/hifuku/example$ python3 pr2_tabletop_world.py 
ERROR: ld.so: object 'libasan.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.12) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
==153921==ASan runtime does not come first in initial library list; you should either link runtime to your application or manually preload it with LD_PRELOAD.
```

`ldd`等でlibasanの場所を見つけたあと, python fileを実行する前に
```
export LD_PRELOAD=/lib/x86_64-linux-gnu/libasan.so.5
```
を行う.
