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

### エラーがでたらその場で終了してもらう方法
```bash
export ASAN_OPTIONS='abort_on_error=1'/
```

### pythonだとcleanな環境でもinterpreterが終了するタイミングでmemory leakおきてる. これはとりあえず無視して..
h-ishida@0cfa39147976:~$ python3
Python 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> quit
Use quit() or Ctrl-D (i.e. EOF) to exit
>>> quit()

=================================================================
==137==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 77080 byte(s) in 70 object(s) allocated from:
    #0 0x7ffb507d0808 in __interceptor_malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cc:144
    #1 0x55a5d07a34b2 in PyObject_Malloc (/usr/bin/python3.10+0x11d4b2)

Direct leak of 4323 byte(s) in 7 object(s) allocated from:
    #0 0x7ffb507d0808 in __interceptor_malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cc:144
    #1 0x55a5d07a4d4c  (/usr/bin/python3.10+0x11ed4c)

Indirect leak of 4719 byte(s) in 5 object(s) allocated from:
    #0 0x7ffb507d0808 in __interceptor_malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cc:144
    #1 0x55a5d07a4d4c  (/usr/bin/python3.10+0x11ed4c)

SUMMARY: AddressSanitizer: 86122 byte(s) leaked in 82 allocation(s).
