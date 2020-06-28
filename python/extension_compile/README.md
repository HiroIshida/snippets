## debug via gdb
see:
https://stackoverflow.com/questions/5048112/use-gdb-to-debug-a-c-program-called-from-a-shell-script
https://stackoverflow.com/questions/48141135/cannot-start-dbg-on-my-python-c-extension/53007303#53007303


## extend
The built module following [this](https://www.tutorialspoint.com/python/python_further_extensions.htm) website does not seem to work, and give me the following error when imported:
```python
>>> import helloworld
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: dynamic module does not define module export function (PyInit_helloworld)
```
After googling it, I found that extra init function is required according to [here](https://realpython.com/build-python-c-extension-module/#writing-the-init-function).
