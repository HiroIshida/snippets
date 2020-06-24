The built module following [this](https://www.tutorialspoint.com/python/python_further_extensions.htm) website does not seem to work, and give me the following error when imported:
```python
>>> import helloworld
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: dynamic module does not define module export function (PyInit_helloworld)
```
After googling it, I found that extra init function is required according to [here](https://realpython.com/build-python-c-extension-module/#writing-the-init-function).
