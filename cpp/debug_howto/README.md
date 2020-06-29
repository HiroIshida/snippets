### attaching the process.
Say I have `main.py` and `f()` inside C function wrapped by python. First embed `usleep(1000000)` or something. While sleeping run 
```bash
gdb --pid <PID-of-main.py>
```
Another tip is (a) to show pid and wait for press as below:
```python
import os
import psutil
pid = os.getpid()
process = psutil.Process(pid)
print(process.name)
###  
input("press")
func_youwanna_debug()
```
The above idea originally comes from this [post](https://stackoverflow.com/questions/57612531/step-from-pdb-in-gdb-when-debugging-c-extension)

### pyenv
https://stackoverflow.com/questions/48141135/cannot-start-dbg-on-my-python-c-extension
As suggested by Jean-Francois Fabre, the python file installed by pyenv is actually a bash script. You can easily make gdb run this script with:

    gdb -ex r --args bash python mycode.py

See this question for other approaches: https://stackoverflow.com/q/5048112/6646912


### commands
https://www.youtube.com/watch?v=bWH-nL7v5F4
layout next
next (n)
break (b)
### how to define custom functions 
https://stackoverflow.com/questions/3832964/calling-operator-in-gdb
