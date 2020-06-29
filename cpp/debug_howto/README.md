### attaching the process.
Say I have `main.py` and `f()` inside C function wrapped by python. First please insert `usleep(1000000)` so that you have time to attach. Without this, the program is just executed. 
```cpp
#include <unistd.h>
// something
void f(){
    std::cout<<"attach this process in 3 sec." <<std::endl;
    usleep(3000000000);
    // some procedures
    return;
}
```
First running python (not from IPython). While the above process waiting for seconds hit this command to attach:
```bash
gdb --pid $(pidof python) #pidof is a bash command
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

### How to trace interactive window 
**This section is copied from [here](https://stackoverflow.com/questions/9257085/how-can-i-scroll-back-in-gdbs-command-window-in-the-tui-mode).**

One way to see the GDB output history in TUI mode is to enable logging:

    set trace-commands on
    set logging on

and then tail the log in another shell:

    cd where/gdb/is/running
    tail -f gdb.txt

This has the advantage of separating scrolling from command interaction, so you can type commands while viewing some past section of the GDB output.

None of the scrolling keys work in my CMD window, so GDB effectively consumes and destroys its own output. Switching out of TUI mode allows me to scroll up, but the output that occurred while in TUI mode is not there--the non-TUI window only shows new output generated after switching out of TUI mode. So far log and tail is the only solution I can find.

------

Edit: if you activate GDB logging (via `set logging on`) before switching to TUI mode, you may find that the logging stops upon entering TUI (this is a bug in GDB). You can toggle it back on:

    set logging off
    set logging on

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
