## example: where is the pudb's cache file that stores the breakpoints

```
sudo strace -e trace=openat -p $(ps -aux |grep pudb|head -1|awk '{print $2}') -o /tmp/pudb_ptrace.txt
cat /tmp/pudb_ptrace.txt |grep -v ".so"|grep -v ".py"
```

```
...
openat(AT_FDCWD, "/home/h-ishida/.config/pudb/internal-cmdline-history.txt", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 8
openat(AT_FDCWD, "/home/h-ishida/.config/pudb/saved-breakpoints-3.8", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 8
...
```

