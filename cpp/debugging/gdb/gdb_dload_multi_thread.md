# debug ros2 composed container
How to find process id
```bash
gdb --pid $(ps -ef|grep parking_cont|awk NR==1'{print $2})'
```
then `continue`
