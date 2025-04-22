In one terminal
```
socat -d -d pty,raw,echo=0 pty,raw,echo=0
```
/det/pts/number is depending on the output of the above command. For example, if the output is `/dev/pts/3` and `/dev/pts/4`, then run the following commands in two different terminals:

```bash
./build/client /dev/pts/3                                                                                                                                                                                          
```
```bash
./build/server /dev/pts/4
```
