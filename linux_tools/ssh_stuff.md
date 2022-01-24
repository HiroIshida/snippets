## 多段sshのための設定
例えば dlbox4 を経由してdlbox内のdocker containerに直にアクセスしたいとき.

```sshconfig
HOST dlbox4-docker
    User root
    HOSTNAME 172.17.02
    PORT 22
    ProxyCommand ssh dlbox4 -W %h:%p
```
