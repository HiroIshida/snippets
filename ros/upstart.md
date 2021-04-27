upstartで立ち上がるものは`/etc/init`以下にあり, ros関係のものは`jsk-pr2-starup.conf`と`robot.conf`がある. それぞれに対応するログは`/var/log/upstart/robot.log` (confをlogにおきかえた名前)の中にある.読む時は`sudo tail -f`とする.  `initictl start robot`や`stop robot`はこのconfファイル以下のものをstartしたりstopしたりしている. 


