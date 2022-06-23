## まとめ
- /etc/initに置かれているconfファイルを読んで実行される. 
- even driven なので例えばjsk-pr2-startup.confだと robot-is-up eventが発行されたら処理がはじまる. 
- こうしたeventはconfファイルの終了と共に発行することができ, 例えば etc/init/robot.conf だと, 終了とともに `initctl emit robot-is-up`を発行し, jsk-pr2-startupをトリガーしている.

## 疑問 
c2-comp-upとかどこでemitされてる?

