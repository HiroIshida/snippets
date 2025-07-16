## Enroot の仕組み
- linuxのコマンドでchrootを使うとあるプロセスのルートディレクトリを指定したものに変更することができる.
- コンテナ的なもののroot以下の構造をまるっと圧縮したsqshファイルというのをrootとして指定すると、コンテナライクに使用することができる.
- また, unshareというprocess空間を分離するコマンドで上記プロセスをPID=1に設定しprocess空間を分離する.

## Enrootにおけるimage vs container
imageはenroot createで起動. containerはenroot startで起動.
[要調べ] enroot createはsqshを解凍する処理が入るから遅い. startは解凍したものをmountするだけだからはやい.
