# 2: A Gentle Start
## 定義

分類器の汎化誤差:
$$ L_{D, f}(h) = \mathbb{P}_{x\sim D}\left[ h(x) \neq f(x) \right] = D(\{x: h(x) \neq f(x)\}) $$
ただし$D(A)$は集合を引数にとって確率を返す関数. 直積測度. 

分類器の経験誤差:
$$ L_{S}(h) = \frac{1}{m} \sum_{i=1}^m \mathbb{I}\{h(x_i) \neq y_i)\} $$

ERM(経験リスク最小化): $h_S$は経験誤差が最小となる仮説であり
$$ h_S = \arg\min_{h\in H} L_{S}(h) $$

## 仮説集合が有限で実現可能仮定をおいた場合.
実現可能仮定(realizable assumption)とは, $\exists h \in H, \mathbb{P}_S [L_{D, f}(h) = 0] = 1$.

RAが成り立つなら, いかなるサンプル集合に対しても$h_S$は$0$の経験誤差を持つ. すなわち, 
$$ \mathbb{P}_S [L_{S}(h_S) = 0] = 1 $$
ERMにしたがって得られた仮説による汎化誤差が$\epsilon$よりも大きくなる確率($S\sim D$のとき)を
$ \mathbb{P}_{S\sim D^m}[L_{D, f}(h_S) > \epsilon] $ を上から抑えたい.
ただし$D_m$の表記がわかりにくいので, 適宜$\mathbb{P}_{S\sim D^m}$とか$\mathbb{P}_{S}$と書き換えます.

悪い仮説, すなわちERMで学習した際に汎化誤差が$\epsilon$よりも大きくなってしまうような仮説集合$H_{B}$は次のようにかける:
$$ H_{B} = \{ h\in H: L_{D, f}(h) > \epsilon \}. $$

また, 経験誤差が0なのに汎化誤差は$\epsilon$にしてしまうような, 紛らわしい仮説を生み出しかねないサンプル集合の集合$M$(misleading samples)は次のようにかける
$$ M = \{S| \exists h \in H_{B}, L_{S}(h) = 0\} $$
ここで$ M = \{S| h_S \in H_{B}\}$ ではないことに注意する. これは, ERMによって得られる経験誤差最小仮説が複数存在するかもしれないからである.

