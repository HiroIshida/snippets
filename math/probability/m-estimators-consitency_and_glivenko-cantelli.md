# M推定量の一致性について
Van der varrt の本にあるようにM推定量の一致性の十分条件には関数$f_n(\theta) - f(\theta)$が0に一様確率収束することが挙げられる. これを示すためには$f$のとりうる集合, $F = \left\{ f(\theta); \theta \in \Theta \right\}$ がGlivenko-Cantelli(GC)クラスであることを示せばよい. 直感的には, $\mathbb{E}[l(x) - u(x)] < \epsilon$ かつ$ l \leq f \leq u$となるような関数のペア$(l, u)$を考え, このようなペアによって$F$を被覆することを考える. この被覆数がいかなる$\epsilon$についても有限であるとき, GC-classである.

曖昧1. $\mathbb{E}[l(x) - u(x)]$はこれ期待値でいんだっけ? 本には$Pf := \int_{}^{} f dP$と書いているけど, 測度論的な積分を理解できてない..

GC-classである十分条件として, $\Theta$がコンパクトで$f$が連続関数であるというのがある (Van der vaart p46下).
詳しいことはわからないが, 直感的な証明は以下のような感じ. 任意のコンパクト空間上の連続関数は一様連続(ハイネカントールの定理)なので, $f(\theta)$について$u_\theta := f_\theta - \epsilon$, $l_\theta := f_\theta + \epsilon$とすると, いかなる$\epsilon$についても任意の$x$について$l \leq f_{\theta} \leq u$が成り立つような$\theta$の近傍が存在する. この近傍の半径を$\delta$とするとき, $\Theta$がコンパクトなら, $N(\theta, \delta)$を用いて$\Theta$を被覆することができる. なので, いかなる$f \in F$についても, $\Theta_{f}:=\left\{\theta; f_{\theta} = f \right\}$の少なくとも一つの要素を含むような$N(\theta^*, \delta)$が存在し, その$\theta^*$に対応する$(l, u)$区間($f_{\theta^*}\pm \epsilon$)は必ず$l\leq f \leq u$を満たす. ので, 上記の方法で作られたlu区間は$F$を被覆する.

注意: 非線形最小二乗法でノイズがのってる場合$y = g(x) + w$と書かれるけど, ここでの確率変数は$x$と$w$なので, $f$は$x$と$w$の関数になっていることに注意する.
