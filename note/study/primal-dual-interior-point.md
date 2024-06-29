だいたい[1]のp498-p510に対応.

## Barrier method
Consider an optimization problem
$$ \min{f(x)} \;s.t.\; c(x) \geq 0 $$
Barrier method solves this problem by solving the following unconstrained problem
$$ x^*(\mu) := \argmin{f(x) - \mu \sum_{i=1}^{m} \log{(c_i(x))}} $$
where $\mu$ is a positive parameter and $c_i(x)$ is the $i$-th constraint.
Each iteration of the barrier method solves the above problem for a decreasing sequence of $\mu$ values. The initial solution of the each subproblem is the solution of the previous subproblem. **ただし, $c_i(x) < 0$ の場合logは定義されない, つまり, 初期解は常に実行可能領域内に入っている必要がある.**

### Barrier method and KKT conditions
Barrier methodにおける目的関数を微分すると
$$ \nabla f(x) - \sum_{i=1}^{m} \frac{\mu}{c_i(x)} \nabla c_i(x) = 0 $$
となる. $\frac{\mu}{c_i(x)}$ はラグランジュ乗数の近似とみなすことができるだろう. ところで, もともとの最適化問題のKKT条件は次のように書けるが,
$$ \begin{align*}
\nabla f(x) + \sum_{i=1}^{m} \lambda_i \nabla c_i(x) &= 0 \\
\lambda_i &\geq  0 \\
c_i(x) & \geq 0 \\
\lambda_i c_i(x) &= 0
\end{align*} $$
$\lambda_i = \frac{\mu}{c_i(x)}$ とおけば, $\mu$が十分小さいとき, Barrier methodの解はKKT条件を満たす近似解となる. ($\mu$は当然positiveだし, $c_i(x) > 0$ でないとlogが定義されないので, $\lambda_i \geq 0$ は明らかに満たされている.) つまり, Barrier methodは$mu$をだんだん小さくしていくことで, KKT条件の近似率を上げていく方法であるともいえる.

### Primal-dual interior point method(PDIPM)
PDIPMでもKKT条件の近似率を徐々に上げていく方法であるといえるが, Barrier methodと異なり, ラグランジュ定数を明示的な最適化変数として扱っている点がポイントである. また, スラック変数$s$を用いて不等式制約を単なるbound制約に置き換えている. つまり, PDIPMのKKTシステムは次のように書け,
$$ \begin{align*}
\nabla f(x) + \sum_{i=1}^{m} \lambda_i \nabla c_i(x) &= 0 \\
c_i(x) - s_i &= 0 \\
\lambda_i s_i &= \mu \\
s_i &\geq 0, \; \lambda_i \geq 0
\end{align*} $$
PDIPMでは最後のbound不等式を除いたKKT systemを$(x, \lambda, s)$について解くことで, step directionを求める. step sizeについては, nonlinear KKT systemのresiudalが最小になり, かつ, $s_i \geq 0, \lambda \geq 0$, となるように選ぶ. (ほんまかいな). backtracking line searchを使う[3]. 
PDIPMはbarrier methodと比べて
- ステップ毎に完全なnonlinear optimization problemを解かなくてもよく, 単に線形方程式を解くだけでよい.
- logが出てこないので, 初期解が実行可能領域内に入っている必要がない. (と僕は思うんだけど, [1, 3]では書いてない?)

## resources
- [1]  Wright, Stephen J. "Numerical optimization." (2006).
- [2] Ryan T. 授業 https://www.youtube.com/watch?v=Wj2HjHDFlKg&ab_channel=RyanT
- [3] スライド: https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf
