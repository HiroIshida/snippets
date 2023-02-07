$F$を累積確率分布とする. $\alpha$-quantile を次のように定義する$\xi_{\alpha}:=\mathrm{inf}(\{x; x > \alpha\})$. また$\xi_{\alpha, n}$をn個のサンプルを順番に並べて$[n\alpha]$とる量, すなわち標本分位点とする.

$\xi_{\alpha, n}$が$\xi_{\alpha}$に確率収束するとは以下の式か成り立つことであり, これを示すことが本ノートのゴールである.
$$
\forall \epsilon > 0, \mathrm{lim}_{n\to \infty}\mathbb{P}[|\xi_{\alpha, n} - \xi_{\alpha}| > \epsilon] \to 0
$$

まず着目すべきなのが, 式中の$|\xi_{\alpha, n} - \xi_{\alpha}| > \epsilon$の形である. 標本からの推定量から目的となる値を引いたもの, という形がHoefddingの不等式における``標本平均から真の平均を引いたもの"という形に似ていることに気づくはずである.

ということに注意して, 次のような式変形をしていく.
$$
\mathbb{P}[\xi_{\alpha,n} - \xi_{\alpha} > \epsilon] = \mathbb{P}\left[  \sum_{i=1}^{n} I[X_i > \xi_{\alpha} + \epsilon] > n\alpha\right]
$$
ここで右辺は``$N$個のサンプルのうち少なくとも$n\alpha$個が$\xi_{\alpha, n} + \epsilon$以上の値をとる確率"である. ここで$I[X_i > \xi_{\alpha} + \epsilon]$を$Y^{+}_i$と書くと, この$Y^+_i$の期待値はサンプルが$\xi_{\alpha, n} + \epsilon$以上の値をとる確率であることがわかる. 次の関係式がなりたつ.
$$
\mathbb{E}[Y^{+}_{i}]=1-F(\xi_{\alpha} + \epsilon)
$$
このことに着目して, 上式をHoefddingの不等式を利用して変形すると
$$
\mathbb{P}[\xi_{\alpha,n} - \xi_{\alpha} > \epsilon] = \mathbb{P}\left[  \sum_{i=1}^{n} Y^+_i > n\alpha\right] = \mathbb{P}\left[ \sum_{i=1}^{n} Y_i^+ - \sum_{i=1}^{n}\mathbb{E}[Y^+_i] < n(F(\xi_{\alpha} + \epsilon) - \alpha) \right]  \leq \mathrm{exp}(-2n (F(\xi_{\alpha} + \epsilon) - \alpha))
$$
同様に次のことが示せる
$$
\mathbb{P}[\xi_{\alpha,n} - \xi_{\alpha} < -\epsilon] \leq \mathrm{exp}(-2n (\alpha - F(\xi_{\alpha} - \epsilon)))
$$
よって, Frechet の不等式を用いて
$$
\mathbb{P}[|\xi_{\alpha, n} - \xi_{\alpha}| > \epsilon] = \mathrm{min}(\mathrm{exp}(-2n (F(\xi_{\alpha} + \epsilon) - \alpha)), \mathrm{exp}(-2n (\alpha - F(\xi_{\alpha} - \epsilon))))
$$
よって$\xi_{alpha, n}$は$\xi_{\alpha}$に確率収束する.
