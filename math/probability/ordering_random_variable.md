step1: $x \sim U(0, 1)$を$n$回繰り返す. サンプルを小さい順に並べたとき$k$番目に小さいサンプルをの座標の確率密度関数を求める.

累積確率密度関数$F(x)$とする. $Pr[x < X < x + \epsilon] = F(x + \epsilon) - F(x)$より確率密度関数の定義から
$$
f(x) = \lim_{\epsilon \to 0}\frac{F(x + \epsilon) - F(x)}{\epsilon} = \lim_{\epsilon \to 0}\frac{\mathrm{Pr}[x < X < x + \epsilon]}{\epsilon}
$$
(日本語めんどいので英語で)Among $n$ particles, $k-1$ particles falls into $[0, x)$, 1 particle into $[x, x + \epsilon)$ and $n - k + 1$ particles into $[x + \epsilon, 1]$. Therefore, using multi-nominal distribution, we can write
$$
f_{n,k}(x) = \lim_{\epsilon \to 0} \frac{1}{\epsilon} n! \frac{x^{k-1}}{(k-1)!}\cdot\frac{(1-x-\epsilon)^{n-k}}{(n-k)!}\cdot \frac{\epsilon}{1} = x^{k-1} (1 - x)^{n-k} n _{n-1}C_{k-1}
$$

step2: Let us calculate $\lim_{n \to \infty}\mathbb{E}[X] = b$ when $k = [b * n]$.

$$
\mathbb{E}[X_{n,k}] = \int_{0}^{1} f_{n, k}(x)x dx = n _{n-1}C_{k-1}\int_{0}^{1} x^{k} (1 - x)^{n-k} dx
$$
Noting that beta-function integral formula $\int_{0}^{1}  x^m(1-x)^n dx = \frac{m!n!}{(m+n+1)!}$, we can compute that
$$
\mathbb{E}[X_{n,k}] = n _{n-1}C_{k-1}\cdot \frac{1}{(n+1)_{n}C_{k}} = k\cdot \frac{n \cdot n!}{(n+1)!}
$$
Thus, $\lim_{n \to \infty}\mathbb{E}[X] = b$ when $k = [b * n]$, and the estimation by the above procedure (sort and take $[\beta n]$-th element) is consistent estimator (一致推定量). これ強一致性かな? 調べる.
