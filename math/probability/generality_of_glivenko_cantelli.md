# Applicability of glivenko-Cantelli theorem to more generic $F$
(C) Hirokazu Ishida

Glivenko-Cantelli theorem states that approximation function $F_n(t):=\frac{1}{n} \sum_{i=1}^{n} [X_i < t]$ uniformly converges to actual cumulative distribution function $F$. That is, $\mathrm{sup}_{t} ||F_n(t) - F(t)|| \to 0$, a.s.

Reading through the proof of the theorem, I found that the proof uses only the three properties of the cumulative distribution function $F$ and its sample-approximation function $F_n$

Property 1: $\forall t, F_n(t) \to F(t)$, a.s.

Property 2: $F$ is bounded function

Property 3: $F$ and $F_n$ is monotonically increasing function

Thus, I think any function-approx-function-pair $(f_n, f)$ satisfies the all three properties, then Glivenko-Cantelli theorem can also be applied to that pair, and conclude that $\mathrm{sup}_t||f_n(t) - f(t)|| \to 0$, a.s.

Is this correct? Or, is there other important properties that $(f, f_n)$ must have?



### Proof of Glivenko-Cantelli
For your information, here is that proof of Glivenko-Cantelli fetched from (Van der vaart, Asymptotic Statistics, Thm. 19.1) .

By the strong law of large numbers, both $F_{n} \to F$, a.s., and $F_n(t-) \to F(t-)$, a.s for given $t$. (Property 1 used). 

Given a fixed $\epsilon > 0$, there exists a partition $-\infty = t_0 < t_1 < \cdots < t_k = \infty$ such that $F(t_i-)-F(t_{i-1}) < \epsilon$ for every $i$. (Property 2 and 3 used)

Now for $t_{i-1}\leq t < t_i$, 

$ F_n(t) - F(t) \leq F_n(t_i -) - F(t_i -) + \epsilon$ (Property 3 used)

$ F_n(t) - F(t) \geq F_n(t_{i-1}) - F(t_{i-1}) - \epsilon$ (Property 3 used)

The onvergene of $F_n(t)$ and $F_n(t-)$ for every fixed $t$ is certainly uniform for $t$ in the finite set $\{t_1,\ldots, t_n\}$. Conclude that $\mathrm{limsup} (\mathrm{sup}_{t}||F_n(t) - F(t)|) \leq \epsilon$ almost surely. This is true for every $\epsilon$ and hence the limit superior is zero.
