What I mean by transitivity of convergence in probability is clarified in the following proposition.

Proposition:
> Let $X_{n}, Y_n, Z_n$ be sequences of random variable. If $(X_n - Y_n) \overset{p}{\to} 0$ and $(Y_n - Z_n) \overset{p}{\to} 0$, then $(X_n- Z_n) \overset{p}{\to} 0$.

I would like someone to check the following my proof is correct.

I found similar question https://math.stackexchange.com/questions/2442541/the-transitivity-of-the-convergence-in-probability but the transitivity in the question is different from mine.

### proof
Proof: According to the definition of the convergence in probability and definition of the limit, $\forall \epsilon >0, \epsilon' > 0$, there exists $N_1 \in \mathbb{N}$ such that $\forall k \geq N_1, \mathrm{P}(|X_k - Y_k| > \epsilon/2) < \epsilon'$. Also, there exists $N_2 \in \mathbb{N}$ such that $\forall k \geq N_2, \mathrm{P}(|X_k - Y_k| > \epsilon/2) < \epsilon'$. Now, by considering $N' = \mathrm{max}(N_1, N_2)$, for any $k > N'$ the following inequalities holds 

$\epsilon' > \mathbb{P}(|X_{k} - Y_{k}| > \epsilon/2 \mathrm{\;or\;} |Y_{k} - Z_{k}| > \epsilon/2) > \mathbb{P}(|(X_{k} -Y_{k}) + (Y_{k} - Z_{k})| > \epsilon)$.

Here, the first inequality holds dut to Fréchet inequalities. And the second inequality holds due to $|a + b| > \epsilon \Rightarrow |a| > \epsilon/2 \lor |b| > \epsilon/2$.

Thus, $\forall \epsilon >0, \epsilon'>0$ there exists $N'$ such that $\forall k>N'$, $\mathbb{P}(|X_{k} - Z_{k}| > \epsilon) < \epsilon'$ and hence $X_n - Z_n \overset{p}{\to} 0$.
