## question
Let $\hat{F}(t, x_{1:m}) := \frac{1}{m} \sum_{i=1}^{m}1_{[x_i < t]}$ be empirical distribution function, where we write $\hat{F}$ explicitly as a function of sample $x_{1:m}$ for clarity.
Also, let us define $\alpha$-sample-quantile as $Q_{\alpha}(x_{1:m}):= \mathrm{inf}\left\{ t; \hat{F}(t, x_{1:m}) \geq \alpha \right\}$.

Now, what tight upper bound can we have for the following probability?:

$$ \mathbb{P}(\; \mathbb{E}_{x'_{1:m}}[\hat{F}(Q_{\alpha}(x'_{1:m}), x'_{1:m})] \;-\; \mathbb{E}_{x'_{1:m}}[\hat{F}(Q_{\alpha}(x_{1:m}), x'_{1:m})] > \epsilon) \leq \quad ??$$

Note that $\mathbb{E}_{x'_{1:m}}$ means expected value when sample $x'_{1:m}$ is sampled many times. Therefore, value in the leftside $\mathbb{E}_{x'_{1:m}}[\hat{F}(x'_{1:m}, Q_{\alpha}(x'_{1:m}))]$ is static (not a random variable). On the other hand, the right side value $\mathbb{E}_{x'_{1:m}}[\hat{F}(x'_{1:m}, Q_{\alpha}(x_{1:m}))]$ is a random variable because it is a function of $x_{1:m}$, which is a random variable.

I think, we could potentially use some famous concentration inequality (e.g. Hoeffding's or Mcdiarmid's inequality), however I've been struggling doing that.


## chatgpt's answer
You're right that a concentration inequality might be useful here. Since we are working with empirical distribution functions and sample quantiles, the Glivenko-Cantelli theorem and the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality are particularly relevant.

First, let's state the DKW inequality. For any $\epsilon > 0$ and any distribution function $F$,

$$
\mathbb{P}\left( \sup_{t} |\hat{F}(t, x_{1:m}) - F(t)| > \epsilon \right) \leq 2 \exp(-2m\epsilon^2).
$$

Now, we want to bound the probability

$$
\mathbb{P}\left( \mathbb{E}_{x'_{1:m}}[\hat{F}(x'_{1:m}, \xi_{\alpha}(x'_{1:m}))] - \mathbb{E}_{x'_{1:m}}[\hat{F}(x'_{1:m}, \xi_{\alpha}(x_{1:m}))] > \epsilon \right).
$$

Let $G_\alpha$ be the true $\alpha$-quantile of the distribution, i.e., $G_\alpha := \inf\{t; F(t) \geq \alpha\}$. Using the DKW inequality, we can write:

$$
\mathbb{P}\left(\left|\xi_\alpha(x_{1:m}) - G_\alpha\right| > \epsilon\right) \leq 2\exp(-2m\epsilon^2).
$$

Then, we can condition on the event $\left|\xi_\alpha(x_{1:m}) - G_\alpha\right| \leq \epsilon$ and use the conditional probability:

$$
\mathbb{P}\left(\mathbb{E}_{x'_{1:m}}[\hat{F}(x'_{1:m}, \xi_{\alpha}(x'_{1:m}))] - \mathbb{E}_{x'_{1:m}}[\hat{F}(x'_{1:m}, \xi_{\alpha}(x_{1:m}))] > \epsilon\right) \leq \mathbb{P}\left(\left|\xi_\alpha(x_{1:m}) - G_\alpha\right| > \epsilon\right) \leq 2\exp(-2m\epsilon^2).
$$

This gives you a tight upper bound on the probability you are looking for, in terms of the sample size $m$ and the difference $\epsilon$. Note that this bound might not be the tightest possible, but it gives you an idea of how the probability decreases as the sample size increases and the difference $\epsilon$ decreases.


