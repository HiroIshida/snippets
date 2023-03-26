## Probability inequality for a statistic with lower bound constraint, where McDiarmid's inequality is used in proof
I'd like to prove a probability inequality involving a statistic and a random variable as in the following statement. My questions are two-fold:
1. Is the following proof using McDiarmid's inequality correct?
2. Is this kind of inequality considering a lower bound constraint common in statistics? If so I would like to know related fields. 

## statement
Let $X_1, ..., X_m \in \mathbb{R}^{n}$ be independent random variables. Let $f: \mathbb{R}^n \to \mathbb{R}$ be a function. Let $Z(X_{1:m}, \tau) := \frac{1}{m}\sum_{i=1}^{m} [f(X_i) > \tau]$ be a random variable dependent on $\left\{ X_i \right\}_{i=1}^{m}$. Let us define a statistic $\tau_{\alpha, \beta}(X_{1:m}):= \mathrm{sup}\left\{\tau: Z(X_{1:m}, \tau) - \sqrt{\frac{1}{2m} \log(\frac{1}{\alpha})} > \beta \right\} $. Let $\phi(X_{1:m}):=Z(X_{1:m}, \tau_{\alpha, \beta}(X_{1:m}))$. Then for all $\alpha < 1$,

$$ \mathbb{P}\left[  \mathbb{E}(\phi(X_{1:m})) > \phi(X_{1:m}) - \sqrt{\frac{1}{2m} \log(\frac{1}{\alpha})} > \beta \right]  \leq \alpha $$

## proof
Noting that $|\phi(X_{1:m}) - \phi(X_1, \ldots, X_i', \ldots, X_m)| < (1/m)$. Therefore, by direct application of McDiarmid's inequality, we have

$$ \mathbb{P}\left[  \mathbb{E}(\phi(X_{1:m})) > \phi(X_{1:m}) - \epsilon \right]  \leq \exp(-2m\epsilon^2) $$

Thus, by substituting $\epsilon = \sqrt{\frac{1}{2m} \log(\frac{1}{\alpha})}$, we can say:

$$ \mathbb{P}\left[  \mathbb{E}(\phi(X_{1:m})) > \phi(X_{1:m}) - \sqrt{\frac{1}{2m} \log(\frac{1}{\alpha})} \right]  \leq \alpha $$

Also, by the definition of $\tau_{\alpha, \beta}$ and $\phi$, $\mathbb{P}[\phi(X_{1:m}) - \sqrt{\frac{1}{2m} \log(\frac{1}{\alpha})} > \beta] = 1$. Thus the statements follows.
Note that the reason why I used McDiarmid's instead of Hoeffding's inequality is that Hoeffding's inequality cannot handle not-independent variables, but McDiarmid's can.
