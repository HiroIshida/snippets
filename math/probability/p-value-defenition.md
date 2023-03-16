Could you provide any formal definition for p-value? Or, do you have any good source for it?

For a day, I've been searching for the formal definition of p-value, but I couldn't yet. The majority of the book of statistics are mathematically non-rigorous and most of them didn't define **p-value** formally. I found two books below that are mathematically rigorous, but the definition by these two are also ambiguous.

For example, in Mathematical Statistics by Jun Shao:

> It is good practice to determine not only whether $H_0$ is rejected or
> accepted for a given $\alpha$ and a chosen test $T_\alpha$, but also
> the smallest possible level of significance at which $H_0$ would be
> rejected for the computed $T_{\alpha}(x)$, i.e. $\hat{\alpha} = \mathrm{inf}\left\{ \alpha \in (0, 1) : T_\alpha(x) = 1 \right\}$. Such an $\hat{\alpha}$, which depends on $x$ and the chosen test and is a
> static, is called the *p-value* for the test $T_{\alpha}$.

However, In this book, $T_{\alpha}$ is not defined before the above statement.

In other book, p63, "Testing Statistical Hypothesis 3rd edition" by E.L. Lehmann:

> ... When this is the case, it is good practice to determine not only
> whether the hypothesis is accepted or rejected at the given
> significance level, but also to determine the smallest significance
> level, or more formally
> 
> $$ \hat{p} = \hat{p}(X) = \mathrm{inf}\left\{ \alpha : X \in S_{\alpha} \right\} $$
> 
> at which the hypothesis would be rejected for the given observation.
> This number, the so-called **p-value** given an idear of how strongly
> the data contradicts the hypothesis.

But unfortunately I couldn't find the definition of $S_{\alpha}$ so the same situation as Shao's book.
