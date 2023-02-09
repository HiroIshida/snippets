## question
Consider we have regions $A, B \subset [0, 1]$ where $A \cap B \neq \varnothing$. Considering $n$ sampling some points $n$ uniformly from $U(0, 1)$, and then let $N^{A}_{(n)}$ be the number of samples fall into $A$ and $N^{B|A}_{(n)}$ be the number of samples fall into $A \cap B$.

What I want to do is try to estimate the volume ratio $q = \mathrm{vol}(A \cap B) / \mathrm{vol}(A)$ by $N^{B|A}_{(n)} / N^{A}_{(n)}$ using $n$ samples, and want to know the error upper bound. Thus, mathematically, I need to know some tight upper bound of $\mathrm{Pr}[|\frac{N^{B|A}_{(n)}}{N^A_{(n)}} - q| \geq \epsilon] < \mathrm{??}$. Note that $q$ is essentially a conditional probability as I wrote in the title. 

**At least, I'm expecting that $N^{B|A}_{(n)}/N^A_{(n)} \to q$ in probability, that is $N^{B|A}_{(n)}/N^A_{(n)}$ is a consistent estimator, and the upper bound should be such that the convergence holds. But, I can't prove it, so it would be nice to know if someone has some idea**.

## note
Note that my original problem is bit more general, in the sense that $\mathbb{R}^n$ instead of $[0, 1]$, more generic probabilistic distribution rather than uniform distribution, but essentially the same as the problem above.

## my attempt so far
First, try to make a inequality for the case when the inside-absolute bars is positive (i.e. $\mathrm{Pr}[N^{B|A}_{(n)}/{N^A_{(n)}} - q > \epsilon]$)

$
N^{B|A}_{(n)}/{N^A_{(n)}} - q > \epsilon \Leftrightarrow \bigvee_{k=1}^{n}\left[ N^A_{(n)}=k \land N^{B|A}_{(n)} \leq \lfloor k(q + \epsilon)\rfloor \right] 
$

Thus, noting that $N^A_{(n)} = k$ is dispersion event for all $k$, we can decompose the probability by summation:

$
\mathrm{Pr}[N^{B|A}_{(n)}/{N^A_{(n)}}] = \sum_{i=1}^{k} \mathrm{Pr}[N^A_{(n)}=k \land N^{B|A}_{(n)} \leq \lfloor k(q + \epsilon)\rfloor]
= \sum_{k=1}^{n} \mathrm{Pr}[N^{A}_{(n)} = k] \cdot \mathrm{Pr}[N^{B|A}_{(n)} > \lfloor k(q + \epsilon)\rfloor \mid N^{A}_{(n)} = k]
$

Now, suppose there is some upper bound like $\mathrm{Pr}[N^{B|A}_{(n)} > \lfloor k(q + \epsilon)\rfloor \mid N^{A}_{(n)} = k] < \mathrm{exp}(-nf(\epsilon))$ where $f$'s domain is $\mathbb{R}_0^+$. If that the case, we can say that

$
\mathrm{Pr}[N^{B|A}_{(n)}/{N^A_{(n)}} -q > \epsilon] \leq \sum_{k=1}^{n} \binom{n}{k} p^{k} (1 - p)^{(n - k)} \cdot \mathrm{exp}(-nf(\epsilon)) =\left(  p\cdot\mathrm{exp}(-nf(\epsilon))+(1-p) \right)^{n} = (1 - p \cdot\mathrm{exp}(-nf(\epsilon)))^n
$
where $p$ is the probability that the sample falls into $A$ (i.e. $\mathrm{vol}(A)$) and in that the case the probability of $k$ samples fall into $A$ follows binomial distribution. Also, note that I used binomial theorem at the end.

I think, the same is hold for negative side: $\mathrm{Pr}[q - N^{B|A}_{(n)}/{N^A_{(n)}} > \epsilon] \leq (1 - p \cdot\mathrm{exp}(-nf(\epsilon))^n$. Thus, the estimation by sample is converge to the true ratio $q$ in probability.

**So, as a building block, proving the upper bound of the form** $\mathrm{Pr}[N^{B|A}_{(n)} > \lfloor k(q + \epsilon)\rfloor \mid N^{A}_{(n)} = k] < \mathrm{exp}(-nf(\epsilon))$ **is essential, but I couldn't**. I considered to apply hoeffding inequality to do that as in sample quantile accuracy bound estimation, but don't know how to do this for conditional case like here.
