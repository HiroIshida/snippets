## find upper bound of the probability that estimator's error is greater than $\epsilon$, when we already have upper bound depending on the target value
estimator's error
Suppose we have an estimator that estimator a value $a \in \mathbb{R}$. And let the estimator's output be $\hat{x}$ which should be well approximate $a$. Also, suppose we have an inequality of the form

$ \mathrm{Pr}[|\hat{x} - a| > \epsilon] \leq f(a, \epsilon) $

where $\epsilon > 0$ and $f: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ is a strictly monotonically decreasing continuous function with respect to the both two parameter. (In my actual problem $f$ takes an exponential form).

What I want to do is find a good upper bound of my estimation which is independent of $a$. The above inequality is useless as it current form because $f$ depends on $a$, which I want to estimate. Is there any way to get upper bound independent from $a$? Or, can you prove that we can't?
