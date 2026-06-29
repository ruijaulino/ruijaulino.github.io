# Tracking Performance of Optimal Strategies

A model may perform well in research and backtesting yet fail once deployed. This can happen for many reasons: signal decay, structural breaks, increased competition, or simply false discovery.

When performance deteriorates, we usually rely on heuristics: stop after a certain drawdown, reduce exposure after a losing streak or manually intervene. The objective here is to formalize this decision.

## Model Signal Survival

Suppose we have training data $D$ from which we estimate a predictive distribution

$$p(y \mid x)$$

where $y \in \mathbb{R}^k$ denotes future returns and $x \in \mathbb{R}^q$ denotes features.

After validating the model out of sample, we deploy it in production.

Once live, however, we must allow for the possibility that the model is wrong. Consider two hypotheses:

- $H_1$: the model is valid and the signal exists;
- $H_0$: the model is invalid and expected returns are zero (this indicates that there is not bias in returns and features have zero correlation with targets).

The predictive distribution becomes a mixture:

$$ p(y_t \mid x_t, \mathcal F_{t-1}) = p(y_t \mid x_t, H_1)p(H_1 \mid \mathcal F_{t-1}) + p(y_t \mid x_t, H_0)p(H_0 \mid \mathcal F_{t-1}) $$

where $\mathcal F_{t-1}$ contains all observations up to time $t-1$.

Define

$$ \phi_t = p(H_1 \mid \mathcal F_t) $$

as the posterior probability that the model is still valid.

To make calculations possible, assume Gaussian distributions

$$ H_1: \quad y_t \sim N(\mu(x_t), C(x_t)) $$

and

$$ H_0: \quad y_t \sim N(0, C(x_t)) $$

Under this mixture, expected returns become

$$ \mathbb E\[y_t \mid x_t,\mathcal F_{t-1}] = \phi_{t-1}\mu(x_t) $$

For small returns, covariance changes due to model uncertainty are negligible, so the optimal bet is approximately

$$ w_t \approx \phi_{t-1} C(x_t)^{-1}\mu(x_t) $$

Thus exposure is simply scaled by posterior confidence. When confidence falls, exposure shrinks automatically.


## Evidence Accumulation

Posterior odds evolve according to Bayes’ rule:

$$ \frac{\phi_t}{1-\phi_t} = \frac{ p(y_t\mid x_t,H_1) }{ p(y_t\mid x_t,H_0) } \frac{\phi_{t-1}}{1-\phi_{t-1}} $$

Recursively,

$$ \frac{\phi_t}{1-\phi_t} = \frac{\phi_0}{1-\phi_0} \prod_{i=1}^{t} \frac{ p(y_i\mid x_i,H_1) }{ p(y_i\mid x_i,H_0) } $$

Define log odds

$$ L_t = \log \frac{\phi_t}{1-\phi_t}$$

Under the previous assumptions,

$$ L_t =\log\frac{\phi_0}{1-\phi_0}-\frac{1}{2}\sum_{i=1}^{t}\mu_i^\top C_i^{-1}\mu_i+\sum_{i=1}^{t}y_i^\top C_i^{-1}\mu_i$$


Define

$$s_t=y_t^\top C_t^{-1}\mu_t$$

as realized strategy return, and

$$m_t= \mu_t^\top C_t^{-1}\mu_t $$

as model-implied expected strategy return.

Then

$$ L_t  = L_0 + \sum_{i=1}^{t} s_i - \frac{1}{2} \sum_{i=1}^{t} m_i = L_0 + t \left(\bar{s} - \frac{1}{2} \mu_S \right) $$

where $\bar{s}$ is the mean strategy return up to $t$ and $\mu_S$ can seen as the expected model-implied strategy return (after many steps). Everything now depends only on the scalar sequence $s_t$.

Equivalently, instead of testing

$$ H_1: \quad y_t \sim N(\mu_t,C_t) $$

against

$$ H_0: \quad y_t \sim N(0,C_t) $$

we may test the scalar strategy statistics

$$ H_1: \quad s \sim N(m, m) $$

against

$$ H_0: \quad s \sim N(0, m) $$

These two hypothesis tests generate exactly the same likelihood ratio. A high-dimensional monitoring problem collapses into a one-dimensional sequential test.



## Stop criteria

Suppose we stop trading whenever posterior confidence falls below some threshold $z$:

$$ \phi_t < z $$

This is equivalent to

$$ L_t < \log\frac{z}{1-z} $$

Assume neutral prior odds:

$$ \phi_0=\frac12 $$

Then the stopping rule becomes

$$ \sum_{i=1}^{t} s_i < \log\frac{z}{1-z} + \frac12 \sum_{i=1}^{t} m_i $$

Define

$$ \bar s = \frac1n \sum_{i=1}^{n}s_i $$

as realized average strategy performance and

$$ \mu_S = \frac1n \sum_{i=1}^{n}m_i $$

as expected strategy performance.

The stopping rule becomes

$$ \bar s < \frac12 \mu_S + \frac1n \log\frac{z}{1-z} $$

As $n$ grows, the correction term vanishes:

$$ \bar s < \frac12\mu_S $$

In the long run, persistent performance below roughly half of expected edge becomes strong evidence that the signal is gone.

