# Factor exposures

Consider a conditional allocation rule $w(x)$, where $x$ is a feature vector and $y$ is the vector of asset returns.

We maximize expected log-growth

$$
G = E\left[\log(1+w(x)^\top y)\right]
$$

Using the second-order approximation


$$
\log(1+z)\approx z-\frac12 z^2
$$


we obtain

$$
G \approx E\left[ w(x)^\top y - \frac12 (w(x)^\top y)^2 \right]
$$

Conditioning on $x$,

$$
G =  E_x\left[ w(x)^\top \mu_{y|x} - \frac12 w(x)^\top M_{y|x} w(x) \right]
$$

where

$$
\mu_{y|x}=E[y\mid x]
$$

and

$$
M_{y|x}=E[yy^\top\mid x]
$$

Since

$$
M_{y|x} = C_{y|x} + \mu_{y|x}\mu_{y|x}^\top
$$

and conditional mean returns are typically small relative to covariance in financial applications, we approximate

$$
M_{y|x}\approx C_{y|x}
$$

The unconstrained optimum is therefore

$$
w^*(x) = C_{y|x}^{-1}\mu_{y|x}
$$


## Decomposition

Assume asset returns admit a linear factor representation

$$
y = B(z)f+\epsilon,
$$

where

* $B(z)$ is a factor loading matrix depending on a state variable $z$,
* $f$ is a vector of factor returns,
* $\epsilon$ is residual noise.

We assume factors include their premia, so they are not centered.

The strategy return is

$$
s = w(x)^\top y = w(x)^\top B(z)f + w(x)^\top\epsilon
$$

The effective factor exposure is therefore

$$
\beta(x,z) = B(z)^\top w(x)
$$

which may vary both because the portfolio changes with $x$ and because the factor loadings evolve with $z$.

This decomposition separates strategy performance into

* systematic factor contribution $w(x)^\top B(z)f$

* residual contribution $w(x)^\top\epsilon$


## Neutralization

### Instantaneous neutralization

The strongest definition requires the strategy to be factor-neutral at every decision time.

We require

$$
w(x)^\top B(z)f=0
$$

for every realization of the factors.

A sufficient condition is

$$
B(z)^\top w(x)=0
$$

The current loading matrix $B(z)$ is known when the portfolio is constructed; the constrained optimization problem is maximization of

$$
E_{x} \left[ w^\top\mu_{y|x} - \frac12 w^\top C_{y|x}w \right]
$$

subject to

$$
B(z)^\top w(x)=0
$$

The solution is

$$
w(x) = C_{y|x}^{-1} \left( \mu_{y|x} - B(z)\lambda(x,z) \right)
$$

where

$$
\lambda(x,z) = \left(B(z)^\top C_{y|x}^{-1} B(z) \right)^{-1} B(z)^\top C_{y|x}^{-1} \mu_{y|x}
$$

Instantaneous neutrality projects the unconstrained optimal portfolio onto the subspace orthogonal to the current factor loadings.


For example, if the only asset is the market itself (we have some feature to predict the SP500, trade via ES futures and want to be market (SP500) neutral - this situation may be common trading few strategies/signals which will be subjected to this market risk), then

$$
B=1
$$

implies

$$
w(x)=0
$$

for every $x$. The strategy cannot trade.


### Distributional neutralization

Instantaneous neutrality removes all factor exposure, including potentially profitable timing. A weaker approach is to impose neutrality only in expectation.

Consider expected factor contribution to strategy

$$
E[w(x)^\top B(z)f]
$$

Using the identity

$$
E[a^\top b] = E[a]^\top E[b] + \text{tr}(\text{Cov}(a,b))
$$

we obtain

$$
E[w(x)^\top B(z)f] = E[B(z)^\top w(x)]^\top E[f] + \text{tr} \left( \text{Cov} (B(z)^\top w(x),f) \right)
$$

Applying the law of total covariance,

$$
\text{Cov}(B^\top w,f) = E\left[ \text{Cov}(B^\top w,f\mid x) \right] + \text{Cov} \left( E[B^\top w\mid x], E[f\mid x] \right)
$$

so expected factor profits consist of three components:

* average systematic exposure $E[B^\top w]^\top E[f]$

* factor timing $\text{Cov} ( E[B^\top w\mid x], E[f\mid x] )$

* beta dynamics $E[ \text{Cov} (B^\top w,f\mid x) ] $

The first component arises from persistent factor exposure.

The second arises because the strategy varies its expected factor exposure according to information predicting factor returns.

The third arises because the factor loadings themselves evolve over time and may co-move with factor returns independently of the trading rule.

This motivates weaker neutrality definitions.


#### Static neutralization

Impose zero average factor exposure:

$$
E[B(z)^\top w(x)] = 0
$$

This means the strategy is neutral _on average_, even if temporarily exposed.

The constrained optimum becomes

$$
w(x) = C_{y|x}^{-1} \left( \mu_{y|x} - \Gamma(x)\lambda \right)
$$

where

$$
\Gamma(x) = E[B(z)\mid x]
$$

This arises from the possibility that $x$ (or a subset of it) is in $z$. and

$$
\lambda = \left( E[ \Gamma(x)^\top C_{y|x}^{-1} \Gamma(x) ] \right)^{-1} E[ \Gamma(x)^\top C_{y|x}^{-1} \mu_{y|x}]
$$

Unlike instantaneous neutrality, the multiplier is constant across feature realizations.

After static neutralization,

$$
E[s] = 
\text{tr}
\left(
\text{Cov}
(B^\top w,f)
\right)
+
E[w^\top\epsilon]
$$

The strategy may therefore still profit from

1. factor timing;
2. beta dynamics;
3. residual predictability.

Only systematic carry arising from average factor exposure is removed.


#### Universal static neutralization
For a factor model satisfying

$$ E[B(z)\mid x] = E[B(z)] $$

we have that:

$$
E[B(z)^\top w(x)] = E[B(z)]^\top E[w(x)]
$$

so imposing

$$
E[w(x)] = 0
$$

eliminates average exposure to every factor model whose loadings are _unpredictable from the strategy information_. With that, optimal constrained solution becomes

$$ 
w(x) =  C_{y|x}^{-1} (\mu_{y|x}-\bar\mu)
$$

where

$$
\bar\mu = E[C_{y|x}^{-1}]^{-1} E[C_{y|x}^{-1}\mu_{y|x}]
$$

A simpler feasible alternative is naive demeaning,

$$
w(x)
\leftarrow
w(x)-E[w(x)]
$$

This satisfies the constraint exactly but is generally suboptimal because it ignores heteroskedastic covariance. If the covariance matrix is constant, demeaning coincides with the optimal solution.


## Complete neutralization

Static neutralization still allows expected profits from factor timing and beta dynamics. To eliminate all expected factor PnL, impose

$$
E[w(x)^\top B(z)f] = 0
$$

Using total expectation,

$$
E[w(x)^\top B(z)f] = E_x \left[ w(x)^\top q(x) \right]
$$

where

$$
q(x) = E[B(z)f\mid x]
$$

The constraint becomes

$$
E_x[w(x)^\top q(x)] = 0
$$

The optimal solution is

$$
w(x) = C_{y|x}^{-1}
(\mu_{y|x}-\lambda q(x))
$$

where

$$
\lambda = \frac{E[\mu_{y|x}^\top C_{y|x}^{-1} q(x) ] }{E[q(x)^\top C_{y|x}^{-1} q(x) ] }
 $$

This removes all expected profits attributable to the factor component, regardless of whether they arise from average exposure, factor timing, or time-varying factor loadings. A strategy may still exhibit realized correlation with the factors while remaining neutral under this definition.







