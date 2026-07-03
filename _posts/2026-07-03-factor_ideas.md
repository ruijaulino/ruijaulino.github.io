# Some thoughts on factor models

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
y = Bf+\epsilon
$$

where:

* $B$ is a factor loading matrix,
* $f$ is a vector of factor returns,
* $\epsilon$ is residual noise.

We assume factors include their premia, so they are not centered.

The strategy return becomes

$$
s = w(x)^\top y = w(x)^\top Bf + w(x)^\top\epsilon
$$

This decomposition separates strategy performance into:

* systematic factor contribution: $w(x)^\top Bf$

* residual contribution: $w(x)^\top \epsilon$

## Neutralization

### Instantaneous neutralization

The strongest definition requires the strategy to be factor-neutral at every decision time.

We require

$$
w(x)^\top Bf=0
$$

for all possible factor realizations $f$.

A sufficient condition is

$$
B^\top w(x)=0
$$

This imposes zero factor exposure pointwise.

The constrained optimization problem is maximization of

$$
E_x \left[ w^\top\mu_{y|x} - \frac12 w^\top C_{y|x}w \right]
$$

subject to

$$
B^\top w(x)=0
$$

The solution is

$$
w(x) = C_{y|x}^{-1} \left( \mu_{y|x} - B\lambda(x) \right)
$$

with

$$
\lambda(x) = (B^\top C_{y|x}^{-1} B)^{-1} B^\top C_{y|x}^{-1}\mu_{y|x}
$$

$\lambda(x)$ is a vector of Lagrange multipliers with adequate dimension. Instantaneous neutrality corresponds to projecting the unconstrained optimal bet onto the subspace orthogonal to the factors.

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

Instantaneous neutrality eliminates all factor exposure, including potentially profitable timing - a weaker approach is to impose neutrality only in expectation.

Consider expected factor contribution to strategy

$$
E[w(x)^\top Bf]
$$

Expanding,

$$
E[w(x)^\top Bf] = E[B^\top w(x)]^\top E[f] + \text{tr}\left(\text{Cov}\left(B^\top w(x),f\right)\right)
$$

This separates:

* average systematic exposure
* exposure arising from factor timing

This motivates weaker neutrality definitions.


#### Static neutralization

Impose zero average factor exposure:

$$
E[B^\top w(x)] = 0
$$

This means the strategy is neutral _on average_, even if temporarily exposed.

The constrained optimum is

$$
w(x) = C_{y|x}^{-1} \left( \mu_{y|x} - B\lambda \right)
$$

where the Lagrange multipliers are global:

$$
\lambda=  E_x \left[ B^\top C_{y|x}^{-1}B \right] ^{-1} E_x \left[ B^\top C_{y|x}^{-1}\mu_{y|x} \right]
$$

Unlike instantaneous neutrality, the multiplier is constant across $x$.

After static neutralization

$$
E[s] =  \text{Cov}(B^\top w(x),f) + E[w(x)^\top\epsilon]
$$

Therefore the strategy may still profit from:

1. factor timing
2. residual predictability

Only systematic carry from average factor exposure is removed.

##### Universal static neutralization

A particularly interesting special case is

$$
B=I
$$

Then the constraint becomes

$$
E[w(x)]=0
$$

Since

$$
E[B^\top w(x)] = B^\top E[w(x)]
$$

this removes average exposure to any (fixed) linear factor model - neutralizing a _specific_ factor model yields a less restrictive condition.

The optimal solution becomes

$$
w(x) = C_{y|x}^{-1} \left( \mu_{y|x} - \bar{\mu} \right)
$$

where

$$
\bar{\mu} = E_x \left[ C_{y|x}^{-1} \right]^{-1} E_x \left[ C_{y|x}^{-1}\mu_{y|x} \right]
$$

A simpler feasible alternative is naive demeaning:

$$w(x) \leftarrow w(x)-E\left[w(x)\right]$$


This satisfies the constraint but is generally suboptimal because it ignores a changing covariance (if present in the model).


### Complete neutralization

Static neutralization still allows profits from factor timing. If we want to eliminate all expected factor PnL, we require

$$
E[w(x)^\top Bf]=0.
$$

Using total expectation

$$
E[w(x)^\top Bf] = E_x \left[ w(x)^\top B,E[f\mid x] \right]
$$

Define predictable factor returns (with features $x$)

$$
m_f(x)=E[f\mid x]
$$

The constraint becomes

$$
E_x[w(x)^\top Bm_f(x)] = 0
$$

We optimize growth subject to this constraint.

The solution is

$$
w(x) = C_{y|x}^{-1} \left( \mu_{y|x} - \lambda Bm_f(x) \right)
$$

where

$$
\lambda = \frac{E_x \left[ \mu_{y|x}^\top C_{y|x}^{-1}Bm_f(x) \right] }{ E_x \left[ m_f(x)^\top B^\top C_{y|x}^{-1} Bm_f(x) \right] }
$$

This removes the component of the strategy expected to earn money through predictable factor returns and does not require zero exposure. A strategy may remain correlated with factors and still be neutral under this definition.


#### Relation to static neutrality

If features do not predict factor variation, then

$$
m_f(x)=\mu_f
$$

The constraint reduces to

$$
\mu_f^\top E[B^\top w(x)] = 0
$$

This is weaker than static neutralization

$$
E[B^\top w(x)] = 0
$$

They coincide only in special cases (for example, a single-factor model).
