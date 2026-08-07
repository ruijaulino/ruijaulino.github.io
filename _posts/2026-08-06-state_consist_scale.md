# State-consistent scaling

Suppose we have estimated the conditional distribution

$$
p(x|z)
$$



where $x$ are the future returns and $z$ represents the available information.

For a feature-dependent optimal allocation $w(z)$, define the strategy return

$$
s=w(z)^\top x
$$

Using the second non-central moment, the Sharpe ratio is

$$
SR=\frac{E[s]}{\sqrt{E[s^2]}}
$$

The optimal portfolio has the form

$$
w(z)=cM(z)^{-1}\mu(z)
$$

where

$$
\mu(z)=E[x|z] \qquad M(z)=E[xx^\top|z]
$$

and $c$ is a constant (the growth optimal portfolio has $c = 1$).

An interesting question is what happens if, after estimating the model, we decide to scale the portfolio differently according to the feature value,

$$
\tilde w(z)=\phi(z)M(z)^{-1}\mu(z)
$$

This includes clipping, confidence scaling, nonlinear transformations, etc - this may happen, for example, because the weights are too large and leverage is exceeded; a trivial example is to bet the full capital with the sign of the predictions.

Define

$$
q(z)=\mu(z)^\top M(z)^{-1}\mu(z)
$$

Then

$$
E[s]=E[\phi(z)q(z)]
$$

and

$$
E[s^2]=E[\phi(z)^2q(z)].
$$

Therefore

$$
SR(\phi) = \frac{E[\phi q]} {\sqrt{E[\phi^2q]}}
$$

Applying Cauchy-Schwarz,

$$
E[\phi q]^2 = E[\phi\sqrt q \sqrt q]^2 \le E[\phi^2q]E[q]
$$

Hence

$$
SR(\phi)^2 \le E[q]
$$


For the optimal strategy, corresponding to a constant scaling $\phi(z)=c$,

$$
SR_*^2=E[q]
$$

Therefore

$$
SR(\phi)^2\leq SR_*^2
$$

with equality if $\phi(z)=c$ for all states with $q(z)>0$.


Among all strategies obtained by rescaling the optimal conditional portfolio, the maximum Sharpe ratio is obtained only by multiplying every _state_ by the same constant. Any feature-dependent scaling decreases Sharpe.

The optimization determines not only the current optimal weight but also the relative scaling between all possible weights. Multiplying by a constant preserves these relative allocations, while any state-dependent transformation changes them.

This suggests that common operations such as clipping, nonlinear transformations or confidence-based leverage should generally reduce the Sharpe ratio unless they arise naturally from a different optimization problem (for example because of transaction costs, leverage constraints or estimation uncertainty).
