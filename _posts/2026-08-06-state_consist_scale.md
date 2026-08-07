# State-consistent scaling

Suppose we have estimated the conditional distribution

$$
p(x\mid z)
$$

where $x$ are future returns and $z$ represents the available information.

For a feature-dependent allocation $w(z)$, define the strategy return

$$
s=w(z)^\top x
$$

Using the second non-central moment, define the second-moment Sharpe as

$$
SR=\frac{E[s]}{\sqrt{E[s^2]}}
$$

The optimal allocation under this objective has the form

$$
w^*(z)=cM(z)^{-1}\mu(z)
$$

where

$$
\mu(z)=E[x\mid z] \qquad M(z)=E[xx^\top\mid z]
$$

and $c$ is an arbitrary global scaling constant (with $c=1$ for the growth-optimal solution).

The fact that $c$ is arbitrary can suggest that the scale of the portfolio is arbitrary. This is true globally, but not independently across states. Consider applying a different scaling depending on the observed features,

$$
\tilde w(z)=\phi(z)M(z)^{-1}\mu(z)
$$

This includes clipping, confidence scaling, nonlinear transformations, or simply limiting positions when the model produces large weights.

Define

$$
q(z)=\mu(z)^\top M(z)^{-1}\mu(z)
$$

Then

$$
E[s]=E[\phi q]
$$

and

$$
E[s^2]=E[\phi^2q]
$$

Therefore,

$$
SR(\phi)^2 = \frac{E[\phi q]^2}{E[\phi^2q]}
$$

By Cauchy–Schwarz,

$$
E[\phi q]^2 = E[\phi\sqrt q \sqrt q]^2 \leq E[\phi^2q]E[q]
$$

and hence

$$
SR(\phi)^2\leq E[q]
$$

For the optimal strategy, where $\phi(z)=c$,

$$
SR_*^2=E[q]
$$

Therefore,

$$
SR(\phi)^2\leq SR_*^2
$$

with equality only when

$$
\phi(z)=c
$$

for all states with $q(z)>0$.

Thus, although the optimal strategy is invariant to a **global** scaling constant, its scale is not arbitrary state by state. The optimization determines not only the portfolio direction at each state, but also the relative exposure across states.

Multiplying all state allocations by the same constant preserves the solution,

$$
w^* (z) \longrightarrow c w^* (z)
$$

while independently rescaling different states,

$$
w^* (z) \longrightarrow c(z )w^* (z)
$$

generally does not.

This gives a simple interpretation: **the relative leverage assigned to different future states is itself part of the optimal strategy.** Post-processing operations such as clipping, nonlinear transformations, or confidence-based scaling destroy this state-consistent scaling and reduce population Sharpe.




## The cost of inconsistent scaling

The loss from state-dependent scaling can also be quantified. Relative to the optimal strategy,

$$
\frac{SR(\phi)^2}{SR_*^2} = \frac{E[\phi q]^2} {E[\phi^2q]E[q]} \leq 1
$$

Define the $q$-weighted expectation

$$
E_q[f] = \frac{E[qf]}{E[q]}
$$

This simply gives more importance to states with large $q(z)$, i.e. states that contribute more to the optimal strategy.

Then

$$
\frac{SR(\phi)^2}{SR_*^2} = \frac{E_q[\phi]^2}{E_q[\phi^2]}
$$

Using

$$
E_q[\phi^2] = E_q[\phi]^2+\text{Var}_q(\phi)
$$

we obtain

$$
\frac{SR(\phi)^2}{SR_*^2} = \frac{1} {1+\dfrac{\text{Var}_q(\phi)}{E_q[\phi]^2}}
$$

Thus, the loss depends only on how much the scaling $\phi(z)$ varies across states, weighted by their importance $q(z)$.

If $\phi(z)$ is constant, then

$$
\text{Var}_q(\phi)=0
$$

and there is no loss. The more the relative scaling changes across important states, the larger the Sharpe loss.

This does not imply that state-dependent transformations should never be used. Leverage constraints, transaction costs, estimation uncertainty, or model misspecification can justify them, but they then correspond to a different optimization problem rather than an improvement of the original optimum.

For example, a practical approach is to use the training data to estimate the range over which the optimal weights vary and use this to choose an appropriate global scaling constant. Extreme positions can then be clipped when necessary. If clipping affects only a small part of the $q$-weighted distribution, the departure from optimality may be small, and the expression above provides a direct way to quantify it.



