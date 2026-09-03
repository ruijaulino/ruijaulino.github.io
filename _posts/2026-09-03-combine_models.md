## Notes on combining models

Consider a set of returns $y$ for which we have several models, each based on a different information set $x_i$. Model $i$ defines a conditional distribution $y \sim p_i(y \mid x_i)$. Optimal allocation to each one is

$$
w_i = \frac{1}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

where $k_i$ is just a scaling constant, $M$ is the second non central moment and $\mu$ is the first moment. This induces a strategy $s_i = w_i^T y$ with properties

$$
E[s_i] = \frac{1}{k_i} E \left[ y^T M_{y \mid x_i}^{-1} \mu_{y \mid x_i} \right] =  \frac{1}{k_i} \left[\mu_{s}\right]_i
$$

$$
E[s_i s_j] = \frac{1}{k_i k_j} E \left[ \mu_{y \mid x_i}^T M_{y \mid x_i}^{-1} y y^T M_{y \mid x_j}^{-1} \mu_{y \mid x_j} \right] = \frac{1}{k_i k_j} \left[ M_s \right]_{ij}
$$

Define the diagonal matrix $V$ with $V_{ii} = \frac{1}{k_i}$. The set of strategies $s$ has expected value $V\mu_s$ and second non central moment $V M_s V$. Maximum growth allocation between them is

$$
\phi = V^{-1} M_s^{-1} \mu_s
$$

which, transalated down to returns, means allocating with

$$
w = \sum_i  \frac{\phi_i}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i} = \sum_i  \frac{\left[M_s^{-1} \mu_s\right]_i k_i}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

Final allocation is a linear combination of the optimal allocations implied by the individual models, with coefficients determined by their joint expected performance and second moments. Importantly, the final allocation is invariant to the arbitrary scaling $k_i$ chosen for each individual strategy (the $k_i$ are retained because individual strategies are developed and evaluated at realistic scales. At the strategy-allocation level, the same scaling must therefore be accounted for; it cancels exactly from the final unconstrained allocation)


### Another view

Ideally, all information would be incorporated simultaneously into a single model $y\mid X$, where $X=(x_1, \cdots, x_m)$. In this case $w^* = \frac{1}{k} M_{y\mid X}^{-1} \mu_{y \mid X}$. This may be difficult, too much estimation errors and/or simply not practical. Instead, each model transforms its information $x_i$ into a decision

$$
w_i = \frac{1}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

Rather than attempting to estimate $w^*$ directly, restrict the combined decision to the linear span of these partial-information decisions:

$$
q = \sum_i u_i w_i
$$

This induces a strategy $z = q^T y$ with properties

$$
E[z] = \sum_i u_i E[w_i^T y] = \sum_i u_i E[s_i] = u^T V \mu_s
$$

$$
E[z^2] = \sum_i u_i u_j E[w_i^T y y^T w_j] = \sum_i u_i u_j E[s_i s_j] = u^T V M_s V u
$$

where the last equalities used the previous definitions. This strategy has growth rate

$$
G = u^T V \mu_s - \frac12 u^T V M_s V u
$$

which is maximized when

$$
u^* = V^{-1} M_s^{-1} \mu_s
$$

Replacing

$$
q = \sum_i  \frac{\left[M_s^{-1} \mu_s\right]_i k_i}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

This is the same allocation obtained by treating the individual model decisions as strategies and optimizing between them.
