# Notes on combining models

Consider a set of returns $y$ for which we have several models, each based on a different information set $x_i$. Model $i$ defines a conditional distribution $y \sim p_i(y \mid x_i)$. Optimal allocation to each one is

$$
\hat{w_i} = \frac{1}{k_i} w_i = \frac{1}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

where $k_i$ is just a scaling constant, $M$ is the second non central moment and $\mu$ is the first moment. This induces a strategy $s_i = \hat{w_i}^T y$ with properties

$$
E[s_i] = \frac{1}{k_i} E \left[ y^T M_{y \mid x_i}^{-1} \mu_{y \mid x_i} \right] =  \frac{1}{k_i} \left[\mu_{s}\right]_i
$$

$$
E[s_i s_j] = \frac{1}{k_i k_j} E \left[ \mu_{y \mid x_i}^T M_{y \mid x_i}^{-1} y y^T M_{y \mid x_j}^{-1} \mu_{y \mid x_j} \right] = \frac{1}{k_i k_j} \left[ M_s \right]_{ij}
$$

Define the diagonal matrix $V$ with $V_{ii} = \frac{1}{k_i}$. Rewritting, strategies $s$ has expected value vector $V\mu_s$ and second non central moment $V M_s V$. Maximum growth allocation between them is

$$
\phi = V^{-1} M_s^{-1} \mu_s
$$

which, propagated down to returns, means allocating with

$$
w = \sum_i  \frac{\phi_i}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i} = \sum_i  \frac{\left[M_s^{-1} \mu_s\right]_i k_i}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

Final allocation is a linear combination of the optimal allocations implied by the individual models, with coefficients determined by their joint expected performance and second moments. Importantly, the final allocation is invariant to the arbitrary scaling $k_i$ chosen for each individual strategy (the $k_i$ are retained because individual strategies are developed and evaluated at realistic scales. At the strategy-allocation level, the same scaling must be accounted for; it cancels in the the final allocation)


## Another view

Ideally, all information would be incorporated simultaneously into a single model $y\mid X$, where $X=(x_1, \cdots, x_m)$. In this case $w^* = \frac{1}{k} M_{y\mid X}^{-1} \mu_{y \mid X}$. This may be difficult, too much estimation errors and/or simply not practical. Instead, each model transforms its information $x_i$ into a decision

$$
\hat{w_i} = \frac{1}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i}
$$

Rather than attempting to estimate $w^*$ directly, restrict the combined decision to the linear span of these partial-information decisions:

$$
q = \sum_i u_i \hat{w_i}
$$

This induces a strategy $z = q^T y$ with properties

$$
E[z] = \sum_i u_i E[\hat{w_i}^T y] = \sum_i u_i E[s_i] = u^T V \mu_s
$$

$$
E[z^2] = \sum_i u_i u_j E[\hat{w_i}^T y y^T \hat{w_j}] = \sum_i u_i u_j E[s_i s_j] = u^T V M_s V u
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

This is the same allocation obtained by treating the individual model decisions as strategies and optimizing between them: mixing weights or mixing strategies should produce the same result. 

### Observations

We can see that

$$ 
[M_s]_{ii}=[\mu_s]_i
$$

since for each individual optimal strategy it's expected return equals the second non-central moment (as it should be if there was only one strategy). This makes $M_s^{-1}\mu_s$ easier to interpret. If $M_s$ is diagonal

$$ M_s^{-1}\mu_s=\mathbf 1 $$

So, in the absence of overlap, the optimal combined allocation is simply the sum of the allocations implied by each model. Each model has already determined its own optimal allocation, so no additional weighting is needed.

The coefficients $M_s^{-1}\mu_s$ can be interpreted as redundancy adjustments between individually optimal decisions: each model has already determined how much to bet from it's own information, while the combination step adjusts those bets only to account for their overlap.


## In practice

Practical use of models always present more challenges than simply following the _theory_. 

#### Computation of $k_i$

The objective of using a weight scaling $k_i$ is to make weights fall into a usable leverage value (provided that value does not make growth negative - which is not expected to happen with small returns). During model estimation one can compute some statistic of weight variation (standard deviation or a quantile) and use it to scale to unit leverage. Under diagonal $M_s$, the final allocation is

$$
\hat{w} = \sum_i k_i \frac{1}{k_i} M_{y \mid x_i}^{-1} \mu_{y \mid x_i} = \sum_i k_i \hat{w_i}
$$

to correct the scale in the overall $w$ we can use

$$
\hat{w} = \sum_i \frac{k_i}{k} \hat{w_i}
$$

with $k = \sum_j = k_j$. This is just multiplication by a constant, should not impact sharpe and relative importance of models/strategies should be preserved.

Now, it can happen that there are few models that dominate the computation of $k$; a fix here is to clip the $k_i$ to a quantile.


#### Computation of $\left[M_s^{-1} \mu_s\right]_i$

As discussed this quantity should be 1 (under diagonal $M_s$ which probably is the most practical case as data may not be synchronous and/or have different histories; recall that this is the performance of the _unormalized_ strategy). Since some models may work better than others (some may even not work) perhaps it can make sense to use this to account for that.

One can use a inner cycle of cross validation to check whether the model performance is positive or negative and use that to clip $\left[M_s^{-1} \mu_s\right]_i$ to zero or one.


#### Uncomparable models

The is another problem: if the models under consideration do not output proper measures of mean and covariance (think for example on the case where one invests proportional to inverse-volatily; what is the expected value and variance? probably we cannot mix this predictions with a model for those quantities). In general it can make more sense to use the second framework with $q = \sum_i u_i \hat{w_i}$: assuming diagonal $M_s$ (note the subscripts were dropped for ease of notation):

$$
u =  \frac{\mu_s}{\sigma_s^2} k \propto \frac{E[y^T M^{-1} \mu]}{E[\mu^T M^{-1} y y^T M^{-1} \mu]} \sqrt{E[\mu^T M^{-2} \mu]}
$$

where we assumed that a proper value for $k$ is related to the scale of the weights $\sigma_w = \sqrt{E[w^T w]} = \sqrt{E[\mu^T M^{-2} \mu]}$. Furthermore, assuming the model captures well the second moment, write as

$$
u \propto \frac{E[y^T M^{-1} \mu]}{\sqrt{E[\mu^T M^{-1} y y^T M^{-1} \mu]}} \sqrt{\frac{E[\mu^T M^{-2} \mu]}{E[\mu^T M^{-1} y y^T M^{-1} \mu]}} = \text{SR}_s \sqrt{\frac{E[\mu^T M^{-2} \mu]}{E[\mu^T M^{-1} y y^T M^{-1} \mu]}} \approx  \text{SR}_s \sqrt{\frac{E[\mu^T M^{-2} \mu]}{E[\mu^T M^{-1} \mu]}}
$$

The last term can be identified as a measure of strategy scale (easy to see at one dimension). This yields the approximation

$$
u \propto \text{SR}_s \frac{1}{\sigma_s}
$$


So, we can just compute $\left( \frac{\mu_s}{\sigma_s^2} \right)_i$ from a inner cross validation cycle (and this statistics are computed with the normalized weights!), clip for positive expected values and normalize. Even further, one can assume equal strategy sharpes (for the positive ones) and just go inverse strategy volatility.








