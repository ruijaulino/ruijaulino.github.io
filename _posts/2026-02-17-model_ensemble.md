# Model Averaging

Ensembling models is a common strategy to improve performance. In many situations, it is easier to produce a collection of simple models than to design a single highly accurate one. How does performance scale as we combine more models?

The common setup is to analyse the growth rate $G$:

$G(w) = \mathbf{E}\left\[ \log (1+w^Ty) \right\] \approx \mathbf{E}\left\[ w^Ty - \frac{1}{2}\left( w^Ty \right)^2 \right\]$

As shown previously, optimal weight is 

$w^* = M_{y\|x}^{-1} \mu_{y\|x}$ 

where $M$ is the second non central moment and $\mu$ is the first moment (both conditional on the feature value). 

Betting with $w^\*$ induces a strategy $s = w^{\*T}y$. Then

$\mathbf{E}\[s\] = \mathbf{E}\[s^2\] = 2 G(w^\*) = \mathbf{E}\_x \left\[ \mu_{y\|x}^{T}  M_{y\|x}^{-1} \mu\_{y\|x} \right\] := g$

So, as usual, studying the properties of $g$ allow us to infer about many aspects of the optimal strategy $s$. Let's make (reasonable) approximations to develop intuition.

### Explained variance interpretation

For unidimensional $y$ and features weakly correlated with the target, conditional variance is similar to variance. Then:

$g \approx \frac{\mathbf{E}\_x \left\[ \mu_{y\|x}^2 \right]}{\sigma^2} = \frac{\text{Var}\left( \mathbf{E} \left\[ y\|x \right] \right)}{\text{Var}(y)}$

This quantity can be identified as the explained variance ratio (correlation ratio, $\eta^2$). For a linear model this is equal to correlation squared $\rho^2$. It represents the fraction of $y$ variance that can be explained with $x$.

Also, if we consider $\mu_{y\|x} \sim N(0, v)$ then

$g = \frac{v}{\sigma^2} $

For example, in a linear model $y = bx + \epsilon$ with $x \sim N(0, q^2)$; also, $\rho(y,x) \approx \frac{bq}{\sigma}$ (for small $b$). With this:

$g = \frac{b^2 q^2}{\sigma^2} \approx \rho^2$

Again, growth rate is controlled by squared correlation.

Now, let's say that we average many simple models. How does $g$ behaves?

We can use the correlation as a proxy for the explained variance ratio (which is the quantity that we are interested on). With several models $h_i$ we can create a prediction for $y\|x$ by combining them with some weight scheme:

$s = \sum_i \beta_i h_i$

The correlation with the target

$\rho(s, y) = \frac{\text{Cov}(s, y)}{\sqrt{\text{Cov}(s, s) \text{Cov}(y, y)}} = \frac{\sum_i \beta_i \rho_i p_i}{\sqrt{\beta^T P \beta}}$

where $\rho_i$ is the correlation between signal $i$ and the target and $P$ is the covariance matrix between the signals $P_{ij} = \text{Cov}(s_i, s_j) = p_i p_j \pi_{ij}$

This expression shows that ensemble performance depends on:

- individual model quality, $\rho_i$
- covariance structure between models, $\pi_{ij}$
- weight allocation, $\beta_i$




### Effective parameters and scaling

Consider the quantity:

$Q = \frac{w^T \mu}{\sqrt{w^T C w}} $

with the matrix $C$ defined as (with $r_{ii} = 1$ like correlations and $q_i$ analogue to a standard deviation)

$C_{ij} = q_i q_j r_{ij}$

This expression is commonly associated with the sharpe ratio but here it is also similar to the expression for the correlation between the ensemble of models and the target under suitable choice of variables.


$Q = \frac{\sum_i w_i \mu_i}{\sqrt{\sum_i w_i^2 q_i^2 + \sum_i \sum_{j!=i} w_i w_j q_i q_j r_{ij}}} = \frac{\sum_i w_i qi \frac{\mu_i}{q_i}}{\sqrt{\sum_i w_i^2 q_i^2 + \sum_i \sum_{j!=i} w_i w_j q_i q_j r_{ij}}}$


To help with the calculations, $m_i = \frac{\mu_i}{q_i}$ and $\gamma_i = w_i q_i$. Then:

$Q = \frac{\sum_i \gamma_i m_i}{\sqrt{\sum_i \gamma_i^2 + \sum_i \sum_{j!=i} \gamma_i \gamma_j r_{ij}}}$


Define weighted average $\bar{m}$ as

$\bar{m} = \frac{\sum_i \gamma_i m_i}{\sum_i{\gamma_i}}$

weighted average $\bar{r}$

$\bar{r} = \frac{\sum_i \sum_{j!=i} \gamma_i \gamma_j r_{ij}}{\sum_i \sum_{j!=i}{\gamma_i \gamma_j}}$

Effective number

$k_{\text{eff}} = \frac{\left( \sum_i \gamma_i \right)^2}{\sum_i \gamma_i^2}$

Noting that $\left( \sum_i \gamma_i \right)^2 = \sum_i \gamma_i^2 + \sum_i \sum_{j!=i} \gamma_i \gamma_j$ we can write $Q$ in terms of _equivalent_ parameters

$Q = \bar{m}\sqrt{\frac{k_{\text{eff}}}{1+\bar{r}\left(k_{\text{eff}}+1\right)}}$

This expression is interesting and we can make several points about it:
- for $r_{ij} = 0$, $Q$ grows with the square root of the effective number of variables $k_{\text{eff}}$.
- if all $\gamma_i$ are equal then $k_{\text{eff}}$ equal the number of variables (all weight the same).
- the effect of _correlations_ $r_{ij}$ can be seen as a reduction in the number of effective variables as now $Q$ scales with the square root of a quantity smaller than $k_{\text{eff}}$.
- under the presence of _correlations_ $r_{ij}$, as we gather more variables, in the limit we can only scale up to $Q = \bar{m}\frac{1}{\sqrt{\bar{r}}}$



## Observation

Making the analogies $\beta_i \sim w_i$, $\rho_i p_i \sim \mu_i$, $P \sim C$ one can see that the performance of an ensemble model can be interpreted with the previous result.

As we add more independent signals the performance ($g$) increases: The ensemble growth rate increases with the effective number of independent predictors. Signals that are unprofitable on their own may make a average signal that beats costs as the expected values grows with signal diversification.

