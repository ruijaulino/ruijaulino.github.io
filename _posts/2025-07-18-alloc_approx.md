# On approximations to allocation

Estimation of parameters and numerical problems make the optimal allocation problem quite difficult to solve. For practical applications it is logical to make approximations to have more robust solutions (ignore correlations for example). The objective here is to show some approximations that can be made in a world where correlations are small with the objective to have both usefull and intuitive formulas. 

The setup is one where we have some features to make predictions on future returns. As discussed in other posts, when presented with new information, the optimal allocation is

$w = C_{y\|x}^{-1} \mu_{y\|x}$

Consider a joint gaussian distribution between assets and features $y, x$. In this case, the conditional distribution is

$p(y\|x) = N\left(\mu_{y\|x}, C_{y\|x}\right) = N\left( \mu_y + C_{yx}C_{xx}^{-1}(x-\mu_x), C_{yy}- C_{yx}C_{xx}^{-1}C_{xy} \right)$



#### Preliminary observations

Let $C$ be a covariance matrix. We can write $C = SRS$ where $S$ is a diagonal matrix with scales and $R$ the correlation matrix. In most cases we need to do some form of regularization due to finite sampling and large dimensions. A way to do this is to consider an approximate correlation matrix

$\hat{R} = I + \epsilon(R-I)$.

where $\epsilon$ is a small value. This will regularize correlations to zero. For small $\epsilon$:

$\hat{R}^{-1} \approx I - \epsilon(R-I)$

If correlations are small (even without regularization) we can use the same idea to write $R^{-1} \approx I - (R-I)$.

We can say that this is a first order/linear approximation to the inverse correlation matrix.


### No predictors

In the case of no features we have a classical asset allocation problem:

$w = C_{yy}^{-1} \mu_y$

There is considerable literature on how to solve this problem with many types of constraints but in practive what happens is that the estimated parameters are not stable enough to yield a robust solution and so people tend to ignore correlations and just go with _inverse volatility_. What is the second order of complexity model that we can do from there?

Let $C_{yy} = S_{y} R_{yy} S_{y}$ with $S_{y}$ a diagonal matrix with scales. Easily we see $C_{yy}^{-1} = S_{y}^{-1} R_{yy}^{-1} S_{y}^{-1}$

Assuming small correlation between the assets (we will have to regularize anyway and so, this will almost surelly be the case), we can approximate:

$w \approx S_y^{-1}\left( I - R_{yy} \right) S_y^{-1}\mu_y$

Another to write this is:

$w_i \approx \frac{1}{\sigma_i} \left(s_i - \sum_{j \neq i} \rho_{ij}s_j \right)$

with $s_i = \frac{\mu_i}{\sigma_i}$ the sharpe ratio of asset $i$. This can be interpreted as a the optimal individual solution penalized with the sharpe-weighted correlation with the other assets; if an asset has positive correlation with the others then it's weight get smaller. By itself, this result can be seen as a numerically stable way to solve allocation between assets.

#### Equal Sharpes

To make a connection with risk parity solutions, let us consider all sharpe ratios $s$ equal. Then:

$w_i \propto \frac{1-\sum_{j \neq i}\rho_{ij}}{\sigma_i}$

This is similar to risk parity but we make a small correction due to correlation. In practice, we can set up a regularization on correlation such that weights do not go against expected values (I will not elaborate on this as it is rather simple).





### With predictors

Considering the predictors we need to look at each term an how it simplifies. $C_{y\|x}$ can be written as:

$C_{y\|x} = S_y \left( R_{yy} - R_{yx}R_{xx}^{-1}R_{xy} \right) S_y$

For small correlations and keep terms in $\mathcal{O}(\rho)$ we have that

$C_{y\|x} \approx S_y R_{yy} S_y = C_{yy}$

which makes sense: for small correlations the conditional correlation is similar to the unconditional one.

The conditional mean:

$\mu_{y\|x} = \mu_y + C_{yx}C_{xx}^{-1}(x-\mu_x) = \mu_y + S_y R_{yx}R_{xx}^{-1} S_{x}^{-1}(x-\mu_x)$

Then the optimal weight

$w \approx C_{yy}^{-1}\left( \mu_y + S_y R_{yx}R_{xx}^{-1} S_{x}^{-1}(x-\mu_x) \right)$

Using again the small correlation expansion

$w \approx S_y^{-1} \left[ (I-R_{yy}) S_y^{-1}\mu_y + (I-R_{yy})R_{yx}(I-R_{xx})S_x^{-1}(x-\mu_x) \right]$

Keeping terms on first order in correlations

$w \approx S_y^{-1} \left[  (I-R_{yy}) S_y^{-1}\mu_y + R_{yx}S_x^{-1}(x-\mu_x)\right]$

We can view this as two terms: the first is an allocation based on how assets move together and a second based on the signals from features. The term $R_{yx}S_x^{-1}(x-\mu_x)$ is a normalized feature value weighted by the correlation with the target.

Taking into account the magnitude of the values that we expect to encounter, we can say first that $S_x^{-1}(x-\mu_x) \sim \mathcal{O}(1)$ as it is a normalized variable: then $R_{yx}S_x^{-1}(x-\mu_x) \sim \mathcal{O}(\rho)$. The term $S_y^{-1}\mu_y$ is small as it relates to how the mean is compared with the scale: for a annual sharpe ratio of 1 (for the asset; observed values for reference assets is smaller) this term is of the order of $1^{-2}$ on a daily basis (in comparisson with $S_x^{-1}(x-\mu_x)$ that should be of order $1$; also, it is expected that correlations are of order $1^{-2}$ as well); with this, $(I-R_{yy}) S_y^{-1}\mu_y = S_y^{-1}\mu_y-R_{yy}S_y^{-1}\mu_y$ can be said to be of order $1^{-2} + \rho 1^{-2} \sim 1^{-2} + \rho^2$. Given that correlations are small, terms from features dominate the cross correlation between assets.

$w \approx S_y^{-1} \left[S_y^{-1}\mu_y + R_{yx}S_x^{-1}(x-\mu_x)\right]$

### Observations

We can view this result as if we use features to determine an allocation then the allocation between the assets does not influence the final allocation: it is like allocating individually to each resulting strategy. This makes sense as the strategies are like new assets and, through the features, they are less correlated between them as the original targets are. So, a usefull approximation is, when allocating between strategies we can ignore the correlations between them (in first order).



## Mixing Signals

Starting from the previous results and considering a single target variable with many (weakly correlated) features predicting it:

$w \approx \frac{1}{\sigma^2} \left[ \sum_j \rho_{j} \frac{x_j-\mu_{x_j}}{\sigma_{x_j}} \right]$


This can be interpreted as a linear combination of normalized signals. For example, consider two signals one representing a trend and another the carry: if we normalize both we can just make a convex combination of them to get an optimal allocation (if we believe that the correlation they have with the target is the same). This procedure is used many times but here we can give a justification on the why and under which conditions it makes sense.



