# Betting with models II

Consider a sequence of returns $x_1,x_2,\cdots,x_n$. If at each period we rebalance with a weight $w_i$ the capital grows as:

$S_n = S_0 \exp \left( n \frac{1}{n} \sum_i \log \left( 1+w_i^Tx_i\right) \right) = S_0 \exp \left(nG\right)$

with $G=\frac{1}{n} \sum_i \log \left( 1+w_i^Tx_i\right)$

Since $x$ are returns let us assume that they are small, i.e, $\|x\|\ll 1$. Then, $\log\left(1+w^Tx\right) \approx w^Tx - \frac{1}{2}\left(w^Tx\right)^2$.

With this, $G \rightarrow \mathbb{E}\left[ \log\left(1+w^Tx\right) \right] \approx \mathbb{E}\left[w^Tx- \frac{1}{2}\left(w^Tx\right)^2\right]$

where the expectation is with respect to both $x$ and $w$ (or wrt to the distribution of the dot product $w^Tx$)


Consider the base case where the distribution of $x$ is the same for all observations $x_i$; as will be clear later, $w$ is a constant vector depending of the properties of the distribution (this is the typical case and, for now, we can just look at it as optimization of $G$ wrt $w$).

$G = \mathbb{E}\left[w^Tx- \frac{1}{2}\left(w^Tx\right)^2\right] = w^T\mu - \frac{1}{2}w^T\Sigma w$

with $\mu$ being the location vector and $\Sigma$ the second non central moment matrix (which we designate by the usual covariance letter because, for the expansion of the logarithm to be valid, we are assuming $\mu$ is small compared to the unit and so the covariance and the second non central moment are similar).

Maximization of $G$ (optimal growth) can be achieved by solving $\frac{\partial G}{\partial w} = \mu-\Sigma w = 0$. This gives $w=\Sigma^{-1}\mu$.

A more realistic situation is one where the returns are dependent on some other variables and, hopefully, we can find a suitable model for it. How should the weights be calculated? 

### Calculus of Variations

Consider a functional as a function that maps a function into a number; an example of this is an integral - we have some function $y(x)$ and by the action of the integral it returns a number:

$I\left(y(x)\right) = \int F\left[y(x),x\right] \text{d}x$

Where $F$ is some function of the function $y(x)$ and $x$.

Suppose that we perturb $y(x)$ with a generic function; also we need this perturbation to be arbitrarily small. For this effect we can consider the change $y(x) \rightarrow y(x)+\epsilon \eta(x)$. The value of the function at the perturbation can be written (using expansion near $\epsilon=0$):

$I[y(x)+\epsilon \eta(x)] = \int F[y(x)+\epsilon \eta(x),x]\text{d}x \approx \int F[y(x)]\text{d}x + \epsilon \int \frac{\partial F}{\partial y}\eta(x) \text{d}x + O(\epsilon^2)$ 

Rearranging we can state that a variation in $I$ for a small perturbation is

$\delta I = \frac{I[y(x)+\epsilon \eta(x),x]-I[y(x),x]}{\epsilon} = \int \frac{\partial F}{\partial y}\eta(x) \text{d}x$

If we are optimizing $I$ with respect to the function $y(x)$ we want the value of $I$ to be stationary with respect to perturbations: $\delta I = 0$ for any generic perturbation $\eta(x)$. This translates into finding the function $y(x)$ where $\frac{\partial F}{\partial y}=0$ (I will not elaborate on the Euler-Lagrange equation as it is not necessary for this problem).

#### Note
Useful for the next calculations, consider the functional:

$I[y(x),z] = \int \int F\left[y(x),z,x\right] \text{d}x \text{d}z$

and we still want to optimize with respect to the function $y(x)$. If we follow the previous calculation we can write

$I[y(x)+\epsilon \eta(x),z] = \int \int F[y(x)+\epsilon \eta(x),x,z]\text{d}x \text{d}z \approx \int \int F[y(x),x,z]\text{d}x \text{d}z + \epsilon \int \left( \int\frac{\partial F}{\partial y} \text{d}z \right) \eta(x) \text{d}x $ 

The stationarity condition with respect to a generic $\eta(x)$ becomes

$\int\frac{\partial F}{\partial y} \text{d}z = 0$





## Using a model for returns

Consider the case where the return vector $x$ is dependent on some features $z$ (this can include past values of $x$ or any other exogenous variable). The joint distribution $p(x,z)$ exists and we can define the conditional $p(x\|z)$.

Again, the capital after $n$ periods is

$S_n = S_0 \exp \left( n \frac{1}{n} \sum_i \log \left( 1+w_i^Tx_i\right) \right) = S_0 \exp \left(nG\right)$

with $G=\frac{1}{n} \sum_i \log \left( 1+w_i^Tx_i\right) \rightarrow \mathbb{E}\left[ \log\left(1+w^Tx\right) \right] \approx \mathbb{E}\left[w^Tx- \frac{1}{2}\left(w^Tx\right)^2\right]$

The weights at some instant $i$ must be a function of the features $z_i$, $w_i=w(z_i)$ ($z$ is the only information that determine $x$ and so the weights must be a function of it). The growth rate becomes

$G = \int \int \left(w^Tx- \frac{1}{2}\left(w^Tx\right)^2\right) p(x,w) \text{d}x \text{d}w = \int \int \left(w(z)^Tx- \frac{1}{2}\left(w(z)^Tx\right)^2\right) p(x\|z)p(z) \text{d}x \text{d}z$

where the second step is true because of conservation of probability (with some abuse of notation, $w$ is a function of $z$ and so, $p(x,w)\text{d}w\text{d}x = p(x,z)\text{d}z\text{d}x \rightarrow p(x\|w)p(w)\text{d}w = p(x\|z)p(z)\text{d}z$. 


Optimization of $G$ is a similar problem to the one of calculus of variations. To find the functional form of the weights that maximize $G$ we can resort to the previous formula from calculus of variations:

$\int \frac{\partial F}{\partial w} \text{d}x = 0$

with $F=\left(w^Tx- \frac{1}{2}\left(w^Tx\right)^2\right) p(x\|z)p(z)$. Now the derivative is $\frac{\partial F}{\partial w} = \left( x - xx^T w \right) p(x\|z)p(z)$. Replacing in the integral (drop the term $p(z)$ ):

$\int \frac{\partial F}{\partial w} \text{d}x = 0 \rightarrow \int x p(x\|z) \text{d}x = \int xx^T p(x\|z) \text{d}x \cdot w$

where we can identify the term on left hand side as the expected value of $x$ given $z$, $\mu_{x\|z}$, and the first term on the right as the conditional second non central moment (which, again, from the approximation, should be similar to the covariance), $\Sigma_{x\|z}$. With this:

$w = \Sigma_{x\|z}^{-1}\mu_{x\|z}$

This result, derived from first principles, makes sense: the optimal weight vector is just the one derived from the conditional distribution. When we have a model that specifies the distribution of the returns given some features (which is the common case), optimal growth is achieved by making the bet with the conditional distribution.


### Optimization of Sharpe

A more sensible optimization to solve in a investment problem is to achieve a better risk-adjusted return

$\text{SR}(w) = \frac{\mathbb{E}\left[w^T x\right]}{\sqrt{\mathbb{E}\left[\left(w^T x\right)^2\right]}}$

we can note that, $\text{SR}(w)=\text{SR}(kw)$, i.e, multiplication by a constant does not change the result; we can consider the maximization constrained to the denominator equal to unit (any other solution can be obtained by multiplication by a constant). Using Lagrange multipliers we can consider maximizing the following functional as the same as optimizing sharpe:

$Q(w(z),z,x) = \mathbb{E}\left[w^T x\right] - 2\lambda \left(\sqrt{\mathbb{E}\left[\left(w^T x\right)^2\right]} -1 \right)$

For ease of notation:

$\mathbb{E}\left[w^T x\right] = \int \int w^Tx p(x,z) \text{d}x \text{d}z = \int \int f(w(z),x,z) \text{d}x \text{d}z = F$

$\mathbb{E}\left[\left( w^T x\right)^2\right] = \int \int \left(w^Tx\right)^2 p(x,z) \text{d}x \text{d}z = \int \int g(w(z),x,z) \text{d}x \text{d}z = G$

where, like previously, we use conservation of probability to interchange $p(x, w)$ and $p(x, z)$. With this new notation, let us perturb $w(z)$ with $w(z)+\epsilon \eta(z)$:

$Q(w(z)+\epsilon \eta(z),z,x) = F+\epsilon F' - 2\lambda \left(\sqrt{G+\epsilon G'} -1\right)$

with $F' = \int \int \frac{\partial f}{\partial w} \eta(x) \text{d}x \text{d}z$ and $G' = \int \int \frac{\partial g}{\partial w} \eta(x) \text{d}x \text{d}z$ (in a similar way to what was done before). Since $\epsilon$ is small, we can approximate $\sqrt{G+\epsilon G'} \approx \sqrt{G} + \frac{1}{2}\epsilon\frac{G'}{\sqrt{G}}$. Putting all terms together we can write:

$Q(w(z)+\epsilon \eta(z),z,x) = Q(w(z),z,x)+\epsilon\left( F' -\lambda \frac{G'}{\sqrt{G}} \right)$

From the constraint $\sqrt{G}=1$ the perturbation becomes

$\delta Q = \int \left(\int \frac{\partial f}{\partial w} -\lambda\frac{\partial g}{\partial w}  \text{d}x \right) \eta(z) \text{d}z$

To make it stationary we need the condition:

$\int \frac{\partial f}{\partial w} -\lambda\frac{\partial g}{\partial w}  \text{d}x = 0$

Replacing the functions $f$ and $g$ defined above:

$\int xp(x\|z) -\lambda xx^T p(x\|z)  \text{d}x = 0$

Solving yields (similar to maximization of growth rate)

$w \propto \Sigma_{x\|z}^{-1}\mu_{x\|z}$

The weights that optimizes sharpe ratio is proportional to the one that maximizes growth rate and proportionality constant must be constant for all values of $z$.




### Scaling must be consistent across time

Optimal growth weights tend to be large and practical constraints force us to _scale_ the weights in some fashion. For example, a common approach is to normalize the weights to unit leverage (sum of absolute value to one); also, from the above results, since the weights that optimize sharpe are proportional to the optimal weights why not make this normalization for each time instant? A trivial case of this is when we have some model for the returns and we just consider the sign of the prediction (go long if positive mean and vice-versa).

Let us illustrate better the optimal weight formula $w=\lambda \Sigma_{x\|z}^{-1}\mu_{x\|z}$ to provide a more comprehensive example on why the scaling factor should be the same across time otherwise the Sharpe ratio is lower.

The problem is easy to formulate in the discrete case. To do that let us assume that our _features_ are indicator variables and for each one the distribution of $x$ has mean $\mu_k$ and covariance $\Sigma_k$.

for one of the output distributions ($j$) we will use as optimal weight $w_j=\phi \cdot \Sigma_j ^{-1} \cdot \mu_j$ (where $\phi$ is a constant different from 1) instead of $\Sigma_j ^{-1} \cdot \mu_j$; for all other output distributions we use $ w_k=\Sigma_k ^{-1} \cdot \mu_k$. 

The sharpe ratio of this weighting scheme is (this is just a weighted mean and variance; the weights $p_i$ represent the number of times distribution $i$ is _seen_):

$\text{SR}=\frac{\phi p_j \mu_j^T \Sigma_j^{-1} \mu_j + \sum_{k-1} p_k \mu_k^T \Sigma_k^{-1} \mu_k }{\sqrt{\phi^2 p_j \mu_j^T \Sigma_j^{-1} \mu_j + \sum_{k-1} p_k \mu_k^T \Sigma_k^{-1} \mu_k }}$

As before, the sharpe ratio of using the optimal weights can be written as (with the $j$ term being taken out of the sum for comparison):

$\text{SRopt}=\frac{ p_j \mu_j^T \Sigma_j^{-1} \mu_j + \sum_{k-1} p_k \mu_k^T \Sigma_k^{-1} \mu_k }{\sqrt{ p_j \mu_j^T \Sigma_j^{-1} \mu_j + \sum_{k-1} p_k \mu_k^T \Sigma_k^{-1} \mu_k }}$

Of course the only difference is the $\phi$ term (the weight that we changed). To test the condition for $\text{SR} \le \text{SRopt}$, we can write in a more simplified way:

$\frac{A+\phi B}{\sqrt{A+\phi^2B}} \le \frac{A+B}{\sqrt{A+B}}=\sqrt{A+B}$

where $A = \sum_{k-1} p_k \mu_k^T \Sigma_k^{-1} \mu_k$ and $B=p_j \mu_j^T \Sigma_j^{-1} \mu_j$. This inequality is true for any $\phi \neq 1$. Also, trivially, one can check that if we multiply all $w_k$ for the same factor nothing changes because the factors cancel out. This concludes that, to achieve an optimal sharpe ratio _all_ optimal weights should be scaled by the same value. This is equivalent to keep proportionality over time of perceived risk of the model. 


