# Allocation Schemes


## PROBLEM STATEMENT

Consider $q$ assets with returns $x_1,x_2,\cdots,x_n$. A sequential investment with an allocation $w$ induces the following wealth dynamics:

$S_n=S_0 \cdot (1+w^T \cdot x_1) \cdot (1+ w^T \cdot x_2) \cdots$

i.e, the capital at discrete time $n$ is the result of reinvestment at each period with some weights $w$ on a sequence of independent returns $x_n$. As a result, geometric growth is created; with some manipulation we can write the following expression:

$S_n=S_0 \exp(n \frac{1}{n} \sum_i \log(1+w^T \cdot x_i))=S_0 \exp(n \cdot G)$

and $G$ can be understood as a geometric growth rate:
$G=\frac{1}{n} \sum_i \log(1+\vec w \cdot \vec x_i) \rightarrow_{LLN} \mathbf{E}[\log(1+ w^T \cdot x)]$

where the expectation should be undersood with respect to the (multivariate) distribution of $\vec x$, $p(\vec x)$.

The objective is to study allocation schemes, i.e, ways to determine $\vec w$, and compare their properties.




### CONSTANT DISTRIBUTION


The optimal allocation can be stated as finding $\vec w$ such that $G$ is maximized. If the distribution of returns, $p(\vec x)$, does not change over time and does do not have _large_ fluctuations then we can consider the Taylor expansion of $\log(1+ w^T \cdot x)$ near $w^T \cdot x=0$ to yield:

$\log(1+z) \approx z-\frac{1}{2}z^2 \rightarrow G=\mathbf{E}[w^T \cdot x]-\frac{1}{2}\mathbf{E}[(w^T \cdot x)^2]$


We can identify $\mathbf{E}[w^T \cdot x]$ as $w\^T \cdot \mu$ and 

$\mathbf{E}[(w^T \cdot x)^2]$ 

as the second non central moment of 

$w^T \cdot x$ (note that $w^T \cdot x$ is a scalar) and this 
moment is related to the variance as 
$\mu_2=\sigma^2+\mu_1^2$; 

for practical application (financial time series) it is realistic to assume that $\mu_1=w^T \cdot \mu$ is small, i.e, 

$\mu_2 \approx \sigma^2 \rightarrow \mathbf{E}[(w^T \cdot x)^2]=w^T \Sigma w$ with $\Sigma$ the covariance matrix of $x$. 

Since $G=w^T \cdot \mu - \frac{ w^T \Sigma w}{2}$, the optimal growth weight vector is $w^\*=\Sigma^{-1} \cdot \mu$ (take derivative of $G$ wrt $w$ and equate to zero) and the maximum growth rate is $G^\*=\frac{1}{2} \mu^T \Sigma^{-1} \mu$. 


##### Comparing investments

It is common practice to compare portfolios/investments/strategies based on Sharpe ratio $\frac{\mu}{\sigma}$. It normally assumed that this metric yields a measure of risk adjusted performance in an normalized way. If prices are stochastic then the way (period) which we measure fluctuations $\mu$ and $\sigma$ matter (and we have to measure them someway!). For a random walk we have that the mean _scales_ as $\mu_T = T \mu$ and $\sigma^2_T=T \sigma^2$. Given this, Sharpe ratio has dimensions $T^{-\frac{1}{2}}$. For example, if we take yearly measurements, we will have an average of _percentage per year_ and a standard deviation of _percentage per square root of year_ and, as an example, a sharpe of 5 on year basis is $\frac{5}{\sqrt{250}}=0.316$ on a daily basis (so we need to compare Sharpe ratios with quantities computed at the same time scale - it is not properly unormalized).


A better argument is to consider that the option with the highest Sharpe will always achieve a higher growth rate if _properly levered_:

Suppose we have two possible sets of assets (which can be just two single assets) to invest from which we know the true properties, i.e, $\vec \mu$ and $\Sigma$. Given that, (from the discussion above) in the first case the optimal allocation is $\vec w_1^*=\Sigma_1^{-1} \cdot \vec \mu_1$ and in the second case is $\vec w_2^*=\Sigma_2^{-1} \cdot \vec \mu_2$. As shown before, the growth rates using optimal weights are $G_1^*=\frac{1}{2} \vec \mu_1^T \Sigma_1^{-1} \vec \mu_1$ and $G_2^*=\frac{1}{2} \vec \mu_2^T \Sigma_2^{-1} \vec \mu_2$.

Under weights $\vec w_1^*$ and $\vec w_2^*$, the portfolio fluctuations have as Sharpe ratios $S_1=\sqrt{\mu_1^T \Sigma_1^{-1} \vec \mu_1}$ and $S_2=\sqrt{\mu_2^T \Sigma_2^{-1} \vec \mu_2}$ respectively, and it is possible to write:

$
G_1^*=\frac{1}{2}S_1^2 \ge G_2^*=\frac{1}{2}S_2^2 \text{ if } S_1>S_2
$

This relation is what gives sense to compare sharpes, not because it measures non dimensional quantities but because they are related to a more fundamental problem: optimizing growth rate.


In the end, finding strategies with high Sharpe is equivalent to find strategies that grow in an optimal way given that the leverage allocated to them is lower than the optimal one (which is usually the case in financial time series).



##### Practical note

Achieving geometric growth is difficult. There is also the problem of mispecification of the future distribution (model is wrong). Given this, it makes sense from a risk management prespective that in a first phase one is looking for aditive growth (not reivesting the capital - at least totaly) and, in an organic way, increase allocation to different models. We could ask now, if this is the case, then doesn't it make sense to consider only the additive problem (for example, we always have the same resources to allocate), which can be formulated as:

$
S_n=S_0+S_0 \cdot \vec w \cdot \vec x_1 +S_0 \cdot \vec w \cdot \vec x_2 + \cdots
$

which would give as optimal weights the ones that maximize $\mathbf{E}[\vec w \cdot \vec x]$? (note a lack of _penalization_ due to fluctuations in growth - if the capital is always the same we can achieve the ensemble average and the issue of non ergodicity - time average different from ensemble average - is non existent).

The answer is no since in practice no resource is infinite; if a idea is losing money then the capital would be reduced at some point even if at every rebalancing (until the reducing/increasing) we commit the same resources to it. The possibility to have the capital reduced or increased (which will always be a function of the real performance) will make the problem more similar to one with geometric growth. This line of thinking gives sense to use sharpe ratios as measure of performance and, as seen, if properly levered (which can happen asymptotically) these strategies achieve the highest growth rate.


#### $\vec w^*$ STRATEGIES HAVE THE HIGHEST SHARPE RATIO

Previously it was shown that a scheme with a higher Sharpe, when properly levered yield a higher growth rate. We can ask the (almost) reverse question as well: does the optimal weight $\vec w^*$ yield the best Sharpe?

Again, given that $\vec \mu$ and $\Sigma$ are known, $\vec w^*=\Sigma^{-1} \cdot \vec \mu$. Investing with $\vec w$ gives a Sharpe ratio of $\sqrt{\vec \mu^T \Sigma^{-1} \vec \mu}$. If we choose _any_ other $\vec w$, the Sharpe ratio of that scheme would be $\frac{\vec w^T \vec \mu}{\sqrt{ \vec w^T \Sigma \vec w}  }$.

Let us compare both cases by checking when the following inequality is true:

$
\frac{\vec w^T \vec \mu}{\sqrt{ \vec w^T \Sigma \vec w}  } \le \sqrt{\vec \mu^T \Sigma^{-1} \vec \mu}
$


by taking the square and manipulating the expression it is possible to write:

$
(\vec w^T \vec \mu)^2 \le (\vec \mu^T \Sigma^{-1} \vec \mu) \cdot ( \vec w^T \Sigma \vec w) 
$

which is true by (generalized) Cauchy-Schwarz Inequality (and $\Sigma$ is semi-definite positive - it is a covariance matrix).

It can be concluded that if we are looking for a high Sharpe ratio we should investing according to $\vec w^*$. If there are any constraints to be imposed on the system they should be incorporated by taking a multiple of $\vec w^*$ as this does not change Sharpe (for example, this shows that a portfolio optimization problem such as mean variance should be solved without constraints and the solution should be scaled after).



##### Numerical verification
The following code demonstrates this result with a numerical experiment. $p$ asset returns during $n$ periods are simulated. After the simulation, the moments $\vec \mu$ and $\Sigma$ are computed (analogous to the _true_ distribution parameters). Now it is easy to compute $\vec w^*$. To generate other candidates, $k$ random vectors without constraints are created (notice that, without any constraints, after many steps, all combinations will be convered, like if we found some $\vec w$ with constraints).

In the end, the code compares the Sharpe of the strategy under $\vec w^*$ with all other random $\vec w$ and check if some $\vec w$ achieved a higher Sharpe (did not happen). 



```python
import numpy as np
import matplotlib.pyplot as plt

# ---------
# PARAMETERS
p=6 # number of assets
n=1000 # number of time steps
scale=0.01 # scale of fluctiations
k=1000000 # number of random weights to be generated
# ---------

# simulate returns
x=np.random.normal(0,scale,(n,p))

# compute optimal strategy for these returns
mu=np.mean(x,axis=0)
cov=np.cov(x.T)
w=np.dot(np.linalg.inv(cov),mu)

# compute sharpe ratio of using kelly weights
s=np.sum(x*w,axis=1)
opt_sr=np.mean(s)/np.std(s)

# simulate many random weights and compute their sharpe ratio - there are no constraints on the weights so
# after many they will simulate weights obtained with constraints as well...
sr=np.zeros(k)
for i in range(k):
    w_random=np.random.normal(0,1,p)
    s_random=np.sum(x*w_random,axis=1)
    sr[i]=np.mean(s_random)/np.std(s_random)

# Compare results
print('Number of random strategies with a better sharpe than Kelly: ', np.where(sr>opt_sr)[0].size)
# Plot histograms of random sharpe ratios 
plt.hist(sr,density=True, bins=30)
plt.axvline(opt_sr)
plt.show()

```

    Number of random strategies with a better sharpe than Kelly:  0



![png](/images/betting/output_7_1.png)


### CHANGING DISTRIBUTION

To extend the previous results and make the connection to a more realistic set up, consider that the returns come from different distributions over time (and are known). It does not make sense to consider a constant rebalancing vector $\vec w$ over time. It must change with the distributions. Then, the sequential problem becomes: 


$
S_n=S_0 \cdot (1+\vec w_1 \cdot \vec x_1) \cdot (1+\vec w_2 \cdot \vec x_2) \cdots
$

Notice that weights change over time. The growth rate can be expressed in an analogous way

$
G=\frac{1}{n} \sum_i \log(1+\vec w_i \cdot \vec x_i)
$

Now, let us group the previous terms by the distribution where they come from


$
G=\sum_k \frac{n_k}{n} \left( \frac{1}{n_k} \sum_{i_k} \log(1+\vec w_k \cdot \vec x_{i_k}) \right)
$

which by the law of large numbers (and then using the Taylor expansion) is


$
G=\sum_k p_k \mathbf{E}[\log(1+\vec w_k \cdot \vec x)]|_{k} \approx \sum_k p_k (\vec w_k \cdot \vec \mu_k - \frac{1}{2} \vec w_k^T \Sigma_k \vec w_k)
$

where $p_k$ represents the probability that the returns comes from distribution $k$ - or in more detail, how many times over time we see returns coming from distribution $k$ (let us not confuse this with a mixture model).


The optimal allocation now depends on the next distribution of returns. If this distribution is $k$ then $\vec w_k=\Sigma_k^{-1} \cdot \vec \mu_k$ (this is easy to verify as all distributions are independent and so taking the derivative in order to each $\vec w_k$ yields this result).


As a simple example, consider the case where we invest (single asset) during $n$ periods and in the first $0.95 \cdot n$ periods the returns come from a normal distribution with $\mu_1$ and $\sigma_1^2$; after that they come from another distribution with $\mu_2$ and $\sigma_2^2$ (note that all of this is known at the beggining). Given that all fluctuations are independent but with diferent distributions, in the first part we should invest accoring to $w^*=\frac{\mu_1}{\sigma_1^2}$ and then go with $w^* = \frac{\mu_2}{\sigma_2^2}$. It is not difficult to conclude that, when the whole period is considered, the first distribution dominates the statistics of the wealth generated as it happened during much more time (in this case $p_1=0.95$).





##### How this is similar to a model output?

Actually, the previous construction can be compared to when a model for the returns is being used.

Consider a model $M$ for the returns; this model will output a future distribution based on some information $I$ (at the current time instant). Now, given the current information $I_t$, we get a output distribution for the next return $\vec x_{t+1}$; if we use the model many times, there will be situations where the information $I_t$ is equal to the information at some other time $I_{t+\tau}$ (consider for example an autoregressive model at one dimension; if today's fluctuation is 1% and in 10 days we see again 1% then our output distribution will be the same). Also, given this output distribution we will allocate according to $\frac{\mu}{\sigma^2}$. Extending this to a continuum of situations, the $p_k$ in $G$ formula should be interpreted as the number of times model $M$ sees information $I$. Also, under the assumptions of any model, all information needed to codify the next distribution is $I$ and so predictions are independent (given $I$) - this completes the analogy.

From this observations it should be clear that the optimal allocation - for growth - at every instant is to take the output distribution and find the optimal weight (Kelly weight) - is our model is correct of course. 

In the end, all of this proves the point of betting more if the model is more confident.

#### $\vec w_k^*$ STRATEGIES HAVE THE HIGHEST SHARPE RATIO

After many periods of using the model/observing changing distributions, the wealth fluctuations have as first moment $\mu_p=\sum_k p_k \vec w_k \vec \mu_k$ and $\sigma_p^2=\sum_k p_k \vec w_k^T \Sigma_k \vec w_k$ as second moment (it is a weighted mean and variance). If at each time instant where the next distribution is $k$, we use allocation $\vec w^*=\Sigma_k^{-1} \cdot \vec \mu_k$, after many observations the Sharpe ratio goes to (substitute $\vec w_k^*$ on $\frac{\mu_p}{\sqrt{\sigma_p^2}}$):

$
S_{w^*}=\frac{\sum_k p_k \vec \mu_k^T \cdot \Sigma_k^{-1} \cdot \vec \mu_k}{\sqrt{\sum_k p_k \vec \mu_k^T \cdot \Sigma_k^{-1} \cdot \vec \mu_k}}=\sqrt{\sum_k p_k \vec \mu_k^T \cdot \Sigma_k^{-1} \cdot \vec \mu_k}
$


Now, if we use any other strategy $\vec w$, the Sharpe ratio is:

$
S_{w}=\frac{\sum_k p_k \vec w_k^T \vec \mu_k}{\sqrt{\sum_k p_k \vec w_k^T \Sigma_k \vec w_k}}
$

As before, let us search for the condition where $S_w \le S_{w^*}$:

$
\frac{\sum_k p_k \vec w_k^T \vec \mu_k}{\sqrt{\sum_k p_k \vec w_k^T \Sigma_k \vec w_k}} \le \sqrt{\sum_k p_k \vec \mu_k^T \cdot \Sigma_k^{-1} \cdot \vec \mu_k}
$

Taking the square, we can write:

$
\frac{\left( \sum_k p_k \vec w_k^T \vec \mu_k \right)^2}{\sum_k p_k \vec w_k^T \Sigma_k \vec w_k} \le \sum_k p_k \vec \mu_k^T \cdot \Sigma_k^{-1} \cdot \vec \mu_k
$

Using a special case of the Cauchy–Schwarz inequality (Bergström's inequality) and then again the (generalized) Cauchy–Schwarz inequality (with covariance matrices positive definite), the left hand side can be bounded two times as:

$
\frac{\left( \sum_k p_k \vec w_k^T \vec \mu_k \right)^2}{\sum_k p_k \vec w_k^T \Sigma_k \vec w_k} \le \sum_k p_k \frac{(\vec w_k^T \cdot \vec \mu_k)^2}{\vec w_k^T \Sigma_k \vec w_k} \le \sum_k p_k \frac{\vec w_k^T \Sigma_k \vec w_k \cdot \vec \mu_k \Sigma_k^{-1} \vec \mu_k}{\vec w_k^T \Sigma_k \vec w_k}=\sum_k p_k \vec \mu_k^T \Sigma_k^{-1}\vec \mu_k
$

And with the last equality the result is proven. The best sharpe ratio is achived when $\vec w_k^*$ is used and any multiple of this strategy has the best sharpe.

##### Other view of the solution

The previous problem can be solved by considering $\sum_k p_k f(\cdot)$ as an expected value (this may be more intuitive when the states $k$ tend to a continuum). In that case, the Sharpe ratio usign $\vec w^*$ is $\sqrt{\mathbf{E}[\vec \mu^T \cdot \Sigma^{-1} \cdot \vec \mu]}$ and for any other generic $\vec w$ is $\frac{\mathbf{E}[\vec w^T \vec \mu]}{\sqrt{\mathbf{E}[\vec w^T \Sigma \vec w]}}$ (with the expectation being taken with respect to the different distributions probability - for example, a mean of the means in each state). As usual, we want to check the following inequality:

$
\frac{\mathbf{E}[\vec w^T \vec \mu]}{\sqrt{\mathbf{E}[\vec w^T \Sigma \vec w]}} \le \sqrt{\mathbf{E}[\vec \mu^T \cdot \Sigma^{-1} \cdot \vec \mu]}
$

Taking the square:

$
\frac{\left( \mathbf{E}[\vec w^T \vec \mu] \right)^2}{\mathbf{E}[\vec w^T \Sigma \vec w]} \le \mathbf{E}[\vec \mu^T \cdot \Sigma^{-1} \cdot \vec \mu]
$

which is always true by:

$
\frac{\left( \mathbf{E}[\vec w^T \vec \mu] \right)^2}{\mathbf{E}[\vec w^T \Sigma \vec w]} \le_{\text{Jensen}} \frac{\mathbf{E}[ \left( \vec w^T \vec \mu \right)^2]}{\mathbf{E}[\vec w^T \Sigma \vec w]} \le_{\text{Jensen}} \mathbf{E}[\frac{\left( \vec w^T \vec \mu \right)^2}{\vec w^T \Sigma \vec w}] \le_{\text{Cauchy-Schwarz}} \mathbf{E}[\frac{\vec w^T \Sigma \vec w \cdot \vec \mu^T \Sigma^{-1} \vec \mu}{\vec w^T \Sigma \vec w}] = \mathbf{E}[\vec \mu^T \cdot \Sigma^{-1} \cdot \vec \mu]
$

which proves the statement.



##### Numerical verification
The following code demonstrates this result with a numerical experiment. The objective is to compare the sharpe ratio by using the optimal weights for each distribution that originated the returns. $p$ asset returns during $n$ periods are simulated - the first half are from a distribution and last one from another. After the simulation, the moments $\vec \mu_k$ and $\Sigma_k$ are computed (analogous to the _true_ distribution parameters) for each distribution. Now it is easy to compute $\vec w_k^*$. To generate other candidates, $k$ random vectors without constraints are created (notice that, without any constraints, after many steps, all combinations will be convered, like if we found some $\vec w$ with constraints).

Not that we are comparing changing the weights to the optimal ones versus a constant allocation. Each individual state is like the previous numerical example and so we do not need to repeat the experiment. It is possible to observe that this is the optimal solution.



```python
import numpy as np
import matplotlib.pyplot as plt

# ---------
# PARAMETERS
p=6 # number of assets
n=1000 # number of time steps
scale=0.01 # scale of fluctiations
k=100000 # number of random weights to be generated
# ---------

# simulate returns
x1=np.random.normal(0,scale,(int(n/2.),p))
x2=np.random.normal(0,scale,(int(n/2.),p))
x=np.vstack((x1,x2))

# compute optimal strategy for these returns
mu1=np.mean(x1,axis=0)
cov1=np.cov(x1.T)
w1=np.dot(np.linalg.inv(cov1),mu1)

mu2=np.mean(x2,axis=0)
cov2=np.cov(x2.T)
w2=np.dot(np.linalg.inv(cov2),mu2)


# compute sharpe ratio of using kelly weights
s=np.hstack((np.sum(x1*w1,axis=1),np.sum(x2*w2,axis=1)))

opt_sr=np.mean(s)/np.std(s)

# simulate many random weights and compute their sharpe ratio - there are no constraints on the weights so
# after many they will simulate weights obtained with constraints as well...
sr=np.zeros(k)
for i in range(k):
    w_random=np.random.normal(0,1,p)
    s_random=np.sum(x*w_random,axis=1)
    sr[i]=np.mean(s_random)/np.std(s_random)

# Compare results
print('Number of random strategies with a better sharpe than Kelly: ', np.where(sr>opt_sr)[0].size)
# Plot histograms of random sharpe ratios 
plt.hist(sr,density=True, bins=30)
plt.axvline(opt_sr)
plt.show()
```

    Number of random strategies with a better sharpe than Kelly:  0



![png](/images/betting/output_12_1.png)


#### DISTRIBUTION CONSISTENT SCALING OPTIMALITY  

In practice we want to control the leverage and it is easy to see that a fractional Kelly strategy will have the same sharpe as the full Kelly when the distribution does not change. When the distribution changes there is nothing preventing us to build a different fractional strategy for each distribution $k$, i.e, having different scalings for different outputs - since they are independent given the model this may seem plausible and makes the problem much simpler to solve; a simple case of this is to have the output of a univariate model and bet on the sign of the expected value. Let us see now this is not the case and the scaling factor should be the same across distribution otherwise the Sharpe ratio is lower.

The problem is easy to formulate in the discrete case. To do that let us assume, for one of the output distributions ($j$) we will use as optimal weight $\vec w_j^*=\phi \cdot \Sigma_j ^{-1} \cdot \vec \mu_j$ (where $\phi$ is constant different from 1) instead of $\Sigma_j ^{-1} \cdot \vec \mu_j$; for all other output distributions we use $\vec w_k^*=\Sigma_k ^{-1} \cdot \vec \mu_k$. The sharpe ratio of this weighting scheme is:

$
S=\frac{\phi p_j \vec \mu_j^T \Sigma_j^{-1} \vec \mu_j + \sum_{k-1} p_k \vec \mu_k^T \Sigma_k^{-1} \vec \mu_k }{\sqrt{\phi^2 p_j \vec \mu_j^T \Sigma_j^{-1} \vec \mu_j + \sum_{k-1} p_k \vec \mu_k^T \Sigma_k^{-1} \vec \mu_k }}
$

As before, the sharpe ratio of using the optimal weights can be written as (with the $j$ term being taken out of the sum for comparison):

$
S_K=\frac{\vec \mu_j^T \Sigma_j^{-1} \vec \mu_j + \sum_{k-1} p_k \vec \mu_k^T \Sigma_k^{-1} \vec \mu_k }{\sqrt{ p_j \vec \mu_j^T \Sigma_j^{-1} \vec \mu_j + \sum_{k-1} p_k \vec \mu_k^T \Sigma_k^{-1} \vec \mu_k }}
$

Of course the only difference is the $\phi$ term (the weight that we changed). To test the condition for $S \le S_K$, we can write in a more simplified way:

$
\frac{A+\phi B}{\sqrt{A+\phi^2B}} \le \frac{A+B}{\sqrt{A+B}}=\sqrt{A+B}
$

where $A=\sum_{k-1} p_k \vec \mu_k^T \Sigma_k^{-1} \vec \mu_k $ and $B=p_j \vec \mu_j^T \Sigma_j^{-1} \vec \mu_j$.This inequality is true for any $\phi < 1$ and equality is verified for the trivial case $\phi=1$. Also, trivially, one can check that if we multiply all $\vec w_k$ for the same factor nothing changes because the factors cancel out. This concludes that, to achieve an optimal sharpe ratio _all_ optimal weights should be scaled by the same value. This is equivalent to keep proportionality over time of perceived risk of the model. Without this the optimal result is not achieved.

This result emphasizes the importance to bound the optimal weights for any model allowing fluctuations on the overall leverage - this will achieve superior results.



##### Numerical verification
The following code demonstrates this result with a numerical experiment. The objective is to compare what happens when the weights from a given distribution are scaled with a constant. $p$ asset returns during $n$ periods are simulated - the first half are from a distribution and last one from another. After the simulation, the moments $\vec \mu_k$ and $\Sigma_k$ are computed (analogous to the _true_ distribution parameters) for each distribution. Now it is easy to compute $\vec w_k^*$. Now to do the comparison we will scale the optimal weights of the second distribution by a factor and check what happened to the overall sharpe.



```python
import numpy as np
import matplotlib.pyplot as plt

# ---------
# PARAMETERS
p=6 # number of assets
n=1000 # number of time steps
scale=0.01 # scale of fluctiations
k=100000 # number of random weights to be generated
# ---------

# simulate returns
x1=np.random.normal(0,scale,(int(n/2.),p))
x2=np.random.normal(0,scale,(int(n/2.),p))
x=np.vstack((x1,x2))

# compute optimal strategy for these returns
mu1=np.mean(x1,axis=0)
cov1=np.cov(x1.T)
w1=np.dot(np.linalg.inv(cov1),mu1)

mu2=np.mean(x2,axis=0)
cov2=np.cov(x2.T)
w2=np.dot(np.linalg.inv(cov2),mu2)


# compute sharpe ratio of using kelly weights
s=np.hstack((np.sum(x1*w1,axis=1),np.sum(x2*w2,axis=1)))
opt_sr1=np.mean(s)/np.std(s)

# compute sharpe ratio of using a scaling factor in one of the kelly weights
s=np.hstack((np.sum(x1*w1,axis=1),np.sum(0.5*x2*w2,axis=1)))

opt_sr2=np.mean(s)/np.std(s)

print('Full Kelly: ', opt_sr1)
print('Distribution 2 scaled: ', opt_sr2)


```

    Full Kelly:  0.1356385247056015
    Distribution 2 scaled:  0.1280333382923466


As predicted, the sharpe ratio is lower. This result has implication for the design of trading strategies on models as it is proved that bounding the kelly weights can provide better practical results.

##### Numerical example with HMM process
The following code provides an example of a strategy build on top on an HMM process. Given that have a well defined output distribution it is possible to bound the kelly weights (just check the maximum kelly weight for each distribution in the output mixture distribution; also, not that the output is a mixture distribution and care must be taken estimating the moments of this mixture!). Let us show that using Kelly weights yield a superior strategy in terms of sharpe ratio rather than using the sign of the prediction. This simple example can open new ideas to use other models.


```python
from seq import simulate,HMM,GaussianEmission
import numpy as np
import matplotlib.pyplot as plt
```


```python
# simulate an HMM process
t=5000
P=np.array([0.5,0.5])
A=np.array([[0.9,0.1],[0.1,0.9]])
n_states=A.shape[0]
mu=np.array([0.01,-0.01])
scale=np.array([0.025,0.05])
# Simulate
emissions=GaussianEmission(n_states,mu=mu,scale=scale)
# emissions=LaplaceEmission(n_states,mu=mu,scale=scale)
z,x=simulate(t,A,emissions,P,plot=True)



```

    ** Conditional distributions **
    State 0
    mu:  0.010192689633421705
    std:  0.0246559689886545
    State 1
    mu:  -0.011013724902729775
    std:  0.049452309769170304



![png](/images/betting/output_19_1.png)



```python
# train the model with the first half of data
n_states=2
emissions=GaussianEmission(n_states)
hmm_model=HMM(n_states,emissions)
hmm_model.train(x[:int(t/2.)],10,10,20)
hmm_model.view_params()
hmm_model.view_convergence(show=True)
```

    ** HMM Parameters **
    Initial state distribution
    [1. 0.]
    Transition
    [[0.92  0.08 ]
     [0.089 0.911]]
    Emissions
    ** Gaussian Emission **
    State 1
    -> Mean:  -0.009319085214435053
    -> Scale:  0.05029615064456281
    State 2
    -> Mean:  0.009207480079514989
    -> Scale:  0.026420330366883755
    



![png](/images/betting/output_20_1.png)



```python
# evaluate a trading strategy with the full distribution output and with the sign of the prediction

w_sign=0
w_frac_k=0
w_full_k=0

s_sign=np.zeros(t)
s_frac_k=np.zeros(t)
s_full_k=np.zeros(t)

# get kelly bound from the mixture
k=hmm_model.emissions.calc_kelly_bound(max_prob=1)

for i in range(1,t):
    s_sign[i]=w_sign*x[i]
    s_frac_k[i]=w_frac_k*x[i]
    mu,cov=hmm_model.predict(x[:i+1],l=50)
    w_sign=np.sign(mu)
    w_full_k=mu/cov # it only one dimension...
    w_frac_k=w_full_k/k # normalize to get a fractional kelly strategy

prec=6
print('Sharpe Sign Strategy: ', round(np.mean(s_sign)/np.std(s_sign),prec) )
print('Sharpe Fractional Kelly Strategy: ', round(np.mean(s_frac_k)/np.std(s_frac_k),prec))

plt.plot(np.cumsum(x),color='k',label='HMM Process')
plt.plot(np.cumsum(s_sign),color='b',label='Sign Strategy')
plt.plot(np.cumsum(s_frac_k),color='r',label='Fractional Kelly Strategy')
plt.legend()
plt.show()
```

    Sharpe Sign Strategy:  0.129937
    Sharpe Fractional Kelly Strategy:  0.143929



![png](/images/betting/output_21_1.png)


The fractional Kelly grew less but has a higher sharpe as expected. If properly leveraged this strategy will have a higher growth rate.

# Relation to model selection

## PROBLEM STATEMENT

We saw that Kelly weights (of a fractional quantities of them) are optimal in several ways including Sharpe. Going forward to the modeling itself, first we give an introduction to model assesement/selection and then show that, by using Kelly weights, we get a proxy for the probability of the model given the data.

## Bayesian Model Comparison and Cross Validation

Given some data $D$ and a set of models $M_j$, we can write the posterior of the models as $p(M_j|D) \propto p(M_j)p(D|M_j)$. This yields a way to compare models (we can assume they have equal prior probabilities without much damage; anyway this just shows that there are no absolute inferences) by comparing the _evidence_ $p(D|M_j)$. 

Of course models will have parameters that need to be determined. Letting $\theta_j$ be the set of parameters for model $M_j$, the evidence can be written as:

$
p(D|M_j)=\int p(D|\theta_j,M_j) p(\theta_j|M_j) \text{d}\theta
$

This view of parameters creates some sort of regularization and will penalize models that are too complex. To illustrate this, consider a model that has only one parameter $\theta_j$ (its a scalar instead of a vector); if $p(D|\theta_j,M_j)$ is narrow near the parameter $\theta_j$ that maximizes the likelihood, $\hat{\theta_j}$, with width $\delta_{\text{posterior}}$ and we use a flat prior so that $p(\theta_j|M_j)=1/\delta_{\text{prior}}$ then the integral becomes:


$
p(D|M_j)=\int p(D|\theta_j,M_j) p(\theta_j|M_j) \text{d}\theta \approx p(D|\hat{\theta_j},M_j)\frac{\delta_{\text{posterior}}}{\delta_{\text{prior}}}
$

and so

$
\log(p(D|M_j))=\log(p(D|\hat{\theta_j},M_j)) + \log(\frac{\delta_{\text{posterior}}}{\delta_{\text{prior}}})
$


and so, if the width of the posterior $\delta_{\text{posterior}}$ is very small in relation to $\delta_{\text{prior}}$ this term is negative and there is a penalty in $p(D|M_j)$; if this term is small this means that the model is fine tuned to the data and so we may be overfitting.

Given this idea and equal prior then a model must be chosen from a balance from quality of fit and complexity.



Another way to look at the problem is, after fitting the models on a set $D_1$, we can compute the probabilities $p(D_2|M_j)$ and solve this by just considering more data - many times this is not practical as data is limited.

### Predictive iterpretation and Cross-Validation

We factorice $p(D|M_j)$ as $p(D|M_j)=p(x_0|M_j) \cdot p(x_1|x_0,M_j)\cdot p(x_2|x_0,x_1,M_j) \cdots p(x_n|x_{0:n-1},M_j)$ (notice that $D$ is a set of observations $x_0,x_1,\cdots$); this factorization tell us that the probability of the data given the model can be interpreted as predictive capacity. Now, to build a link with cross validation, consider that the observations are independent; we can change their order and we still obtain the same value for $p(D|M_j)$. Let $\sigma_{k_1}$ be a permutation of the indexes of $x_i$ in $D$ and let $X_1^{k_1}=X_{0:n-p}^{k_1}$ be the first $n-p$ observations of permutation $k_1$ and $X_2^{k_1}=X_{n-p:n}^{k_1}$ the last $p$ observations. In the same fashion we can integrate across parameters $\theta_j$:

$
p(D|M_j)=\int p(D|\theta_j,M_j) p(\theta_j|M_j) \text{d}\theta = \int p(X_2^{k_1}|X_1^{k_1},\theta_j,M_j) p(X_1^{k_1}|\theta_j,M_j) p(\theta_j|M_j) \text{d}\theta
$


## Kelly weighting is a proxy to probability of model

Now we go to the important result of this section: if we use Kelly weights given the model output distribution we can compute at the same time an estimation of out of sample Sharpe/performance and make probabilistic model comparisson - which is not true under other allocation scheme!

First let us consider the case where the model that generated the data (the true model) is under the set of models being considered and let us consider the situation where we have abundant data and is possible to fit on a set $D_1$ and evaluate on a set $D_2$. After the model is fit we make distribution predictions on $D_2$; if the correct model is on the set of models (assuming we have sufficient data to make precise inference of parameters) and we invest according to $w=\Sigma^{-1}\mu$ - the weights can change for every point; this was discussed previously - it was shown that this will give the highest Sharpe ratio and no other allocation scheme will achieve that. Since this is an upper bound then any other model will have a lower out of sample Sharpe ratio if the allocation for each model is made with Kelly. Since the true model will have the highest $p(D|M)$ then the result is proved.

When the true model is not present (and this is the case for any practical application) it is not easy to prove that $p(D|M_1)>p(D|M_2)$ implies $SR_1>SR_2$ for the Kelly allocation. We can argue/conjecture that, by using Kelly, if $SR_1>SR_2$ then we must be closer to the true distribution and since Kelly yields an upper bound then it is proved. The best way to do this it make numerical experiments!

#### Numerical experiment

We can devise a simple experiment where we simulate an autoregressive process. This process will depend on $x_{t-2}$ only.

First we simulate many observations of the process and fit several AR processes of different orders. After that we can simulate more (no need for cross validation as we are able to generate as many observations as we like), make predictions and simulate an investment strategy with Kelly weights. Then we can check if the correct model is found by looking at the best sharpe ratio. Also, we will test a strategy where we make prediction with the sign of expected return and check if this is also able to find the correct model.




```python
import numpy as np
import matplotlib.pyplot as plt
import seqmodels

scale=0.01
a=np.array([0.0,0.5,0.0,0.0])
n=1000

def simulate_ar(n,a,scale):
    x=np.zeros(n)
    for i in range(a.size,n):
        x[i]=np.dot(a[::-1],x[i-a.size:i])+np.random.normal(0,scale)
    return x

x=simulate_ar(n,a,scale)

plt.plot(x)
plt.show()


```


![png](/images/betting/output_26_0.png)



```python
class ARp(object):
    def __init__(self,p=1):
        self.p=p
        self.a=None
        self.scale=None
        
    def estimate(self,x):
        # build lags
        x=np.copy(x)
        _,x,y=seqmodels.ts_embed(x,self.p)
        c=np.dot(x.T,x)/x.shape[0]
        d=np.dot(x.T,y)/x.shape[0]
        self.a=np.dot(np.linalg.inv(c),d).ravel()#[::-1]
        self.scale=np.sqrt(np.mean(np.power(y.ravel()-np.sum(x*self.a,axis=1),2)))

    def predict(self,x):
        x=np.copy(x)
        x_est=np.zeros(x.size)
        _,x_,y_=seqmodels.ts_embed(x,self.p)
        x_est[self.p:]=np.sum(x_*self.a,axis=1)
        return x_est
    
    def view(self):
        print('** PARAMS **')
        print('a: ',self.a)
        print('scale: ', self.scale)
```


```python
# TRAIN SEVERAL MODELS
models=[]
p_orders=np.arange(1,8)
for i in range(p_orders.size):
    print('P: ' ,p_orders[i])
    model=ARp(p=p_orders[i])
    model.estimate(x)
    model.view()
    models.append(model)
```

    P:  1
    ** PARAMS **
    a:  [-0.05574695]
    scale:  0.011494560402165392
    P:  2
    ** PARAMS **
    a:  [ 0.48044356 -0.02896925]
    scale:  0.010086207510461423
    P:  3
    ** PARAMS **
    a:  [-0.01636713  0.47996948 -0.02110419]
    scale:  0.010089913302653557
    P:  4
    ** PARAMS **
    a:  [-0.05989298 -0.0175888   0.50872486 -0.0221248 ]
    scale:  0.010076892985517661
    P:  5
    ** PARAMS **
    a:  [-0.05513112 -0.06105369  0.01045496  0.50770685 -0.02541254]
    scale:  0.01006571494377389
    P:  6
    ** PARAMS **
    a:  [ 0.0206824  -0.05463826 -0.07154391  0.01032496  0.50895578 -0.02440856]
    scale:  0.01006810251207452
    P:  7
    ** PARAMS **
    a:  [-0.01383318  0.0203291  -0.04763689 -0.07117946  0.00948513  0.5077636
     -0.02442523]
    scale:  0.010067515557718358



```python
# SIMULATE OUT OF SAMPLE DATA
x_oos=simulate_ar(n,a,scale)
```


```python
print('Using Kelly weights')
sr=[]
for i in range(p_orders.size):
    print('P: ' ,p_orders[i])
    x_est=models[i].predict(x_oos)
    w=x_est/np.power(models[i].scale,2)    
    s=w*x_oos
    sr.append(np.mean(s)/np.std(s))
    print('Sharpe: ', np.mean(s)/np.std(s))
    print('-----------')
print('Best model order: ', p_orders[np.argmax(sr)])
```

    Using Kelly weights
    P:  1
    Sharpe:  0.013905427360979416
    -----------
    P:  2
    Sharpe:  0.45541005806982
    -----------
    P:  3
    Sharpe:  0.45406705968385114
    -----------
    P:  4
    Sharpe:  0.4510755749846265
    -----------
    P:  5
    Sharpe:  0.4527483815665523
    -----------
    P:  6
    Sharpe:  0.4523233281125876
    -----------
    P:  7
    Sharpe:  0.45128148625268916
    -----------
    Best model order:  2



```python
print('Sign prediction')
sr=[]
for i in range(p_orders.size):
    print('P: ' ,p_orders[i])
    x_est=models[i].predict(x_oos)
    w=np.sign(x_est)#/np.power(models[i].scale,2)    
    s=w*x_oos
    sr.append(np.mean(s)/np.std(s))
    print('Sharpe: ', np.mean(s)/np.std(s))
    print('-----------')
print('Best model order: ', p_orders[np.argmax(sr)])
```

    Sign prediction
    P:  1
    Sharpe:  0.004899669444309259
    -----------
    P:  2
    Sharpe:  0.45129113917951647
    -----------
    P:  3
    Sharpe:  0.44942632296515816
    -----------
    P:  4
    Sharpe:  0.4465852799621619
    -----------
    P:  5
    Sharpe:  0.44968629536826266
    -----------
    P:  6
    Sharpe:  0.46253083243772547
    -----------
    P:  7
    Sharpe:  0.46661436593276795
    -----------
    Best model order:  7


##### Conclusion
Using Kelly weights was able to identify the correct model. By doing this we can estimate the out of sample performance and select the best parsimonius model. Of course this conclusions can change from run to run due to finite sampling but on average the conclusions hold!



```python

```

