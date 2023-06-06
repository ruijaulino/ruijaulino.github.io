# Tracking a distribution


A common and much discussed trading idea/_model_ is to say that if something is going up[down] one should buy[sell] - trend following; doing this induces some convexity in the PnL distribution and, together with it's simplicity, makes it worth the study (which asset classes are better modelled by it, time scales, costs, etc). Much of it is done with some heuristic rules (taking a position based on the movement of the recent past, using moving averages crossovers and/or combining many approaches) and then rely on diversification usign the same idea on many assets. 

From a statistical prespective we need future fluctuations to be positivelly correlated with the previous ones and this can be a path to create a model; my objective here is to make a _filter_ to track a changing distribution in a way that 1) the final model resembles a trend following idea and 2) make a distributional prediction in order to use the betting schemes that were discussed in other posts of this blog (of couse having a model with these properties also enables a more rigorous _fitting_/_learning_ since we can optimize parameters for probability of data given parameters and not for resulting strategy sharpe ratio for example).

For this end, I will consider a sequence of returns and assume that they are generated from a normal distribution with $\mu$ and $\Sigma$ that vary _slowly_ over time and are not observed directly.


## Model

Let $x_t$ be the (multivariate) returns at some instant $t$:

$x_t \sim p(x_t\|\mu_t,\Sigma_t)$

Also, the state/hidden variables/parameters, depend on their previous values under some law

$\mu_t,\Sigma_t \sim p(\mu_t,\Sigma_t\|\mu_{t-1},\Sigma_{t-1})$


Given a new return observation, the objective is to compute the filtering distribution:

$p(\mu_t,\Sigma_t\|x_{1:t})$

Assume that, at some instant $t$ we have the joint distribution for the latent variables. A natural (and known) candidate for this is the Normal-Inverse-Wishart distribution. It is conjugate to the Gaussian (which we are assuming as distribution for the returns); it's general form is

$p(\mu,\Sigma\|m,\lambda,\nu,V) = \text{N}(\mu\|m,\frac{1}{\lambda}\Sigma) \text{IW}(\Sigma\|\nu,V)   \propto \|\Sigma\|^{-\frac{\nu+d+2}{2}} \exp \left(  -\frac{\lambda}{2} (\mu-m)^T \Sigma^{-1} (\mu-m) - \frac{1}{2} \text{Tr}(\Sigma^{-1}V) \right) $

When used as prior for the parameters of a Gaussian, $p(\mu,\Sigma\|x) \propto p(x\|\mu,\Sigma)p(\mu,\Sigma)$, the posterior is also a NIW distribution.

With that in mind, let us assume that at time $t-1$, the distribution of the state variables is:

$p(\mu_{t-1},\Sigma_{t-1}\|x_{1:t-1}) = \text{NIW}\left( m_{t-1},\lambda_{t-1},\nu_{t-1},V_{t-1} \right)$

From Bayes:

$p(\mu_t,\Sigma_t\|x_{1:t}) \propto p(x_t\|\mu_t,\Sigma_t)p(\mu_t,\Sigma_t\|x_{1:t-1})$

Now, the second term can be written as (due to the dependencies assumed in the model)

$p(\mu_t,\Sigma_t\|x_{1:t-1}) = \int \int p(\mu_t,\Sigma_t,\mu_{t-1},\Sigma_{t-1}\|x_{1:t-1}) \text{d} \mu_{t-1} \text{d} \Sigma_{t-1} = \int \int p(\mu_t,\Sigma_t\|\mu_{t-1},\Sigma_{t-1}) p(\mu_{t-1},\Sigma_{t-1}\|x_{1:t-1}) \text{d} \mu_{t-1} \text{d} \Sigma_{t-1}$

We do not have a law for latent variables dynamics but, it would be interesting and practical if $p(\mu_t,\Sigma_t\|\mu_{t-1},\Sigma_{t-1})$ had such a form that the previous expression reduced to a NIW distribution as well; this construction is quite difficult to make but we can make the assumption that this happens, i.e, $p(\mu_t,\Sigma_t\|x_{1:t-1}) \sim NIW(\cdot)$. One sensible way to transform the parameters is to assume that latent variables expectations is kept unchanged but their uncertainty increases (of course one can think of more transformation that make sense but we will stay with this simple one). With this in mind, let us consider/assume (__prediction step__):

$p(\mu_t,\Sigma_t\|x_{1:t-1}) \sim \text{NIW}\left( m_{t}^-,\lambda_{t}^-,\nu_{t}^-,V_{t}^- \right)$

with

$m_{t}^-=m_{t-1}$

$\lambda_{t}^-=\phi \lambda_{t-1}$

$\nu_{t}^-=\phi (\nu_{t-1}-d-1)+d+1$

$V_{t}^-=\phi V_{t-1}$

It is easy to see that this transformation of parameters, for $\phi \in (0,1)$, preserves the expected values of $\mu$ and $\Sigma$. 

Now, going back to the estimation of the filtering distribution 

$p(\mu_t,\Sigma_t\|x_{1:t}) \propto p(x_t\|\mu_t,\Sigma_t)p(\mu_t,\Sigma_t\|x_{1:t-1})$

we can observe that $p(\mu_t,\Sigma_t\|x_{1:t})$ is also a NIW distribution with (__update step__):

$p(\mu_t,\Sigma_t\|x_{1:t}) \sim \text{NIW}\left( m_{t},\lambda_{t},\nu_{t},V_{t} \right)$

with 

$m_{t}=\frac{\lambda_t^- m_t^-+x_t}{1+\lambda_t^-}$

$\lambda_{t}=\lambda_{t}^- + 1$

$\nu_{t}=\nu_t^- + 1$

$V_{t}=V_t^- + \frac{\lambda_t^-}{\lambda_t^-+1}(x_t-m_t^-)(x_t-m_t^-)^T$


This recursive formulas can be used as new information arrives to estimate the current mean and covariance. Also, we can _predict_ the next values (although, by design they are the same as the ones just estimated) and use that information to make a bet; also, we can notice that notice that this simple model is just an estimate of mean and covariance with the most recent observations weighted in an exponential way ($\phi$ controls the decay). 


#### On the limit

When data arrives, parameters change; in the limit of many iterations one can observe that $\lambda$ converges to $\lim_{t \rightarrow \inf} \lambda_t = \lambda = \frac{1}{1-\phi}$ and $\nu$ to $\lim_{t \rightarrow \inf} \nu_t  = \nu = p + 1 + \frac{1}{1-\phi}$ (the fact that we need more observations than variables to estimate the covariance is reflected here). Replacing this into the updates for $m_t$ and $V_t$ we see that, after many iterations, they are updated as

$m_t = \phi m_{t-1} + (1-\phi) x_t$

and

$V_t = \phi \left( V_{t-1} + (x_t-m_{t-1})(x_t-m_{t-1})^T \right)$

In particular, consider some iterations for $m_t$:

$t=0 \rightarrow m_0$

$t=1 \rightarrow m_1=\phi m_0 + (1-\phi)x_1$

$t=2 \rightarrow m_2=\phi^2 m_0 + \phi(1-\phi)x_1 + (1-\phi) x_2$

$t=3 \rightarrow m_2=\phi^3 m_0 + \phi^2(1-\phi)x_1 + \phi(1-\phi) x_2 + (1-\phi) x_3$

and for $V_t$:

$t=0 \rightarrow V_0$

$t=1 \rightarrow V_1=\phi V_0 + \phi (x_1-m_0)(x_1-m_0)^T$

$t=2 \rightarrow V_2=\phi^2 V_0 + \phi^2 (x_1-m_0)(x_1-m_0)^T + \phi (x_2-m_1)(x_2-m_1)^T$

$t=3 \rightarrow V_3=\phi^3 V_0 + \phi^3 (x_1-m_0)(x_1-m_0)^T + \phi^2 (x_2-m_1)(x_2-m_1)^T + \phi (x_3-m_2)(x_3-m_2)^T$

And so, at any given instant, the previous observations are weighted as $\phi^t(1-\phi)=\phi^t-\phi^{t+1}$ for the mean and $\phi^{t+1}$ for the covariance ($t$ means the relative distance to the current time step), which represents the fact that we will use more _observations_ to estimate the mean (in other words, distance past observations have a larger weight than on the covariance) when $1>\phi>\frac{1}{2}$ (and the reverse).


#### Other considerations

In practice we still have to determine the parameter $\phi$; this can be done by usign the prediction error decomposition and noticing that the posterior predictive is a T-distribution - this way we have a _trend following_ model that can be trained as a regular model instead of focusing on sharpe ratios or other second order metrics that may hide the fact that the model may not fit the data that well.

As exposed, the idea can be used to track a multivariate distribution; care must be taken here if one is to invert the estimated covariance later (for betting): as the number of assets grow the inverse gets too unstable. I believe it is better to solve each problem individually or just consider a diagonal covariance for betting.

Another consideration is the bets: given the future prediction we can bet with $w=\Sigma^{-1} \mu$ but we need to normalize this quantity in a proper way (in the sense of the other posts in this blog); this can be done with some training data to check the leverage values that are observed.


## Numerical experiment

As a example, the following code implements the previous ideas in a simple way (these should not be regarded as final trading models; the objective is to show a crude application of the model).



```python
import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python

def wfbt(strategy,x):
    '''
    strategy: class with a method get_weight
    x: numpy (n,p) array with returns
    '''    
    n=x.shape[0]
    p=x.shape[1]
    s=np.zeros(n,dtype=float)
    leverage=np.zeros(n,dtype=float)
    w=np.zeros(p,dtype=float)
    for i in range(1,n):
        leverage[i]=np.sum(np.abs(w))
        s[i]=np.dot(w,x[i])
        x_=x[:i+1]
        w=strategy.get_weight(x_)
    return s,leverage


class GaussianTrack(object):    
    def __init__(self,phi):        
        self.phi=phi
        self.l=10 # some number just to initialize
        self.v=self.l
        self.m=None
        self.k=self.l
        self.S=None
        self.d=None
        self.init_var=1      

    def update(self,x):
        '''
        x: numpy array with the current observation
        '''
        if self.d is None:            
            self.d=x.size              
        # initialize S
        if self.S is None:
            self.S=self.init_var*np.eye(self.d)*self.v 
            self.m=np.zeros(self.d,dtype=float)
        # bayesian filter
        # predict
        self.v=self.phi*(self.v-self.d-1)+self.d+1        
        self.S=self.phi*self.S            
        self.k=self.phi*self.k
        # update
        self.S=self.S+(self.k/(self.k+1))*(x-self.m)[:,None]*(x-self.m)
        self.m=(self.k*self.m+x)/(1+self.k)        
        self.v+=1
        self.k+=1                                            
        return self
    
    def predict(self):        
        cov=self.S/(self.v-self.d-1) 
        mean=self.m                         
        return mean,cov
    
    def get_weight(self,x):
        mean,cov=self.update(x[-1]).predict()
        return np.dot(np.linalg.inv(cov),mean)
    
```

### Example with Cotton Futures


```python
data=pd.read_csv('daily_data.csv',index_col='Dates',parse_dates=True,infer_datetime_format=True)
x=data[['CT1 Comdty']].copy(deep=True)
x=x.resample('W').last()
x=x.pct_change()
x=x.dropna()
dates=x.index
x=x.values

phi=0.95
strategy=GaussianTrack(phi=phi)

s,l=wfbt(strategy,x)
plt.plot(dates,np.cumsum(s))
plt.grid(True)
plt.show()
plt.plot(dates,l)
plt.grid(True)
plt.show()

```


![png](/images/trackingd/output_6_0.png)

    



![png](/images/trackingd/output_6_1.png)

    


### Example with Bund Futures


```python
data=pd.read_csv('daily_data.csv',index_col='Dates',parse_dates=True,infer_datetime_format=True)
x=data[['RX1 Comdty']].copy(deep=True)
x=x.resample('W').last()
x=x.pct_change()
x=x.dropna()
dates=x.index
x=x.values

phi=0.95
strategy=GaussianTrack(phi=phi)

s,l=wfbt(strategy,x)
plt.plot(dates,np.cumsum(s))
plt.grid(True)
plt.show()
plt.plot(dates,l)
plt.grid(True)
plt.show()
```


![png](/images/trackingd/output_8_0.png)

    



![png](/images/trackingd/output_8_1.png)  


