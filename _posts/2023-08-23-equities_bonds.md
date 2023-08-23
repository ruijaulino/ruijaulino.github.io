# Equities and Bonds


Althouh there are many asset classes, a fundamental decision for many people/institutions is how to invest in stocks or bonds - or more generally, between growth and income. In the recent past we saw the demise of both (and many strategies that bet on both) with a fast raise in interest rates accompanied by a fall in value.

In general, this investment will be sequential: we allocate to both classes, get rewarded (or penalized) with a return and, finally, rebalance the exposure. Assuming a starting capital $S_0$, if $x_t$ is the vector of returns that the assets had from $t-1$ to $t$ then, proceding in this manner yield the following dynamics for the capital:


$S_n=S_0 \cdot (1+w x_1) \cdot (1+w x_2) \cdots (1+w x_n)$

where $w$ is a vector representing the fraction of the capital allocated to each asset (in general, $w$ can be different at each time step). Manipulating:

$S_n=S_0 \exp \left( n \frac{1}{n}\sum_i \log(1+w x_i) \right)$

From the previous expression we can identify the dynamics of the capital as geometric growth with growth rate $G=\frac{1}{n}\sum_i \log(1+w x_i)=\mathbb{E}(\log(1+wx))$.

A natural criteria for anyone is to maximize the growth rate of the capital. Expanding $\log(1+z)$ near $z=0$ yields:

$G=w\mu-\frac{1}{2}w^T\Sigma w$

which has as (unconstrained) maximum $w = \Sigma^{-1}\mu$. For the particular (and on focus here) case of two assets, the solution to the problem is:

$$\begin{bmatrix} w_1 \\\ w_2 \end{bmatrix} = \begin{bmatrix} \sigma_1^2 & \sigma_1 \sigma_2 \rho \\\ \sigma_1 \sigma_2 \rho & \sigma_2^2 \end{bmatrix}^{-1}  \begin{bmatrix} \mu_1 \\\ \mu_2 \end{bmatrix} = \frac{1}{\sigma_1\sigma_2(1-\rho^2)} \begin{bmatrix} \frac{\sigma_2}{\sigma_1}\mu_1-\rho\mu_2 \\\ \frac{\sigma_1}{\sigma_2}\mu_2-\rho\mu_1 \end{bmatrix}$$

One can observe that, in the absence of any correlation, $\rho \rightarrow 0$, the solution is equivalent to the Kelly solution for each individual asset; the maximum leverage increases with decreasing scales and correlations and the allocation to one of the assets can even turn negative for positive expected returns (depending on the correlation magnitude).


The objective here is to estimate the parameters of the previous model for appropriate benchmarks for stocks and bonds. With this, we want to see what does the data tell us about optimal allocations and, if possible, build models for their behaviour.

The benchmarks used are the SPY and IEF ETFs. Let us look at the data first.



```python
import numpy as np
import datetime as dt
import pandas as pd
import time
import matplotlib.pyplot as plt
```


```python
data=pd.read_csv('data.csv',index_col='Dates',parse_dates=True,infer_datetime_format=True)
data.plot()
plt.grid(True)
plt.show()
```


    
![png](/images/equities_bonds/output_2_0.png)
    



```python
data=data.pct_change()
data=data.dropna()
data.plot()
plt.grid(True)
plt.show()
```


    
![png](/images/equities_bonds/output_3_0.png)
    



```python
# function to compute the sharpe ratio
def sharpe(z,mult=np.sqrt(260)):
    return np.round(mult*np.mean(z)/np.std(z),2)
def annual_return(z,mult=260):
    return np.round(mult*np.mean(z),2)
def annual_vol(z,mult=np.sqrt(260)):
    return np.round(mult*np.std(z),2)
```

A first quantity we can compute is the Sharpe ratio $\text{SR}=\frac{\mu}{\sigma}$ (please recall the discussions on other posts - in particular Making Bets with Models - for more information); also let us see the annual returns and volatility for both assets.


```python
# individual statistics
r=data.values
print('SPY Annual return: ', annual_return(r[:,0]))
print('IEF Annual return: ', annual_return(r[:,1]))
print()
print('SPY Annual vol: ', annual_vol(r[:,0]))
print('IEF Annual vol: ', annual_vol(r[:,1]))
print()
print('SPY Sharpe: ',sharpe(r[:,0]))
print('IEF Sharpe: ',sharpe(r[:,1]))
```

    SPY Annual return:  0.11
    IEF Annual return:  0.03
    
    SPY Annual vol:  0.19
    IEF Annual vol:  0.07
    
    SPY Sharpe:  0.6
    IEF Sharpe:  0.51
    

Equities have a much higher annual return and volatility but the Sharpe ratios look similar; let us leave it for now.

## 60-40 Benchmark

A typical benchmark is to allocate 60% to stocks and 40% to bonds. Let us what this strategy in particular provides (as a note, here and anywhere in this post, I do not plot compounded returns for simplicity).


```python
# 60/40 portfolio
w_base=np.array([0.6,0.4])
r=data.values
s_base=np.sum(r*w_base,axis=1)
print('60/40 Annual return: ',annual_return(s_base))
print('60/40 Annual vol: ',annual_vol(s_base))
print('60/40 Sharpe: ',sharpe(s_base))
plt.plot(np.cumsum(s_base))
plt.grid(True)
plt.show()
```

    60/40 Annual return:  0.08
    60/40 Annual vol:  0.11
    60/40 Sharpe:  0.76
    


    
![png](/images/equities_bonds/output_8_1.png)
    


This strategy achieved a higher risk/return ratio; the annual return is decent and there are some large drawdowns. We can say that strategies of this type are assuming the joint distribution of the assets is constant over time. If this is the case we can try to estimate what a optimal allocation can be.

## Constant distribution

The most basic model is to assume that the distribution is constant across time instants. If this assumption is true we just need to estimate it's parameters (the problem now is a statistical one). Recall that we want to estimate the optimal allocation:

$w=\Sigma^{-1} \mu$

As a crude experiment (and trying to be predictive), we can use the first half of data to calculate $w$ and compute the equity curve with those weights.


```python
r=data.values # numpy (n,2) array with returns
r_=r[:int(r.shape[0]/2)]
mu=np.mean(r_,axis=0)
cov=np.cov(r_.T)
w_opt_first_half=np.dot(np.linalg.inv(cov),mu)
print('Optimal G weights [first half of data]: ', np.round(w_opt_first_half,2))
s_opt_first_half=np.sum(r*w_opt_first_half,axis=1)
print('Optimal G Sharpe [in all data]: ',sharpe(s_opt_first_half))
plt.plot(np.cumsum(s_opt_first_half))
plt.grid(True)
plt.show()
```

    Optimal G weights [first half of data]:  [ 4.74 17.72]
    Optimal G Sharpe [in all data]:  0.93
    


    
![png](/images/equities_bonds/output_11_1.png)
    


Let us forget for now that the weights can be very large and are not feasible for practical use, and  the estimated values are quite diferent (also in proportion) to the traditional 60/40. Also, the sharpe is higher (although we are usign the training data as part of the test).

Having a single estimate is not of much use. One way to use the data we have is to make a bootstrap estimate of the optimal weights. 


```python
# bootstrap estimate of allocation for optimal growth rate
n_boot=1000
r=data.values
idx=np.arange(r.shape[0],dtype=int)
w_opt_samples=np.zeros((n_boot,2))
for j in range(n_boot):    
    # bootstrap from r
    idx_=np.random.choice(idx,size=idx.size,replace=True)
    r_=r[idx_]    
    mu=np.mean(r_,axis=0)
    cov=np.cov(r_.T)
    w_opt_samples[j]=np.dot(np.linalg.inv(cov),mu)
# average optimal weight
w_opt_boot=np.mean(w_opt_samples,axis=0)
plt.title('SPY weight bootstrap distribution')
plt.hist(w_opt_samples[:,0],density=True,bins=30,alpha=0.5)
plt.axvline(w_opt_boot[0],label='Mean SPY weight')
plt.legend()
plt.grid(True)
plt.show()
plt.title('IEF weight bootstrap distribution')
plt.hist(w_opt_samples[:,1],density=True,bins=30,alpha=0.5)
plt.axvline(w_opt_boot[1],label='Mean IEF weight')
plt.legend()
plt.grid(True)
plt.show()
plt.title('Ratio of IEF to SPY weight bootstrap distribution')
plt.hist(w_opt_samples[:,1]/w_opt_samples[:,0],density=True,bins=30,alpha=0.5)
plt.axvline(w_opt_boot[1]/w_opt_boot[0],label='Mean ratio IEF/SPY weight')
plt.legend()
plt.grid(True)
plt.show()
print('Optimal ratio of weights [bootstrap]: ', np.round(w_opt_boot[1]/w_opt_boot[0],2))
print('Optimal weights [bootstrap]: ', np.round(w_opt_boot,2))
```


    
![png](/images/equities_bonds/output_13_0.png)
    



    
![png](/images/equities_bonds/output_13_1.png)
    



    
![png](/images/equities_bonds/output_13_2.png)
    


    Optimal ratio of weights [bootstrap]:  2.67
    Optimal weights [bootstrap]:  [ 4.47 11.93]
    

Now we have a clearer picture of what are the optimal weights and their distribution; they vary a lot and their means are not (by chance probably) that different that what we calculated with the first half of data. 

Another quantity of interest is of ratio of weight in IEI to the weight in SPY: we can see that the weight in IEI is $\approx 2.7$ times the weight in SPY (and the distribution is quite concentrated around this value).

### Normalization

To make the weights feasible, a possible approach is to normalize them to unity. Trivially, the multiplication by a constant does not change the Sharpe ratio but will change the original geometric growth rate:

$G=w^T\mu - \frac{1}{2}w^T\Sigma w$

into a smaller one:

$G_k = k w^T\mu - k ^2 \frac{1}{2}w^T\Sigma w$

Note that $0 \le k \leq 1$ .

If we consider as a measure of _risk_ the amount of money bet, we can say that, when we multiply $w$ by $k$ we are reducing our exposure by a factor $k$; of course, our growth rate is going to be lower because we bet a smaller amount. If we compare how the growth rate is reduced in proportion to how the exposure was reduced is it possible to conclude that it's reduction is smaller as:

$\frac{G_k}{G} \ge \frac{k}{1} \rightarrow -k^2 w^T\Sigma w + k w^T\Sigma w \ge 0$

(where we wrote the inequality to the side where it means that the reduction in growth rate was lower than the reduction in exposure) which is true for any $0 \le k \le 1$. This shows that we can have a smaller growth rate reduction in proportion to what was reduced in the exposure and justifies that having a fractional Kelly bet is an interesting decision.

A natural factor $k$ is one such that the sum of weights (absolute value) is equal to a desired leverage (for example, no leverage); note that, since the weights are large we can do this without problem: if the weights were small then, the act of multiplying by a constant to get a desired leverage, could lead to a negative growth rate. Doing this for the previously calculated bootstrap weights:


```python
w_opt_boot_norm=np.array(w_opt_boot)
w_opt_boot_norm/=np.sum(np.abs(w_opt_boot_norm))
print('Optimal weights normalized [bootstrap]: ', np.round(w_opt_boot_norm,2))
```

    Optimal weights normalized [bootstrap]:  [0.27 0.73]
    

As discussed in previously, this weights, properly levered achieve the optimal growth rate.

### Backtest

The previous analysis does not tell us nothing about what we can expect from such strategy - we are just using the data to estimate the parameters. We use cross-validation to estimate the out-of-sample performance of such strategy; dividing the data into training and testing sets, we estimate the weights on the training data (using the same procedure by considering the average bootstrap optimal weight), normalize them and bet with them on the testing set.


```python
# estimate with cross-validation
n_boot=1000
k_folds=20 # this value was set to a larger than usual one so we can see the
           # how spread the weights are across folds
r=data.values
n=r.shape[0]
idx=np.arange(n,dtype=int)
idx_folds=np.array_split(idx,k_folds)
s_cv=np.zeros(n)
folds_w=np.zeros((k_folds,2))
for i in range(k_folds):
    train_idx=np.where(~np.in1d(idx,idx_folds[i]))[0]    
    test_idx=idx_folds[i]
    # estimate weight by bootstrap average
    tmp=np.zeros((n_boot,2))
    for j in range(n_boot):    
        idx_=np.random.choice(train_idx,size=train_idx.size,replace=True)
        r_train_=r[idx_]    
        mu=np.mean(r_train_,axis=0)
        cov=np.cov(r_train_.T)
        tmp[j]=np.dot(np.linalg.inv(cov),mu)
    # average optimal weight
    w=np.mean(tmp,axis=0)    
    w/=np.sum(np.abs(w)) # normalize
    folds_w[i]=w
    s_cv[test_idx]=np.sum(r[test_idx]*w,axis=1)  

print('Optimal G Sharpe [Cross-Validation]: ',sharpe(s_cv))
print('Optimal G Annual return [Cross-Validation]: ',annual_return(s_cv))
print('Optimal G Annual vol [Cross-Validation]: ',annual_vol(s_cv))
plt.title('CV Strategy')
plt.plot(np.cumsum(s_cv))
plt.grid(True)
plt.show()
plt.title('Training sets weights distribution')
plt.hist(folds_w[:,0],alpha=0.5,density=True,label='Training SPY weights')
plt.hist(folds_w[:,1],alpha=0.5,density=True,label='Training IEF weights')
plt.legend()
plt.grid(True)
plt.show()
```

    Optimal G Sharpe [Cross-Validation]:  0.87
    Optimal G Annual return [Cross-Validation]:  0.05
    Optimal G Annual vol [Cross-Validation]:  0.06
    


    
![png](/images/equities_bonds/output_18_1.png)
    



    
![png](/images/equities_bonds/output_18_2.png)
    


Shapewise, we see a large improvement over the 60/40; the annual return is 5% compared to the 8% of the benchmark but comes with a large reduction in the scale of the fluctuations. The weights are quite stable over training sets.

With those weights one can leverage and prop up the returns (although it will expose us to more risks and exacerbate the model/estimation errors that we may make here).

In terms of modelling as constant distribution across time instants that can be levered, this seems to be what the data tell us to do.

### Relation to Risk Parity
The previous weights are similar to the ones one would obtain if we do the so called _risk parity_ approach - allocation proportional to the inverse scale of the distribution (compare with the weights obtained by bootstrap; the cross validation part was to assess out of sample performance):


```python
# let just do this with all data for simplicity (we could also do a bootstrap to estimate this)
scale=np.std(r,axis=0)
w_rp_all_data=1/scale
w_rp_all_data/=np.sum(np.abs(w_rp_all_data))
print('Risk parity weights [all data]: ', np.round(w_rp_all_data,2))
```

    Risk parity weights [all data]:  [0.26 0.74]
    

We can and should ask why are those values so close to each other; what properties does the data has that explains this? Can we use that to make a model?

If we consider that the Sharpe ratio is constant and the same for equities and bonds then

$\text{SR}=\frac{\mu}{\sigma}=\text{const}$

and the optimal weights become (proportional to):

$w \propto \Sigma^{-1} \vec{\sigma}$

Solving the two dimensional system:

$$\begin{bmatrix} w_1 \\\ w_2 \end{bmatrix} \propto \begin{bmatrix} \sigma_1^2 & \sigma_1 \sigma_2 \rho \\\ \sigma_1 \sigma_2 \rho & \sigma_2^2 \end{bmatrix}^{-1}  \begin{bmatrix} \sigma_1 \\\ \sigma_2 \end{bmatrix}$$

We get that the optimal weights are proportional to the inverse scales:

$$\begin{bmatrix} w_1 \\\ w_2 \end{bmatrix} \propto \frac{1}{1+\rho} \begin{bmatrix} \frac{1}{\sigma_1} \\\ \frac{1}{\sigma_2} \end{bmatrix}$$

and the amount of leverage that can be applied increases with a decrese in correlation.

Given the previous result we can conclude that it seem that both sharpe ratios are equal. Let us estimate the Sharpes with a bootstrap and compare the distributions to check if we have evidence of this.


```python
# bootstrap estimate of assets sharpe ratios
n_boot=1000
r=data.values
idx=np.arange(r.shape[0],dtype=int)
sr_samples=np.zeros((n_boot,2))
for j in range(n_boot):    
    # bootstrap from r
    idx_=np.random.choice(idx,size=idx.size,replace=True)
    r_=r[idx_]    
    mu=np.mean(r_,axis=0)
    scale=np.std(r_,axis=0)
    sr_samples[j]=np.sqrt(260)*mu/scale  
plt.hist(sr_samples[:,0],density=True,bins=30,alpha=0.5,label='SPY SR Distribution')
plt.hist(sr_samples[:,1],density=True,bins=30,alpha=0.5,label='IEF SR Distribution')
plt.legend()
plt.grid(True)
plt.show()
plt.hist(sr_samples[:,0]-sr_samples[:,1],density=True,bins=30,alpha=0.5,label='Difference Distribution')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](/images/equities_bonds/output_23_0.png)
    



    
![png](/images/equities_bonds/output_23_1.png)
    


Although the SPY has a higher sharpe, the distributions overlap quite a lot; looking at the distribution of Sharpe ratio difference we cannot reject that the difference has mean zero (we just have to eyeball it).

It seems that we can explain the calculated optimal weights with the equality in sharpe ratios; also, since we normalized the weights then the correlations do not matter (we can say that they control the maximum leverage but we are not interested in this as the this value is too large).

### We cannot eat risk-adjusted returns

All the previous analysis assume we are able to leverage our exposure to some point and, if that is the case, the previous analysis is the optimal solution - as seens in other posts, the optimal sharpe portfolio can be levered to achieve the optimal growth rate; one could achieve that using futures for example. We became also more exposed to estimation (and model) errors which, under leverage, can lead to catastrophic results.

All this is important because, even if we achieve a _nice_ risk-adjusted result, in the end we may end up not making enough money for the goals. 

As a complement, let us repeat the analysis to the constrained case ($\sum w_i = 1$) of no leverage and find the growth optimal portfolios.



```python
# eval growth rate
def calc_gr(mu,cov,alpha):
    '''
    alpha: proportion in asset 1
        the other has 1-alpha
    '''
    w=np.array([alpha,1-alpha])
    return np.dot(mu,w)-0.5*np.dot(w,np.dot(cov,w))

# brute force to find optimal growth rate
def optimal_gr(r,n_div=100,view=False):
    alphas=np.linspace(0,1,n_div)
    gr=np.zeros(n_div)
    # precompute the statistics
    mu=np.mean(r,axis=0)
    cov=np.cov(r.T)
    for i in range(n_div):
        gr[i]=calc_gr(mu,cov,alphas[i])
    if view:
        plt.title('Growth rate')
        plt.plot(alphas,gr)
        plt.grid(True)
        plt.xlabel('Proportion in asset SPY')
        plt.show()
    return alphas[np.argmax(gr)]

# bootstrap estimate of allocation for optimal growth rate
n_boot=1000
r=data.values
idx=np.arange(r.shape[0],dtype=int)
alpha_opt_samples=np.zeros(n_boot)
for j in range(n_boot):    
    # bootstrap from r
    idx_=np.random.choice(idx,size=idx.size,replace=True)
    r_=r[idx_]    
    alpha_opt_samples[j]=optimal_gr(r_,n_div=500,view=False)
# average optimal proportion in SPY
alpha_opt_boot=np.mean(alpha_opt_samples)
print('Constrained optimal weight in SPY [bootstrap]: ', np.round(alpha_opt_boot,2))
plt.title('SPY weight bootstrap distribution')
plt.hist(alpha_opt_samples,density=True,bins=30,alpha=0.5)
plt.axvline(alpha_opt_boot,label='Mean SPY weight')
plt.legend()
plt.grid(True)
plt.show()
```

    Constrained optimal weight in SPY [bootstrap]:  0.89
    


    
![png](/images/equities_bonds/output_26_1.png)
    


The allocation is quite concentrated in equities but that is expected as there is the constraint in leverage and, to achieve a higher growth rate (and due to the fact that daily fluctuations in the sample are not that wild), we have to allocate much more to the asset class that grows more. We can also say that, without the possiblity to leverage the exposure (of if we do not want to have the model risk), just buy equities.

In the same fashion, let us try to evaluate the out of sample performance of this strategy through cross-validation.


```python
# estimate with cross-validation
n_boot=1000
k_folds=20
r=data.values
n=r.shape[0]
idx=np.arange(n,dtype=int)
idx_folds=np.array_split(idx,k_folds)
sc_cv=np.zeros(n)
folds_w=np.zeros((k_folds,2))
for i in range(k_folds):
    train_idx=np.where(~np.in1d(idx,idx_folds[i]))[0]    
    test_idx=idx_folds[i]
    # estimate weight by bootstrap average
    tmp=np.zeros(n_boot)
    for j in range(n_boot):    
        idx_=np.random.choice(train_idx,size=train_idx.size,replace=True)
        r_train_=r[idx_]    
        tmp[j]=optimal_gr(r_train_,n_div=500,view=False)
    # average optimal weight
    spy_w=np.mean(tmp)
    w=np.array([spy_w,1-spy_w])
    folds_w[i]=w
    sc_cv[test_idx]=np.sum(r[test_idx]*w,axis=1)  

print('Constrained optimal G Sharpe [Cross-Validation]: ',sharpe(sc_cv))
print('Constrained optimal G Annual return [Cross-Validation]: ',annual_return(sc_cv))
print('Constrained optimal G Annual vol [Cross-Validation]: ',annual_vol(sc_cv))

plt.title('CV Constrained Strategy')
plt.plot(np.cumsum(sc_cv))
plt.grid(True)
plt.show()
plt.title('Training sets weights distribution')
plt.hist(folds_w[:,0],alpha=0.5,density=True,label='Training SPY weights')
plt.hist(folds_w[:,1],alpha=0.5,density=True,label='Training IEF weights')
plt.legend()
plt.grid(True)
plt.show()
```

    Constrained optimal G Sharpe [Cross-Validation]:  0.57
    Constrained optimal G Annual return [Cross-Validation]:  0.1
    Constrained optimal G Annual vol [Cross-Validation]:  0.17
    


    
![png](/images/equities_bonds/output_28_1.png)
    



    
![png](/images/equities_bonds/output_28_2.png)
    


Not a fantastic result, but we have a annual return and sharpe similar to the SPY and a reduction in volatility (it is what it is: no leverage, no tricks to grow the capital faster rather than investing in the one that grows more).

### Validity of the approximation

The optimal growth problem was formulated with a second order approximation to the growth rate. Let us repeat the previous analysis by considering the full (constrained) problem: maximization of 

$$\mathbb{E}(\log(1+wx))$$




```python
# eval growth rate
def calc_gr_full(r,alpha):
    '''
    alpha: proportion in asset 1
        the other has 1-alpha
    '''
    w=np.array([alpha,1-alpha])
    return np.mean(np.log(1+r*w))

# brute force to find optimal growth rate
def optimal_gr_full(r,n_div=100,view=False):
    alphas=np.linspace(0,1,n_div)
    gr=np.zeros(n_div)
    for i in range(n_div):
        gr[i]=calc_gr_full(r,alphas[i])
    if view:
        plt.title('Growth rate')
        plt.plot(alphas,gr)
        plt.grid(True)
        plt.xlabel('Proportion in asset SPY')
        plt.show()
    return alphas[np.argmax(gr)]

# bootstrap estimate of allocation for optimal growth rate
n_boot=1000
r=data.values
idx=np.arange(r.shape[0],dtype=int)
alpha_opt_samples=np.zeros(n_boot)
for j in range(n_boot):    
    # bootstrap from r
    idx_=np.random.choice(idx,size=idx.size,replace=True)
    r_=r[idx_]    
    alpha_opt_samples[j]=optimal_gr_full(r_,n_div=500,view=False)
    
# average optimal proportion in SPY
alpha_opt_boot=np.mean(alpha_opt_samples)
print('FULL PROBLEM Constrained optimal weight in SPY [bootstrap]: ', np.round(alpha_opt_boot,2))
plt.title('FULL PROBLEM SPY weight bootstrap distribution')
plt.hist(alpha_opt_samples,density=True,bins=30,alpha=0.5)
plt.axvline(alpha_opt_boot,label='Mean SPY weight')
plt.legend()
plt.grid(True)
plt.show()
```

    FULL PROBLEM Constrained optimal weight in SPY [bootstrap]:  0.89
    


    
![png](/images/equities_bonds/output_31_1.png)
    


We obtained a similar result and so the approximation is good enough.

### Notes

Staring from the data and solving for the optimal resource allocation problem, by observation, the _risk parity_ idea is consistent with the data. Of course this assumes a static world where the distribution does not change; also, the popular notion of leveraging to prop up the returns also makes sense as the optimal allocation is quite large - of course there is a great deal of variation over parameters (just look at the histograms of optimal weights) but on average this can make sense.


If no leverage, just buy equities of put a small allocation to bonds; otherwise rebalance between $\approx 30/70$ with some leverage. Of couse, nothing will prevent you to loose money when both fall in value.

## Changing distribution

In practice it is rare to see having a fixed proportion in each asset; given the recent past it is _thought_ desirable to update the parameters with most recent data. For example, one can say that the allocation is given by

$w_i=\frac{1/\sigma_i^t}{\sum_j 1/\sigma_j^t}$

where $\sigma_i^t$ is, for example, the standard deviation of the past $l$ returns for asset $i$. Examples and opinions appart, those procedures assume that the parameters change over time. We can formalize the idea as having a return vector $x_t$ at instant $t$ that was generated from some distribution with parameters $\mu_t,\Sigma_t$ (let us characterize this distribution with the first and second moment for simplicity and in an obvious connection to the Gaussian - even being a wrong one). In order to build a model we need to specify how and why the parameters (the unobserved - only estimated - states) vary over time. The most simple idea we can think of is to assume that they depend on the imediate previous values. With this:

$x_t \sim p(x_t\|\mu_t,\Sigma_t)$

$\mu_t,\Sigma_t \sim p(\mu_t,\Sigma_t\|\mu_{t-1},\Sigma_{t-1})$

Future returns are difficult to estimate and we can take advantage of the observation that the Sharpe ratios tend to be similar and use this in our model; if this is the case we only need to estimate a time-varying covariance (then $\mu_i \propto \sigma_i = \sqrt{\Sigma_{ii}}$). A model like this can be estimated in a simple manner by just computing the covariance with the previous $l$ observations and the optimal allocation follows by $w \propto \hat{\Sigma}^{-1} \hat{\sigma}$. Following the ideas of how to make a bet with a model we cannot renormalize the weights at each time step to a given level of leverage or we do not achieve the best sharpe ratio (this idea is similar to having a target risk of the portfolio where the leverage is allowed to fluctuate but still not optimal in the sense of the idea of having a changing distribution).

As a first test, let run a walk forward estimation of performance for this type of idea and test what happens when we use the full covariance versus just usign the rolling standard deviation and the effect of normalization of weights to unit leverage. 


```python
# function to perform a walk forward backtest
def bt(model,x):
    '''
    model: class with a method get_weight
    x: numpy (n,p) array with returns
    '''    
    n=x.shape[0]
    p=x.shape[1]
    s=np.zeros(n,dtype=float)
    leverage=np.zeros(n,dtype=float)
    w=np.zeros(p,dtype=float)
    for i in range(n):
        leverage[i]=np.sum(np.abs(w))
        s[i]=np.dot(w,x[i])
        w=model.get_weight(x[:i+1])
    return s,leverage
```


```python
class RP(object):
    def __init__(self,l=20,normalize=False,full_cov=False):
        self.l=l
        self.normalize=normalize
        self.full_cov=full_cov
    def get_weight(self,x):
        if x.shape[0]<=self.l:
            return np.zeros(x.shape[1])
        else:
            if self.full_cov:
                cov=np.cov(x[-self.l:].T)
                scale=np.sqrt(np.diag(cov))
                w=np.dot(np.linalg.inv(cov),scale)           
            else:
                scale=np.std(x[-self.l:],axis=0)
                w=1/scale
            if self.normalize:
                w/=np.sum(np.abs(w))
            return w
```


```python
x=data.values
l=20
model_rp1=RP(l=l,normalize=False)
model_rp2=RP(l=l,normalize=True)
model_rp3=RP(l=l,normalize=False,full_cov=True)
model_rp4=RP(l=l,normalize=True,full_cov=True)
s_rp1,lev_rp1=bt(model_rp1,x)
s_rp2,lev_rp2=bt(model_rp2,x)
s_rp3,lev_rp3=bt(model_rp3,x)
s_rp4,lev_rp4=bt(model_rp4,x)
print('Sharpe simple RP [Walk-Forward]: ',sharpe(s_rp1))
print('Sharpe simple constrained RP [Walk-Forward]: ',sharpe(s_rp2))
print()
print('Sharpe simple RP Covariance [Walk-Forward]: ',sharpe(s_rp3))
print('Sharpe simple constrained RP Covariance [Walk-Forward]: ',sharpe(s_rp4))

plt.title('[Normalized] Equity curves of simple RP strategies')
plt.plot(np.cumsum(s_rp1/np.std(s_rp1)),label='Unconstrained')
plt.plot(np.cumsum(s_rp2/np.std(s_rp2)),label='Constrained')
plt.plot(np.cumsum(s_rp3/np.std(s_rp3)),label='Cov Unconstrained')
plt.plot(np.cumsum(s_rp4/np.std(s_rp4)),label='Cov Constrained')
plt.legend()
plt.grid(True)
plt.show()
```

    Sharpe simple RP [Walk-Forward]:  1.04
    Sharpe simple constrained RP [Walk-Forward]:  0.99
    
    Sharpe simple RP Covariance [Walk-Forward]:  1.26
    Sharpe simple constrained RP Covariance [Walk-Forward]:  0.99
    


    
![png](/images/equities_bonds/output_37_1.png)
    


The first conclusion is that the sharpe ratios are higher for non-constrained solutions and are higher that those obtained previously by considering a constant allocation - this lead us to the conclusion that time-varying covariances model better the data. Also, the unconstrained solution with the full covariance has a considerable higher sharpe than the others - we can attribute this to the change in leverage that is created due to the time-varying nature of correlation (when we normalized to unit leverage the advantage disapeared and so, given the theoretical solution to the problem, we can make this association); also, this model had a much lower drawdown when both assets feel recently - they becase quite correalated and the ammount invested was reduced. 

Of course, to compare the equity curves we need to normalize the returns of the strategies because, it the unconstrained ones, the weights can vary a lot (also, non of those are _valid_ strategies in the sense that we need to estimate the bounds of the weights to proper normalize them).

Since the idea seems to improve the results, we need to find a way to estimate the out-of-sample performance. In particular, we need to estimate a bound to the weights and also optimize the lookback window. We can use the cross validation idea to run the model on a training set without any constraint and measure a bound to weights that we are expecting to see; then we evaluate on the testing set with the weights normalized (divided) by this bound. 

A practical way to choose a bound is to assume that the weights sometime will exceed the maximum leverage and try to match how many times we are expecting this to happen. The following code illustrates this ideas (the lookback window is not optimized - more on that later).


```python
# function to perform a walk forward/cross validation backtest
def cvbt(model,x,k_folds=5):
    '''
    model: class with a method get_weight
    x: numpy (n,p) array with returns
    '''    
    n=x.shape[0]
    p=x.shape[1]
    idx=np.arange(n,dtype=int)
    idx_folds=np.array_split(idx,k_folds)    
    s=np.zeros(n,dtype=float)
    leverage=np.zeros(n,dtype=float)
    for i in range(k_folds):
        train_idx=np.where(~np.in1d(idx,idx_folds[i]))[0]    
        test_idx=idx_folds[i]    
        x_test=x[test_idx]
        x_train=x[train_idx]
        # estimate model
        model.estimate(x_train)
        w=np.zeros(p)

        for j in range(x_test.shape[0]):
            leverage[test_idx[j]]=np.sum(np.abs(w))
            s[test_idx[j]]=np.dot(w,x_test[j])
            w=model.get_weight(x_test[:j+1])
    return s,leverage
```


```python
# let us reformulate the previous class
class RP(object):
    
    def __init__(self,l=20,normalize=False,full_cov=False,max_lev=1,quantile=0.9):
        self.l=l
        self.normalize=normalize
        self.full_cov=full_cov
        self.max_lev=max_lev
        self.lev_norm=1
        self.quantile=quantile
        
    def estimate(self,x):
        n=x.shape[0]
        p=x.shape[1]
        s=np.zeros(n,dtype=float)
        lev=np.zeros(n,dtype=float)
        w=np.zeros(p,dtype=float)
        for i in range(n):
            lev[i]=np.sum(np.abs(w))
            s[i]=np.dot(w,x[i])
            w=self.get_weight_base(x[:i+1])          
        lev.sort()
        self.lev_norm=lev[int(self.quantile*lev.size)]

    def get_weight_base(self,x):
        if x.shape[0]<=self.l:
            return np.array([0.3,0.7])
        else:
            if self.full_cov:
                cov=np.cov(x[-self.l:].T)
                scale=np.sqrt(np.diag(cov))
                w=np.dot(np.linalg.inv(cov),scale)           
            else:
                scale=np.std(x[-self.l:],axis=0)
                w=1/scale
        return w
    
    def get_weight(self,x):
        w=self.get_weight_base(x)
        if self.normalize:
            w/=np.sum(np.abs(w))
        w/=self.lev_norm
        if np.sum(np.abs(w))>self.max_lev:
            w/=np.sum(np.abs(w)) 
            w*=self.max_lev
        return w
```


```python
x=data.values
l=20
model_rp5=RP(l=l,full_cov=True)
s_rp5,lev_rp5=cvbt(model_rp5,x,k_folds=5)
print('Sharpe simple RP Covariance [Cross-Validation]: ',sharpe(s_rp5))
print('Annual return simple RP Covariance [Cross-Validation]: ',annual_return(s_rp5))
print('Annual vol simple RP Covariance [Cross-Validation]: ',annual_vol(s_rp5))
plt.title('Equity curve')
plt.plot(np.cumsum(s_rp5))
plt.grid(True)
plt.show()
plt.title('Leverage')
plt.plot(lev_rp5)
plt.show()
```

    Sharpe simple RP Covariance [Cross-Validation]:  1.22
    Annual return simple RP Covariance [Cross-Validation]:  0.03
    Annual vol simple RP Covariance [Cross-Validation]:  0.03
    


    
![png](/images/equities_bonds/output_42_1.png)
    



    
![png](/images/equities_bonds/output_42_2.png)
    


The annual return is quite low but most of time we have many cash to the sides - if one has the capability to leverage a bit this can be a nice strategy.

What is more relevant to a practical application is bounding the weights to a feasible set and that exercise is independent of _how_ (under which criteria) a lookback windows is optimized - should we choose the one that yields the higher sharpe? This question is ill defined because, in reality, what we did is a quite phenomenological approach and we did not specify a model on how stuff changes; without that it is difficult/ambiguous to define what is a good fit. Let us try to answer that in the next section.

#### A state space model to estimate Covariance

Considering models like the ones described above, a natural question is how can we estimate a covariance that changes.

We have a hidden/unobserved variable $\Sigma_k$ (the covariance); under this covariance, multivariate normal returns $x_k$ with zero mean are generated. Also, the next covariance depends on the previous one. In the end, this is a model that describes a sequence of observarions with heteroscedastic noise.

The model consists on a sequence of conditional distributions (state space model; the state is the hidden covariance):

$\Sigma_k \sim p(\Sigma_k \| \Sigma_{k-1})$

$x_k \sim p(x_k \| \Sigma_k)$

Under this model, we want to find a algorithm to estimate $p(\Sigma_k \| x_{0:k})$ (this means the distribution of current covariance, $\Sigma_k$, given all observations of $x_i$ until now). With this and the transition mechanism, $p(\Sigma_k \| \Sigma_{k-1})$, we can predict the next covariance given the observations that we made until now.

First, assume that at some instant $k-1$ we have a distribution for the covariance given all the observations until now - $p(\Sigma_{k-1} \| x_{0:k-1})$ - and we know it's parameters. A suitable form for this distribution is a Inverse-Wishart distribution. Let us write this as:

$p(\Sigma_{k-1} \| x_{0:k-1}) = \text{IW}(\nu_{k-1},S_{k-1})$

where $\nu_{k-1}$ is the degree of freedom parameter and $S_{k-1}$ is the scale matrix, both at instant $k-1$ (again, assume these parameters are known).

Now we can use the Bayes theorem to write:

$p(\Sigma_k \| x_{0:k}) \propto p(x_k \| \Sigma_k) p(\Sigma_k\|x_{0:k-1})$

Now, $p(x_k \| \Sigma_k) = \text{N}(x_k\|0,\Sigma_k)$ (it is just a normal distribution with covariance $\Sigma_k$). If $p(\Sigma_k\|x_{0:k-1})$ is a Inverse-Wishart with some parameters, then (since it is conjugate to the Normal) $p(\Sigma_k \| x_{0:k})$ will also be Inverse-Wishart with some other parameters.

The other term is more involved. In terms of known quantities we can write it as (Chapman-Kolmogorov equation):

$p(\Sigma_k\|x_{0:k-1})=\int p(\Sigma_k\|\Sigma_{k-1}) p(\Sigma_{k-1}\|x_{0:k-1}) d\Sigma_{k-1}$

In general, it is difficult to build a distribution $p(\Sigma_k\|\Sigma_{k-1})$ such that, when multiplied by an Inverse-Wishart ( $p(\Sigma_{k-1}\|x_{0:k-1})$ ) and integrated produces another Inverse-Wishart (which would simplify the calculations from what we saw above). Anyway, it is reasonable to say that $p(\Sigma_k\|\Sigma_{k-1})$ has such a form that $p(\Sigma_k\|x_{0:k-1})$ is a Inverse-Wishart with parameters $\nu_k^-$ and $S_k^-$ (where these parameters are functions on the previously known ones, $\nu_{k-1}$ and $S_{k-1}$).

It makes sense to consider that our prediction (expected value) for covariance is the same as it was before (we have no new information and so we cannot add anything) but has a higher uncertainty (larger variance). The following new parameters keep $\mathbf{E}[\Sigma_k]=\mathbf{E}[\Sigma_{k-1}]$ but increases the uncertainty.

$\nu_k^- = \phi(\nu_{k-1}-d-1)+d+1$

$S_k^- = \phi S_{k-1}$

with $\phi \in (0,1)$. 

Going back to the estimate of $\Sigma_k$ we can write:

$p(\Sigma_k \| x_{0:k}) \propto p(x_k \| \Sigma_k) p(\Sigma_k\|x_{0:k-1}) = \text{IW}(\nu_k^- + 1, S_k^- + \vec{x_k}\vec{x_k}^T)$

and so $p(\Sigma_k \| x_{0:k})$ is a Inverse-Wishart with updated parameters $\nu_k^- + 1$ and $S_k^- + \vec{x_k}\vec{x_k}^T$.

This model can be interpreted as estimating the current covariance with the past observation with exponentially decaying weights (with the decay controled by parameters $\phi$).

##### Parameter estimation

The previous model depends on a single parameters $\phi$ that we can interpret as a _lookback window_ to calculate the covariance. For a set of data (sequence $x_{0:T}$) we calculate it's optimal value by considering:

$p(\phi\|x_{0:T}) \propto p(x_{0:T}\|\phi) p(\phi)$

We can write the sequence probability as

$p(x_{0:T}\|\phi) = \Pi_{k=0}^T p(x_k\|x_{0:k-1},\phi)$

which can be interpreted as a _prediction error_. The value that maximizes the probability of future unseen data should be optimal.

The posterior predictive distribution when $x_k$ is normal and the covariance is Inverse-Wishart is a (multivariate) T-distribution. 

$p(x_k\|x_{0:k-1},\phi)=\text{T}(x_k \| S_k^-/(\nu_k^- -d +1),\nu_k^- -d +1)$

A more convinient way to consider the problem is to write

$\log p(\phi\|x_{0:T}) \propto \log p(x_{0:T}\|\phi) \log p(\phi) = \log p(\phi) + \sum_{k=0}^T \log p(x_k\|x_{0:k-1},\phi)$

To optimize $\phi$ we just run for a part of the data with some values and find the best one. 


Basically, this model is similar to estimate with the previous values but now they are weighted exponentially (this is usually seen done in practice and so this is a way to formalize those ideas); more important, given that the model is defined, we can fit it by maximum likelihood, eliminating the ambiguity in the _how_ should the parameters be optimized.


As was done before, let us do first a simple test with some parameters to get some intuition about improvements (or not) that we can get doing this.


```python
# Covariance tracking estimator strategy
class CovTrack(object):
    def __init__(self,l=20,phi=0.95):
        self.l=l 
        self.phi=phi
        self.v=l
        self.S=None
        self.d=None

    def get_weight(self,x):        
        w=np.zeros(x.shape[1])
        n=x.shape[0]
        if self.d is None:
            self.d=x.shape[1]                
        if n>self.l:
            # initialize
            if self.S is None:
                self.S=np.cov(x[:-1].T)*self.v           
            # bayesian filter
            # predict
            self.v=self.phi*(self.v-self.d-1)+self.d+1
            self.S=self.S*self.phi
            # update
            self.v+=1
            self.S+=x[-1][:,None]*x[-1]            
            # make allocatiom
            cov=self.S/(self.v-self.d-1)           
            er=np.sqrt(np.diag(cov))                  
            w=np.dot(np.linalg.inv(cov),er)
        return w    
```


```python
l=20
x=data.values
model_ct=CovTrack(l=l,phi=0.9)
s_ct,lev_ct=bt(model_ct,x)
print('Sharpe CovTrack [Walk-Forward]: ',sharpe(s_ct))
plt.title('[Normalized] Equity curves of CovTract strategy')
plt.plot(np.cumsum(s_ct/np.std(s_ct)))
plt.grid(True)
plt.show()
```

    Sharpe CovTrack [Walk-Forward]:  1.38
    


    
![png](/images/equities_bonds/output_47_1.png)
    


At first we can see an improvement over the previous results; it seems that exponential weighting the observations fits better to the data. 

Finally, let us use (again) cross-validation to estimate the out-of-sample performance of the model; now we optimize also $\phi$ in each training set and calculate a bound to the weights.



```python
# let us reformulate the previous class
# Covariance tracking estimator strategy
from scipy.stats import multivariate_t

class CovTrack(object):
    def __init__(self,l=20,phi=0.95,max_lev=1,quantile=0.9,n_div=5):
        self.l=l 
        self.phi=phi
        self.v=l
        self.max_lev=max_lev
        self.lev_norm=1
        self.n_div=n_div
        self.quantile=quantile        
        self.S=None
        self.d=None

    def estimate(self,x):
        self.S=None
        self.d=None    
        n=x.shape[0]
        p=x.shape[1]
        phi_values=np.linspace(0.7,0.99,self.n_div)
        prob_values=np.zeros(self.n_div)
        v_values=self.l*np.ones(self.n_div)
        S_values=np.zeros((self.n_div,p,p)) 
        logp=np.zeros(self.n_div)
        lev=np.zeros((n,self.n_div))
        is_init=False
        if self.d is None:
            self.d=x.shape[1]         
        
        for i in range(n):
            if i>self.l:
                # initialize
                if not is_init:
                    aux=np.cov(x[:-1].T)*self.l  
                    S_values[:]=aux
                    is_init=True
                # bayesian filter
                # predict
                v_values=phi_values*(v_values-self.d-1)+self.d+1
                S_values=S_values*phi_values[:,None,None]
                # store leverage to compute the bound                
                # and
                # compute logp of observation based on predicted parameters
                logp_tmp=np.zeros(self.n_div)
                for j in range(self.n_div):
                    cov=S_values[j]/(v_values[j]+1-self.d)          
                    er=np.sqrt(np.diag(cov))                        
                    lev[i,j]=np.sum(np.abs(np.dot(np.linalg.inv(cov),er)))
                    logp_tmp[j]=np.log(multivariate_t.pdf(x[i],
                                               loc=np.zeros(self.d),
                                               shape=cov,
                                               df=(v_values[j]+1-self.d)))
                logp+=logp_tmp
                # update
                v_values+=1
                S_values[:]+=x[i][:,None]*x[i]
        self.phi=phi_values[np.argmax(logp)]
        lev=lev[:,np.argmax(logp)]       
        lev.sort()
        self.lev_norm=lev[int(self.quantile*lev.size)] 

    def get_weight(self,x):        
        w=np.array([0.3,0.7])
        n=x.shape[0]
        if self.d is None:
            self.d=x.shape[1]                
        if n>self.l:
            # initialize
            if self.S is None:
                self.S=np.cov(x[:-1].T)*self.v           
            # bayesian filter
            # predict
            self.v=self.phi*(self.v-self.d-1)+self.d+1
            self.S=self.S*self.phi
            # update
            self.v+=1
            self.S+=x[-1][:,None]*x[-1]            
            # make allocatiom
            cov=self.S/(self.v-self.d-1)           
            er=np.sqrt(np.diag(cov))                  
            w=np.dot(np.linalg.inv(cov),er)
        w/=self.lev_norm
        if np.sum(np.abs(w))>self.max_lev:
            w/=np.sum(np.abs(w)) 
            w*=self.max_lev            
        return w    
```


```python
model_ct1=CovTrack(l=20,phi=0.95,max_lev=1,quantile=0.9,n_div=10)
s_ct1,lev_ct1=cvbt(model_ct1,x,k_folds=5)
print('Sharpe CovTrack [Cross-Validation]: ',sharpe(s_ct1))
print('Annual return CovTrack Covariance [Cross-Validation]: ',annual_return(s_ct1))
print('Annual vol CovTrack Covariance [Cross-Validation]: ',annual_vol(s_ct1))
plt.title('Equity curve')
plt.plot(np.cumsum(s_ct1))
plt.grid(True)
plt.show()
plt.title('Leverage')
plt.plot(lev_ct1)
plt.show()
```

    Sharpe CovTrack [Cross-Validation]:  1.33
    Annual return CovTrack Covariance [Cross-Validation]:  0.03
    Annual vol CovTrack Covariance [Cross-Validation]:  0.03
    


    
![png](/images/equities_bonds/output_50_1.png)
    



    
![png](/images/equities_bonds/output_50_2.png)
    


We can see that the sharpe improved and so, we can say that the estimator is better. Another interesting point that we can make is that, when we optimize the parameters for the whole dataset, we get that the optimal $\phi$ is:


```python
final_model=CovTrack(l=20,phi=0.95,max_lev=1,quantile=0.9,n_div=10)
final_model.estimate(x)
print('Optimal phi: ', round(final_model.phi,3))
```

    Optimal phi:  0.926
    

Which is different from $\phi=1$; this supports that the parameters change over time and doing this increases the probability of our sequence given the model.

### Notes
We did all this analysis for US market benchmarks which had a great performance in the past years. One could try the same ideas with other markets and check if the results hold.

If you do not consider the changing nature of correlation the result is similar to just holding a constant rebalanced portfolio (sharpe of 1 versus sharpe of 0.9). 


