## Scaling of diversification

Suppose we have $N$ assets $x_i$ to invest on; also, their joint distribution, characterized by the first two moments, is $N(\mu,C)$ (I wrote normal for simplicity). Let us think of these assets as variables with a positive expected value - for example, they can be (_estimated_) strategies returns (and we want to spread our capital across them) and, of course, we design/select them to have positive mean. If we allocate a fraction $f_i$ to strategy/asset $i$ we obtain a ensemble strategy $s$:

$s=\sum_i f_i x_i$

Without loss of intuition, let us say that $f_i=\frac{1}{N}$; the first moments of $s$ become:

$\mu_s=\sum_i \frac{1}{N} \mu_i = \bar{\mu}$

$\sigma_s^2=\text{cov}(\sum_i \frac{1}{N} x_i,\sum_i \frac{1}{N} x_i) = \frac{1}{N^2} \sum_{i,j} C_{ij}$

Now, if the covariance is diagonal (all correlation are zero), we can write:

$\sigma_s^2= \frac{1}{N} \bar{\sigma^2}$

with $\bar{\sigma^2}=\frac{1}{N}\sum_i C_{ii} = \frac{1}{N} \sum_i \sigma_i^2$

And so, without correlations, the sharpe ratio of the ensemble of strategies becomes:

$\text{SR}=\frac{\bar{\mu}}{\bar{\sigma}} \sqrt{N}$

This expression illustrates the power of diversification; when the number of uncorrelated streams of returns increases, the sharpe ratio grows with the square root of it.

Let us make a simulation:


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# use the same mean and scale for all assets
# it is easier and does not change the stated conclusions
# in this case the average mean and scale is just the
# mean and scale..
def simulate_sr(n_obs,n,mu,scale,rho=0):
    R=rho*np.ones((n,n))+(1-rho)*np.diag(np.ones(n))
    cov=scale*scale*R
    x=np.random.multivariate_normal(mu*np.ones(n),cov,size=n_obs)
    s=np.mean(x,axis=1)
    sr=np.mean(s)/np.std(s)
    return sr

# number of observations to be generated
n_obs=5000
# number of assets to be tested
n_values=[1,5,10,20,40,80,150,300]
mu=0.01 # make all means the same
scale=0.02 # make all scales the same

sr_est=[]
sr_theo=[]
for n in n_values:
    sr_est.append(simulate_sr(n_obs,n,mu,scale))
    sr_theo.append(np.sqrt(n)*mu/scale)
plt.plot(n_values,sr_est,'.-',label='Simulation')
plt.plot(n_values,sr_theo,'-',label='Theoretical')
plt.xlabel('Number of assets')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](output_2_0.png)
    


## Effect of correlations

If there are correlations present within strategies/assets then it is intuitive that the diversification does not work that well; if correlations are positive, then when something bad/good happend it tends to happen to all of them and so, it makes sense that the concept of _diversification_ does not help that much. If we look at the formula for the second moment:


$\sigma_s^2=\frac{1}{N^2} \sum_{i,j} C_{ij} = \frac{1}{N^2} \sum_{i} C_{ii} + \frac{1}{N^2} \sum_{i,j!=i} C_{ij} = \frac{1}{N^2} \sum_{i} \sigma_i^2 + \frac{1}{N^2} \sum_{i,j!=i} \sigma_i\sigma_j\rho_{ij}$ 

If all means, scales and correlations are the same then

$\mu_s=\mu$

$\sigma_s^2 = \frac{\sigma^2}{N} \left( 1+ (N-1)\rho \right)$

and so

$\text{SR}=\frac{\mu}{\sigma} \sqrt{\frac{N}{1+(N-1)\rho}}$

Let us make a simulation:


```python
rho=0.1 # correlation
sr_est=[]
sr_theo=[]
for n in n_values:
    sr_est.append(simulate_sr(n_obs,n,mu,scale,rho))
    sr_theo.append(np.sqrt(n/(1+(n-1)*rho))*mu/scale)
plt.plot(n_values,sr_est,'.-',label='Simulation')
plt.plot(n_values,sr_theo,'-',label='Theoretical')
plt.xlabel('Number of assets')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](output_4_0.png)
    


It looks like there is a limit on the increase in sharpe that we get from diversification in the presence of correlations. If we take the limit of large $N$ then:

$\text{SR} \rightarrow \frac{\mu}{\sigma}\frac{1}{\sqrt{\rho}}$



```python
plt.axhline((mu/scale)*(1/np.sqrt(rho)),label='Limit')
plt.plot(n_values,sr_est,'.-',label='Simulation')
plt.plot(n_values,sr_theo,'-',label='Theoretical')
plt.xlabel('Number of assets')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](output_6_0.png)
    


### Effective number

Under zero correlations we have $\text{SR} \propto \sqrt{N}$; with correlations $\propto \sqrt{\frac{N}{1+(N-1)\rho}}$. Making an equivalence between the two, we can say that the presence of correlations _decreases_ the effective number of strategies; to have the same diversification effect as the zero correlation case, we can say we only have $M=\frac{N}{1+(N-1)\rho} < N$ strategies (of course all of this is considering positive correlation which is the case we should be worried); the presence of a small correlation can destroy all the diversification.
