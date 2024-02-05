
# An approximation to model assesment

When testing multiple models on the same dataset we are introducing a bias on out-of-sample expected performance. In the end, to eliminate this bias, we need to cross validate the choice of model but this can be quite computationally expensive. 

The objective here is to try to approximate this quantity.


Consider a data set $D$ where 
$D_i=\{x_i,y_i\}$ (
$x_i \in \mathbb{R}^n$, 
$y_i \in \mathbb{R}$, 
$D_i$ 
are independent) and set $M$ of models; a model $M_i$ is a mapping 
$\mathbb{R}^n \rightarrow \mathbb{R}$ 
that depend on some parameters 
$\theta_j$ 
that are fit with the data. The models can be different learning algorithms and/or the same learning algorithm with different sets of hyperparameters.

To select the best model we can consider the quantity:

$p(M_i\|D) \propto p(M_i)p(D\|M_i)$

If all models have the same prior 
$p(M_i)$ 
selecting the best model can be done by evaluating 
$p(D\|M_i)$. 
Focusing on this quantity and noting that there are parameters 
$\theta_j$ to be fit we can write:

$p(D\|M_i)=\int p(D\|\theta_i,M_i)p(\theta_i\|M_i) \text{d} \theta_i$

In general terms, this integral is dominated by the parameters that maximize 
$p(D\|\theta)$ 
and if this distribution is quite narrow compared to the prior, then there is a penalization. This way, when comparing 
$p(D\|M_i)$ 
for several models, there is a penalty on complexity (means the model is overfit) and the most parsimonious one is considered the best.

Another way to look at the problem this is to consider 
$D=\{D_1,D_2\}$ 
(i.e, $D$ as a union of two subsets that exhaust $D$) and note that 
$p(D\|M_i)=p(D_2\|D_1,M_i)p(D_1\|M_i)$ 
(data points in $D$ are independent). Integrating the parameters we get:

$p(D\|M_i)=\int p(D_2\|D_1,\theta_i,M_i)p(D_1\|\theta_i,M_i)p(\theta_i\|M_i) \text{d} \theta_i$

With the same intuition, this integral is dominated by 
$p(D_2\|\hat{\theta_i},M_i)$ 
with 
$\hat{\theta_i}$ 
the parameters that maximize 
$p(D_1\|\theta_i,M_i)$. 
In rough term this approximates to (imagine a single parameter 
$\theta$ 
and univariate functions - narrow near their maximum values - 
$p(D_1\|\theta)$
) and 
$p(D_2\|\theta)$ 
with the two function having maximums in different places; we are selecting 
$\theta$ 
based on one of the maximums; the integral is dominated by their product near this 
$\theta$):

$p(D\|M_i) \approx p(D_2\|\hat{\theta_{D_1}}) p(D_1\|\hat{\theta_{D_1}}) \frac{\delta_{\text{post}\_{D_1}} \delta_{\text{post}\_{D_2}}}{\delta_{\text{prior}}}$

This way lead us to measure 
$p(D\|M_i)$ 
as _predictive_ capacity given that we fit on data from the same distribution. 

In theory we can generate many divisions 
$\{D_1^k,D_2^k\}$ 
of $D$ (again, observations are independent); since any division should evaluate to 
$p(D\|M_i)$ 
it is possible to write:

$p(D\|M_i)=\frac{1}{k} \sum_k p(D_2^k\|D_1^k,M_i)p(D_1^k\|M_i) = \frac{1}{k} \sum_k \int p(D_2^k\|D_1^k,\theta_i,M_i)p(D_1^k\|\theta_i,M_i)p(\theta_i\|M_i) \text{d} \theta_i$

##### All disjoint sets

The previous formulation becomes more intuitive and easy to evaluate if we consider all possible disjoint sets. To give an example, consider 
$D=\{d_1,d_2,d_3,d_4\}$, 
i.e, $D$ is a set of four observations. We can write 

$P(D\|M_i)^4=p(d_1\|d_{2,3,4},M_i)p(d_{2,3,4}\|M_i)p(d_2\|d_{1,3,4},M_i)p(d_{1,3,4}\|M_i)p(d_3\|d_{2,1,4},M_i)p(d_{2,1,4}\|M_i)p(d_4\|d_{2,3,1},M_i)p(d_{2,3,1}\|M_i)$ 

given that all observations are independent conditonal on model (but the conditionals remain because the model will depend on the data where it is fit) the non conditional terms are a product of probabilities and:

$P(D\|M_i)^4=p(d_1\|d_{2,3,4},M_i)p(d_2\|d_{1,3,4},M_i)p(d_3\|d_{2,1,4},M_i)p(d_4\|d_{2,3,1},M_i)p(D\|M_i)^3$ 

and so

$P(D\|M_i)=p(d_1\|d_{2,3,4},M_i)p(d_2\|d_{1,3,4},M_i)p(d_3\|d_{2,1,4},M_i)p(d_4\|d_{2,3,1},M_i)$


This is true for any divison of observations and considering that we make all combinations possible (for example, dividing the data like in k fold cross validation also works).


Now, let 
$D^k=\{D_1^k,D_2^k\}$ 
be disjoint sets from a division like explained above (for example k fold):

$p(D\|M_i)=\Pi_k p(D_2^k\|D_1^k,M_i) = \Pi_k \int p(D_2^k\|D_1^k,\theta_i,M_i) \frac{p(D_1^k\|\theta_i,M_i)p(\theta_i\|M_i)}{\int p(D_1^k\|\theta_i',M_i)p(\theta_i'\|M_i) \text{d} \theta_i'} \text{d} \theta_i$

We can observe that now, the term with 
$p(D_1^k\|\theta)$ 
is normalized (there is a multiplication by a term near 1 from 
$\hat{\theta}$ 
with 
$p(D_2)$; 
using the same intuition this approximates to:


$p(D\|M_i)\approx \Pi_k p(D_2^k\|\hat{\theta_{D_1^k}})$


This is the idea of cross-validation. Usually one is interested in cross-validation error but this is a proxy to the (log)probability of the data (this is more or less clear for a gaussian distribution).

##### Bias in assesing performance

If we try many models and the objective is to select the best one, this procedure can do it but there will be a bias to make inference on how the model will perform on future data. There is a need to penalise that we are choosing a model (the same this as with the parameters).

A typical procedure is to do nested cross validation to make inference of out of sample performance (i.e, test the performance of the model selection procedure), although if one is only interested on model choice this is not necessary as the bias (in principle) is the same across all models.

A way to look at nested cross validation is to consider the probability of $D$ under the set of models $M$:

$p(D)=\int p(D,M) \text{d}M = \int p(D\|M)p(M) \text{d}M$

separating again $D$ into subsets:

$p(D)=\int p(D,M) \text{d}M = \int p(D\|M)p(M) \text{d}M = \int p(D_2\|D_1,M)p(D_1\|M)p(M) \text{d}M$

Finally taking into account that models have parameters:

$p(D)=\int \int p(D_2\|D_1,\theta,M)p(D_1\|\theta,M)p(\theta\|M)p(M) \text{d}M \text{d}\theta$

Again, we can interpreted this integral as being 
$p(D_2\|\hat{\theta},\hat{M})$, 
with 
$\hat{\theta},\hat{M}$ 
the best parameters and the best model in $D_1$. Since the integral is difficult we can resort to the same logic and intepret this as predictive capacity: dividing $D_1$ into 
$\{D_{11},D_{12}\}$:

$p(D)=\int \int p(D_2\|D_1,\theta,M)p(D_{12}\|D_{11}\theta,M)p(D_{11}\|\theta,M)p(\theta\|M)p(M) \text{d}M \text{d}\theta$

and so this can read as 
$p(D_2\|\hat{\theta},\hat{M})$ 
with 
$\hat{\theta},\hat{M}$ 
being the parameters of the best model trained on $D_1$ which is the best model that performed on $D_{12}$ with parameters trained on $D_{11}$. (note that the previous ideas of using disjoint sets can be translated here)

This quantity should give a non biased $p(D)$ under all models considered.

There are two problems:
- model selection, $p(M\|D)$ (which as as proxy $p(D\|M)$)
- model assesment, $p(D)$


##### Approximatation of p(D)


Since
$p(D)=\int p(D,M) \text{d}M = \int p(D\|M)p(M) \text{d}M$

and giving equal prior to all models we can write:

$p(D)=\frac{1}{m} \sum_m p(D\|m)$


if we choose as model the one that has the highest 
$p(D\|M)$ 
then we can say that the best model performance is lower than 
$p(D\|M)$ 
given that the average is lower than the maximum value.


In practice we deal with 
$\log p(D\|M)$ 
and 
$\log p(D)$. 
Let 
$M^\*$ 
be the model with highest $p(D\|M)$. Then:

$\log p(D) = \log \left( \frac{1}{m} \sum_m p(D\|m) \right) = \log \left( \frac{1}{m} \sum_m \exp(\log(p(D\|m))) \right) = \log \left( \frac{1}{m} \sum_m \exp(\log(p(D\|m))-\log(p(D\|M^\*)))\exp(\log(p(D\|M^\*))) \right)$

in the end

$\log p(D) = \log p(D\|M^\*) + \log \left( \frac{1}{m} \sum_m \exp[ \log p(D\|m) - \log p(D\|M^\*)] \right)$

This is important due to under and overflow of calculations.


To get an approximation for average error (for example, the average error in cross validation) it is useful to gain some intuition from a normal distribution. Given some independent data $X$, 
$\log p(X)=\frac{1}{2} \left[ N\log(2\pi) + N\log(\sigma) + \frac{N}{\sigma^2} \frac{1}{N} \sum_i (x_i - \mu)^2 \right]$ 
and:

$\epsilon = - \sigma^2 \left[ \log \sigma^2 +\log(2\pi) + 2 \log p(X)/N \right]$

and so a proxy for 
$\log p(X)$ 
can be 
$N \epsilon$. 
So for error, the average out of sample error will be corrected as 
$\log p(D)$ 
but with 
$N \epsilon (D|M)$ 
with 
$\epsilon (D\|M)$ 
the average cross validation error for a model.


This makes sense as, since we want to penalize error, if we have a model that has a average error over many samples small this model should be much better than the others and there should be less diference in out of sample performance.


## Tests
   
To test the idea some artificial data will be created and the formula tested. 

The data has $n$ observations of $p$ variables(possible features) $x$. $y$ will be a linear function of one of the variables.

- feature i (columns), $x_i \propto N(0,\sigma)$
- target, $y \propto N(a \cdot x_0,\sigma)$
  
The models are a simple regressions (no intercept) and each model is based on a single feature. The idea here is to many features that are uncorrelated to the target. 

We know that if we perform cross validation then we can select the best model (in this case, the model that has the true feature) but its out of sample performance is biased. To evaluate the true value we can perform nested cross validation where the inner loop selects the feature and the performance is evaluated on the outter loop. Also, we can compute the out of sample performance from the formula/correction above and check if they match. In order to have another basis of comparisson, after a single run we can generate more data and check how the model selected by cross validation performed - this value can be compared with the nested, cross-validation and cross-validation corrected values.


   


```python
# some imports
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import time
np.random.seed(0)
```


```python
def gauss_prob(y,mu,scale):
    return np.exp(-0.5*np.power((y-mu)/scale,2))/(scale*np.sqrt(2*np.pi))
```


```python
# simple cross validation class
# receives data and the model
# evaluates CV log prob and CV error
class CV(object):
    def __init__(self,k=3):
        self.k=k
    
    def run(self,model,x,y):
        x=np.copy(x)
        y=np.copy(y)
        idx=np.arange(x.shape[0])
        np.random.shuffle(idx)
        folds_idx=np.array_split(idx,self.k)
        test_probs=[]
        test_errs=[]

        for k in range(len(folds_idx)):
            test_idx=folds_idx[k]
            train_idx=[]
            for j in range(len(folds_idx)):
                if j!=k:
                    train_idx.append(folds_idx[j])
            train_idx=np.hstack(train_idx)
            m=deepcopy(model)
            m.train(x[train_idx],y[train_idx])
            
            test_prob=m.eval_prob(x[test_idx],y[test_idx])
            test_err=m.eval_err(x[test_idx],y[test_idx])
  
            test_probs.append(test_prob)
            test_errs.append(test_err)

        test_probs=np.hstack(test_probs)
        test_errs=np.hstack(test_errs)

        return test_probs,test_errs

# simple linear model
# the index of x is set on instantiation
class Model(object):
    def __init__(self,idx):
        self.idx=idx
        self.a=None
        self.scale=None
    
    def view(self):
        print('** Model IDX %s **'%self.idx)
        print('->     a=%s'%self.a)
        print('-> scale=%s'%self.scale)
        
    def train(self,x,y):
        '''
        MLE estimation of model
        '''
        x=np.copy(x)
        y=np.copy(y)
        x_=x[:,self.idx]
        self.a=np.dot(y,x_)/np.dot(x_,x_)
        self.scale=np.sqrt(np.mean(np.power(y-x_*self.a,2)))
    
    def eval_prob(self,x,y):
        mu=self.a*x[:,self.idx]
        prob=gauss_prob(y,mu,self.scale)
        return prob 
    
    def eval_err(self,x,y):
        return np.power((y-self.a*x[:,self.idx]),2)

# simple nested model
# during training it selected the index of x to be used
class NestedModel(object):
    def __init__(self,k=5):
        self.k=k
        self.idx=None
        self.a=None
        self.scale=None
    
    def view(self):
        print('** Model IDX %s **'%self.idx)
        print('->     a=%s'%self.a)
        print('-> scale=%s'%self.scale)
        
    def train(self,x,y):
        '''
        MLE estimation of model
        '''
        x=np.copy(x)
        y=np.copy(y)
        logps=[]
        for idx in range(x.shape[1]):
            m=Model(idx)
            # make choice with probability - should not make diffference as what matter is having a choice
            p,_=CV(self.k).run(m,x,y)
            logps.append(np.sum(np.log(p)))
        self.idx=np.argmax(logps)
        x_=x[:,self.idx]
        self.a=np.dot(y,x_)/np.dot(x_,x_)
        self.scale=np.sqrt(np.mean(np.power(y-x_*self.a,2)))
    
    def eval_prob(self,x,y):
        mu=self.a*x[:,self.idx]
        prob=gauss_prob(y,mu,self.scale)
        return prob # np.sum(np.log(prob))
    
    def eval_err(self,x,y):
        # return np.power((y-self.a*x[:,self.idx])/self.scale,2)
        return np.power((y-self.a*x[:,self.idx]),2)
```


```python
# this function runs a single simulation
# 1 - generate data
# 2 - create a model for each feature
# 3 - evaluate CV error/log prob
# 4 - select model with highest CV prob / minimum CV error 
# 5 - correct previous quantities to estimate OOS performance
# 6 - run CV on the nested model
# 7 - generate more data and evaluate the model selected by CV previously
# return metrics
def run_simulation(n,n_f,a,scale,b=0):
    x=np.random.normal(0,scale,(n,n_f))
    y=a*x[:,0]+b*x[:,1]+np.random.normal(0,scale,n)

    cv_logp=[]
    cv_err=[]

    for i in range(0,n_f):
        model=Model(i)
        logp,err=CV(5).run(model,x,y)
        cv_logp.append(np.sum(np.log(logp)))
        cv_err.append(np.mean(err))


    cv_logp=np.array(cv_logp)
    cv_err=np.array(cv_err)

    cv_err_min=np.min(cv_err)

    # make prediciton for OOS logP
    cv_logp_max=np.max(cv_logp)
    cv_logp_norm=cv_logp-cv_logp_max
    theo_nested_logp=np.log(np.mean(np.exp(cv_logp_norm)))+cv_logp_max 
    
    # make prediction for OOS err
    logp_proxy=-1*cv_err*y.size
    logp_proxy_max=np.max(logp_proxy)
    logp_proxy_norm=logp_proxy-logp_proxy_max
    theo_nested_err=(np.log(np.mean(np.exp(logp_proxy_norm)))+logp_proxy_max)
    theo_nested_err=-1*theo_nested_err/y.size

    nested_model=NestedModel(5)
    prob,err=CV(5).run(nested_model,x,y)
    nested_logp=np.sum(np.log(prob))
    nested_err=np.mean(err)

    # simulate on an OOS sample with the same size with the model selected from CV and trained on all data
    cv_model=Model(np.argmax(cv_logp))
    cv_model.train(x,y)    
    
    x_oos=np.random.normal(0,scale,(n,n_f))
    y_oos=a*x_oos[:,0]+b*x_oos[:,1]+np.random.normal(0,scale,n)          
    
    prob=cv_model.eval_prob(x_oos,y_oos)
    err=cv_model.eval_err(x_oos,y_oos)

    oos_logp=np.sum(np.log(prob))
    oos_err=np.mean(err)

    return nested_logp,nested_err,cv_logp_max,cv_err_min,oos_logp,oos_err,theo_nested_logp,theo_nested_err


```


```python
# this functions runs a full sudy where the previous function is called multiple times
# after create histograms of the performance metrics
def run_study(n_simulations,n,n_f,a,scale,b):

    print('** RUN STUDY **')
    print('N SIMULATIONS: ', n_simulations)
    print('N POINTS: ', n)
    print('COEFF: ', a)
    print('SCALE: ', scale)
    print('EXTRA COEFF: ', b)
    time.sleep(0.5)
    nested_logp=[]
    nested_err=[]
    nested_sr=[]
    
    cv_logp_max=[]
    cv_err_min=[]
    cv_sr_max=[]
    
    oos_logp=[]
    oos_err=[]
    oos_sr=[]
    
    theo_nested_logp=[]
    theo_nested_err=[]    
    theo_nested_sr=[]
    
    for n_sim in tqdm(range(n_simulations)):

        nested_logp_,nested_err_,cv_logp_max_,cv_err_min_,oos_logp_,oos_err_,theo_nested_logp_,theo_nested_err_=run_simulation(n,n_f,a,scale,b)

        nested_logp.append(nested_logp_)
        nested_err.append(nested_err_)

        cv_logp_max.append(cv_logp_max_)
        cv_err_min.append(cv_err_min_)

        oos_logp.append(oos_logp_)
        oos_err.append(oos_err_)

        theo_nested_logp.append(theo_nested_logp_)
        theo_nested_err.append(theo_nested_err_)

        
    nested_logp=np.array(nested_logp)
    nested_err=np.array(nested_err)
    nested_sr=np.array(nested_sr)

    cv_logp_max=np.array(cv_logp_max)
    cv_err_min=np.array(cv_err_min)
    cv_sr_max=np.array(cv_sr_max)
    
    oos_logp=np.array(oos_logp)
    oos_err=np.array(oos_err)
    oos_sr=np.array(oos_sr)
    
    theo_nested_logp=np.array(theo_nested_logp)
    theo_nested_err=np.array(theo_nested_err)
    theo_nested_sr=np.array(theo_nested_sr)
    
    
    plt.hist(nested_logp,label='LogP Nested',alpha=0.5,color='k')
    plt.axvline(np.median(nested_logp),color='k')

    plt.hist(theo_nested_logp,label='LogP Theo',alpha=0.5,color='r')
    plt.axvline(np.median(theo_nested_logp),color='r')

    plt.hist(cv_logp_max,label='LogP CV MAX',alpha=0.5,color='b')
    plt.axvline(np.median(cv_logp_max),color='b')

    plt.hist(oos_logp,label='LogP OOS',alpha=0.5,color='g')
    plt.axvline(np.median(oos_logp),color='g')

    plt.legend()
    plt.show()

    plt.hist(nested_err,label='ERR Nested',alpha=0.5,color='k')
    plt.axvline(np.median(nested_err),color='k')

    plt.hist(theo_nested_err,label='ERR Theo',alpha=0.5,color='r')
    plt.axvline(np.median(theo_nested_err),color='r')

    plt.hist(cv_err_min,label='ERR CV MIN',alpha=0.5,color='b')
    plt.axvline(np.median(cv_err_min),color='b')

    plt.hist(oos_err,label='ERR OOS',alpha=0.5,color='g')
    plt.axvline(np.median(oos_err),color='g')

    plt.legend()
    plt.show()

```

## TEST 1
True model is among the set of models. The number of irrelevant features is increased from simulation to simulation.


```python
# few features, true model is among the tested models
n_simulations=1000
n=50
n_f=3
a=0.5
b=0.
scale=1
run_study(n_simulations,n,n_f,a,scale,b)

```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.0


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:14<00:00, 70.13it/s]



![png](/images/apr_ma/output_12_2.png)



![png](/images/apr_ma/output_12_3.png)



```python
n_simulations=1000
n=50
n_f=10
a=0.5
b=0.
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.0


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:46<00:00, 21.53it/s]



![png](/images/apr_ma/output_13_2.png)



![png](/images/apr_ma/output_13_3.png)



```python
n_simulations=1000
n=50
n_f=30
a=0.5
b=0.
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.0


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:47<00:00,  4.41it/s]



![png](/images/apr_ma/output_14_2.png)



![png](/images/apr_ma/output_14_3.png)



```python
n_simulations=1000
n=50
n_f=100
a=0.5
b=0.
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.0


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [12:42<00:00,  1.31it/s]



![png](/images/apr_ma/output_15_2.png)



![png](/images/apr_ma/output_15_3.png)



```python
n_simulations=1000
n=50
n_f=200
a=0.5
b=0.
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.0


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [18:10<00:00,  1.09s/it]



![png](/images/apr_ma/output_16_2.png)



![png](/images/apr_ma/output_16_3.png)


### Comments

The correction seem to approximate well the out of sample performance; in particular, the nested CV tends to overestimate the error and the correction tends to underestimate. As the number of irrelevant features grow we can observe that the correction approximates better the performance with out of sample data. When there are few irrelevant features in proportion to the data there is no significative difference.

This can be an artifact due to the correct model being among the set of models. Let us check what happens when this is not the case.

### TEST 2
True model is not among the set of models. An aditional dependency is introduced (parameter $b$).


```python
n_simulations=1000
n=50
n_f=3
a=0.5
b=0.1
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.1


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:14<00:00, 70.87it/s]



![png](/images/apr_ma/output_19_2.png)



![png](/images/apr_ma/output_19_3.png)



```python
n_simulations=1000
n=50
n_f=10
a=0.5
b=0.1
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.1


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:44<00:00, 22.39it/s]



![png](/images/apr_ma/output_20_2.png)



![png](/images/apr_ma/output_20_3.png)



```python
n_simulations=1000
n=50
n_f=30
a=0.5
b=0.1
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.1


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:16<00:00,  7.34it/s]



![png](/images/apr_ma/output_21_2.png)



![png](/images/apr_ma/output_21_3.png)



```python
n_simulations=1000
n=50
n_f=100
a=0.5
b=0.1
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.1


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [07:29<00:00,  2.22it/s]



![png](/images/apr_ma/output_22_2.png)



![png](/images/apr_ma/output_22_3.png)



```python
n_simulations=1000
n=50
n_f=200
a=0.5
b=0.1
scale=1
run_study(n_simulations,n,n_f,a,scale,b)
```

    ** RUN STUDY **
    N SIMULATIONS:  1000
    N POINTS:  50
    COEFF:  0.5
    SCALE:  1
    EXTRA COEFF:  0.1


    100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [15:40<00:00,  1.06it/s]



![png](/images/apr_ma/output_23_2.png)



![png](/images/apr_ma/output_23_3.png)


### Comments

Although continues to underestimate out of sample performance, it seems to be a good approximation.


## Final remarks

The intuition of the correction is that, if we have a model that dominates the others in terms of cross validation performance then the out of sample performance of that model will be very similar to the one that we would determine if we cross validate the model selection procedure: this makes sense because, if a model is considerably better than the others it will always be chosen and so, trivially, the CV error is a good measure of OOS performance. This seems a useful result.

