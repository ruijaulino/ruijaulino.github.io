# Allocation with costs

Let us consider the optimal allocation with costs. At every bet we are penalized with a cost $c$; with that, capital after $n$ periods is:

$S_n = S_0 \cdot \left(1-c(z_1) + w(z_1)^Tx_1 \right) \cdots \left( 1-c(z_n) + w(z_n)^Tx_n\right)$

$c(z_i)$ is the percentage cost we pay when presented with information $z_i$. Working the expression:

$S_n = S_0 \exp \left(nG\right)$

with 

$G = \frac{1}{n}\sum_i \log \left(1-c(z_i) + w(z_i)^Tx_i\right) \rightarrow \mathbb{E}\left[ \log \left(1-c(z) + w(z)^Tx\right) \right]$

A logical objective is to maximize the growth rate.

A common model for costs is to assume proportionality to the transacted value $w$: $c(z_i) = c^T \|w(z_i)\|$ (with $c$ being a percentage value). Here we can make two distinct cases: the simplest one where we make bets that are isolated in time, let's say, everyday we make a transaction during a few minutes - in this case, when we bet tomorrow we start from a initial position of zero. The other case is when we just _adjust_ the allocation given new information (every day we check what should be the optimal weight and adjust our exposure; in practice we tend to pay less). This is more complicated as the cost must be written as $c^T \|w(z_i) - w(z_{i-1})\|$. If we look at $\mathbb{E}\left[ \log \left(1-c(z) + w(z)^Tx\right) \right]$ there is not clear way to incorporate this cost function - what makes sense is to assume that, over time, we will see many times information $z_{i-1}, z_i$ and the growth rate to be maximized should be

$G = \mathbb{E}\left[ \log \left(1-c^T\|w(z) - q\| + w(z)^Tx\right) \right]$

where $q$ is the allocation we already have (what we are saying is that we can interpret the problem as optimizing the allocation given that we start always from $q$; since we have no way to know what we have in any period of time it makes sense to consider the generic problem and make sequential decisions based on that). We can expand the growth rate:

$G \approx \mathbb{E}\left[ \log \left(1-c^T\|w(z) - q\|\right) + \frac{w(z)^Tx}{1-c(z)} - \frac{1}{2}\frac{(w(z)^Tx)^2}{(1-c(z))^2} \right]$

which for small cost vector $c$:

$G \approx \mathbb{E}\left[ c^T\|w(z) - q\| + w(z)^Tx - \frac{1}{2}(w(z)^Tx)^2 \right]$

Now

$G = \int \left[ \int \left( c^T\|w(z) - q\| + w(z)^Tx - \frac{1}{2}(w(z)^Tx)^2 \right) p(x\|z) \text{d}x \right] p(z) \text{d}z$

The optimization condition:

$\frac{\partial G}{\partial w} = 0 \rightarrow \frac{\partial }{\partial w}  \int \left( c^T\|w(z) - q\| + w(z)^Tx - \frac{1}{2}(w(z)^Tx)^2 \right) p(x\|z) \text{d}x = 0$

which translates into

$\mu_{x\|z} - C_{x\|z}w - \sum_j c_j \text{sign}(w_j - q_j) = 0$
 
is equivalent to find

$\text{argmin}_w \frac{1}{2} w^T C w - \mu^T w + \sum_j c_j \|w-q\| = L$

Inspired by lasso, we can devise a coordinate descent algorithm for this (minimize with respect to $w_i$ while keeping $w_{j \neq i}$ constant and iterate).

First notice that $\frac{1}{2} w^T C w = \frac{1}{2} C_{ii} w_i^2 + w_i \sum_{k \neq i} C_{ik}w_k + \text{const}$ and $\mu^T w = \mu_i w_i + \text{const}$. Then, regarding $w_i$, the objective function can be written as:

$L = L_i +\text{const} = \frac{1}{2} C_{ii} w_i^2 + w_i \sum_{k \neq i} C_{ik}w_k - \mu_i w_i + c_i\|w_i-q_i\| + \text{const} = \frac{1}{2} C_{ii} w_i^2 -r_i w_i + c_i\|w_i-q_i\| + \text{const}$

with $r_i = \mu_i - \sum_{k \neq i} C_{ik}w_k$. Now we can maximize $G$ with respect to $w_i$.


### Solution to the subproblem

Consider the minimization of $f = \frac{1}{2} \sigma^2 w^2 - w\mu + c\|w-b\|$. We can see that, if $\mu > b\sigma^2 + c$ then $w = \frac{\mu-c}{\sigma^2}$ and if $\mu < b\sigma^2 - c$ then $w = \frac{\mu+c}{\sigma^2}$. Otherwise $w = b$. This roughly means that, if the expected value is larger than the costs to change position we allocate otherwise we don't do nothing.

With this we just need to iterate until convergence by cycling through variables.

### Degenerate cases

It can happen that the optimal solution have a negative expected growth rate and we still need to take that bet. Consider that we have a large negative allocation and now we have a small positive expectation but not enough to cover the costs of changing the position; we should reduce the short exposure at a cost but now we expect to lose a bit money just in the right proportion to minimize the effect of the transaction. 

### Numerical implementation



```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def soft(m, v, c, b):
    '''
    Solution to the subproblem
    '''
    if m > b*v + c:
        return (m-c) / v
    elif m < b*v - c:
        return (m+c) / v
    else:
        return b

def alloc(mu, C, c, q = None, max_iter=100, tol=1e-6, convergence_periods = 5):
    p = mu.size
    if q is None: q = np.zeros(p)
    if not isinstance(c, np.ndarray): c = c*np.ones(p)
    max_iter = max(max_iter, 2*convergence_periods)
    # initialize    
    w = mu / np.diag(C)
    w_hist = np.zeros((max_iter, w.size))
    w_hist[0] = w
    for n in range(1, max_iter):
        for j in range(p):
            # Compute residual for j-th coordinate
            r_j = mu[j] - np.dot(C[j,:], w) + C[j,j] * w[j]
            w[j] = soft(r_j, C[j,j], c[j], q[j])
        converged = False
        # it may happen that the solution is jumping
        # from one state into another. detect this
        # behaviour and quit
        if n > convergence_periods:
            for l in range(convergence_periods+1):
                if np.linalg.norm(w - w_hist[n-l]) < tol:
                    converged = True
        w_hist[n] = w
        if converged:
            break
    if n == max_iter - 1:
        print('Coordinate descent did not converge...')
    return w    
```


```python
# example
mu = np.array([0.1, 0.05])
C = np.array([[0.2**2,0.2*0.05*0.1],[0.2*0.05*0.1,0.05**2]])
c = 0.05
q = np.array([1,-1])
w = alloc(mu, C, c, q = q, max_iter=100, tol=1e-6, convergence_periods = 5)
w
```




    array([ 1.26262626, -0.50505051])



In this example, for variable $x_2$, clearly it does not pay costs but since we start with a $-1$ position, the optimal decision is to reduce the allocation and not flip sides.




## With a term structure

The previous problem formulation has the characteristic that we optimize the decision as new information comes. There is another class of problems that we can think of which can be stated as: in the next $p$ periods we will make a sequence of decisions but we already know them - for example, imagine that an asset tends to go up overnight and is flat during trading hours (well this one is known) or there is a pattern that an asset is flat in the first days of the month then down and up on the end of the month; in the end _seasonal_ models fit on this framework: on period 1 we take an action, then period 2 comes and we take another one and so on; when we are at period one we already know what we want to do in period 2 up to $p$; after period $p$ the sequence repeats back in to period 1. For this models we estimate the distribution for these periods and make a bet. The point is, we still have a sequence of decisions but we already know the changes in allocation that will happen in the future; some changes may be small and not worth to be done bue to costs but in the end, the optimal decision depends on the whole sequence of actions. 

Let us focus on the case of a single asset and a term structure model (i.e, a model for returns on some periods in the future).

After $n$ sequences (each sequence has $p$ sequential returns), the capital is

$S_n = S_0 \Pi_i \left( (1-c_1+w_1 x_1)\cdots(1-c_p+w_p x_p) \right)_i = S_0 \exp(nG)$

where $c_m$ is the cost we pay on period $m$ on the term structure. The growth rate is

$G = \frac{1}{n}\sum_i \sum_j \log \left( 1-c_j+w_jx_{ij}\right) \approx \sum_j \mathbb{E}\left[\log \left( 1-c_j+w_j x_j\right) \right]$

For small costs

$G = \sum_j \left( \mu_j w_j - \frac{1}{2} \sigma_j^2 w_j^2 - c_j \right)$ 

Now, following a cost model proportional to the transaction weight we can write $c_j = c \|w_j-w_{j-1}\|$ (when $j = 1$ we use $w_p$ to make it cyclical); this represents that we pay a cost on the weight change. Writting as a minimization problem ($L = -G$):

$L = \sum_j \left( -\mu_j w_j + \frac{1}{2} \sigma_j^2 w_j^2 + c \|w_j-w_{j-1}\| \right)$ 

How to minimize $L$? Following the same idea of coordinate descent, we can write how $L$ depends on a single $w_i$:

$L = L_i + \text{const} = \frac{1}{2}\sigma_i^2 w_i^2 - w_i \mu_i + c \cdot \left( \|w_i - w_{i-1}\| + \|w_{i+1}-w_{i}\|\right) + \text{const}$

We can see that there is a dependency on the next and previous weight, reflecting the fact that the optimization takes into account the whole sequence of decisions to be taken.

### Solution to the subproblem

Consider the minimization of $f = \frac{1}{2}\sigma^2 w^2 - w \mu + c \left( \|w - a\| + \|b -w\|\right)$. This is a bit more complicated than previously but we can still solve. For $b>a$ (or just flip variables):

- for $\mu<a\sigma^2-2c$, $w = \frac{\mu+2c}{\sigma^2}$
- for $a\sigma^2<\mu<b\sigma^2$, $w = \frac{\mu}{\sigma^2}$
- for $\mu>b\sigma^2+2c$, $w = \frac{\mu-2c}{\sigma^2}$
- otherwise $w=a$ if $f(w=a)<f(w=b)$ else $w=b$

As before, we just need to iterate over variables until convergence.

### Numerical implementation

The code below implements the idea. It is assumed that when the bet sequence ends we go to the start of another one; it is trivial to change the algorithm to consider that the sequence is isolated.


```python
def tsoft(v, m, c, a, b):
    w = 0.
    if a<=b:
        if m < a*v-2*c:
            w = (m+2*c) / v
        elif a*v < m < b*v:
            w = m / v
        elif m > v*b + 2*c:
            w = (m-2*c) / v       
        else:
            w = b
            if 0.5*v-m*a+c*(b-a) <= 0.5*v-m*b+c*(b-a):
                w = a
    else:
        if m < b*v-2*c:
            w = (m+2*c) / v
        elif b*v < m < a*v:
            w = m / v
        elif m > v*a + 2*c:
            w = (m-2*c) / v
        else:
            w = b
            if 0.5*v-m*a+c*(a-b) < 0.5*v-m*b+c*(a-b):
                w = a
    return w

def term_alloc(m, v, c, max_iter=100, tol=1e-6, convergence_periods = 5):
    '''
    m: vector of expected values
    v: vector variances
    c: float with pct cost
    '''
    # initialize with c=0 solution
    assert m.size > 1, "we must have more than one sequential estimate"
    max_iter = max(max_iter, 2*convergence_periods)
    w = m / v
    p = m.size
    w_hist = np.zeros((max_iter, w.size))
    w_hist[0] = w
    for n in range(1, max_iter):
        for j in range(p):
            w[j] = tsoft(v[j], m[j], c, np.roll(w, 1)[j], np.roll(w, -1)[j])
        converged = False
        if n > convergence_periods:
            for l in range(convergence_periods+1):
                if np.linalg.norm(w - w_hist[n-l]) < tol:
                    converged = True
        w_hist[n] = w
        if converged:
            break
    if n == max_iter - 1:
        print('Coordinate descent did not converge...')
    return w
```


```python
# example
m = np.array([0.01,-0.003,0.04])
sigma = np.array([0.02,0.02,0.09])
v = sigma**2
c = 0.002
w = term_alloc(m, v, c, max_iter=100, tol=1e-6)
plt.plot(m / v, label="Raw weights (mu / sigma^2)", alpha=0.5)
plt.plot(w, label="Term weights")
plt.legend()
plt.grid(True)
plt.title("Circular Trading Weights with Directional Cost")
plt.show()
```


    
![png](/images/costs_alloc/output_11_0.png)
    


In this example we see that, although we expect a negative return on period 2 it is not worth to short it.

As a final example, let us consider the Nasdaq ETF returns overnigh and during trading hours with 1 basis point of costs.


```python
# Expected value overnight and during trading hours
m = np.array([ 5.42184628e-04, -7.08220270e-06])
# Scale overnight and during trading hours
sigma = np.array([0.00904901, 0.01470038])
# Compute variance
v = sigma**2
c = 0.0001
w = term_alloc(m, v, c, max_iter=100, tol=1e-6)
plt.plot(m / v, label="Raw weights (mu / sigma^2)", alpha=0.5)
plt.plot(w, label="Term weights")
plt.legend()
plt.grid(True)
plt.title("Circular Trading Weights with Directional Cost")
plt.show()
```


    
![png](/images/costs_alloc/output_13_0.png)
    


We see that the optimal decision is to keep some allocation during trading hours. If we increase the costs to 2 basis points we have that the optimal solution is not change the position.


```python
m = np.array([ 5.42184628e-04, -7.08220270e-06])
sigma = np.array([0.00904901, 0.01470038])
v = sigma**2
c = 0.0002
w = term_alloc(m, v, c, max_iter=100, tol=1e-6)
plt.plot(m / v, label="Raw weights (mu / sigma^2)", alpha=0.5)
plt.plot(w, label="Term weights")
plt.legend()
plt.grid(True)
plt.title("Circular Trading Weights with Directional Cost")
plt.show()
```


    
![png](/images/costs_alloc/output_15_0.png)
    


