# Bayesian filtering for returns sequences

Consider observations $y_t\in\mathbb{R}^k$. We assume that we care mainly about first and second moments and we use a normal

$$
y_t \mid \mu_t,\Sigma_t \sim \text{N}(\mu_t,\Sigma_t)
$$

where

* $\mu_t$ is the conditional mean
* $\Sigma_t$ is the conditional covariance matrix.

In general, both quantities depend on observable variables,

$$
\mu_t=f(x_t),\qquad
\Sigma_t=g(z_t)
$$

where $f$ and $g$ are generic models. Let

$$
\mathcal{F}_t=\{y_{1:t},x_{1:t},z_{1:t}\}
$$

denote the information available up to time $t$.

Our objective is to estimate the conditional distribution of $(\mu_t,\Sigma_t)$ sequentially as new observations become available.

The Bayesian filtering recursion is

$$
p(\mu_t,\Sigma_t\mid\mathcal{F}_t) \propto p(y_t\mid\mu_t,\Sigma_t) p(\mu_t,\Sigma_t\mid\mathcal{F}_{t-1})
$$

where

$$
p(\mu_t,\Sigma_t\mid\mathcal{F}_{t-1}) = \iint p(\mu_t,\Sigma_t\mid\mu_{t-1},\Sigma_{t-1}) p(\mu_{t-1},\Sigma_{t-1}\mid\mathcal{F}_{t-1}) d\mu_{t-1},d\Sigma_{t-1}
$$

Assume that the filtering distributions at step $t-1$ can be approximated as

$$
p(\mu_{t-1},\Sigma_{t-1}\mid\mathcal{F}_{t-1}) = q_{\mu}(\mu_{t-1}\mid\theta_{t-1}) q_{\Sigma}(\Sigma_{t-1}\mid\gamma_{t-1})
$$

A reasonable approximation for the dynamical model $p(\mu_t,\Sigma_t\mid\mu_{t-1},\Sigma_{t-1})$ is one such that we propagate expectations but increase variability while keeping the same functional form (similar to kalman, state space models) - the idea is that we do not have a definite law on how $\mu, \Sigma$ evolve and the baseline choice is that they stay the same in expectation but now we have less certainty about their values; This behavior is reflected on how parameters $\theta, \gamma$ change and, we will assume that the functional forms remain equal but with evolved parameters:

$$
p(\mu_t,\Sigma_t\mid\mathcal{F}_{t-1}) \approx q_\mu(\mu_t\mid \theta_t^-) q_\Sigma(\Sigma_t\mid \gamma_t^-)
$$

where $\theta_t^-$ and $\gamma_t^-$ denote the parameters of the predictive distributions before observing $y_t$.

After observing $y_t$, we compute the posterior using a mean-field variational approximation:

$$
p(\mu_t,\Sigma_t \mid \mathcal{F}_t) \approx Q_\mu(\mu_t) Q_\Sigma(\Sigma_t)
$$

The coordinate update for the covariance distribution is

$$
\log Q_\Sigma(\Sigma_t) = \mathbb{E}_{Q_\mu} \left[ \log p(y_t \mid \mu_t,\Sigma_t) \right] + \log q_\Sigma(\Sigma_t) + \text{const}
$$

which yields

$$
Q_\Sigma(\Sigma_t) \propto \exp\left(-\frac12\log|\Sigma_t| -\frac12 \text{tr} \left(\Sigma_t^{-1}S_t \right) + T_{q_\Sigma}(\Sigma_t) \right) $$

where

$$
S_t = \mathbb{E}_{Q_\mu} \left[(y_t-\mu_t)(y_t-\mu_t)^\top \right]
$$

and $T_{q_\Sigma}(\Sigma_t)$ collects the contribution of the predictive distribution $q_\Sigma$.

For most financial applications, the conditional expected return is typically much smaller than the realized return,

$$
|\mu_t| \ll |y_t|
$$

so that

$$
S_t \approx y_t y_t^\top
$$

Consequently,

$$
Q_\Sigma(\Sigma_t) \propto \exp\left(-\frac12\log|\Sigma_t| -\frac12 \text{tr} \left(\Sigma_t^{-1}y_ty_t^\top \right) + T_\Sigma(\Sigma_t) \right)
 $$

This approximation shows that, in practice, the covariance model can often be updated almost independently of the conditional mean model using only realized returns.

Once the covariance update has been performed, the conditional mean is updated according to

$$
\log Q_\mu(\mu_t) = \mathbb{E}_{Q_\Sigma} \left[\log p(y_t \mid \mu_t,\Sigma_t) \right] + \log q_\mu(\mu_t) + \text{const}
$$

or equivalently,

$$
Q_\mu(\mu_t) \propto \exp\left(-\frac12 (y_t-\mu_t)^\top \mathbb{E}_{Q_\Sigma} \left[\Sigma_t^{-1} \right] (y_t-\mu_t) + T_{q_{\mu}}(\mu_t) \right)
$$


and $T_{q_{\mu}}(\mu_t)$ denotes the contribution of the predictive distribution $q_\mu$.


### Posterior predictive

$$
p(y_{t+1} \mid \mathcal{F}_t) = \iint p(y_{t+1}\mid \mu_{t+1}, \Sigma_{t+1}) p(\mu_{t+1}, \Sigma_{t+1} \mid \mathcal{F}_t) \text{d}\mu_{t+1} \text{d}\Sigma_{t+1}
$$

Which for many cases can be approximated with a normal (with moment matching for example).


## Applications

Let's cover interesting cases of the formalism.

### Static parameters

The posterior satisfies

$$
p(\mu,\Sigma\mid\mathcal F_t) \propto p(y_t\mid\mu,\Sigma) p(\mu,\Sigma\mid\mathcal F_{t-1})
$$

Recursively

$$
p(\mu,\Sigma\mid\mathcal F_t) \propto p(\mu,\Sigma) \prod_{s=1}^{t} p(y_s\mid\mu,\Sigma)
$$

This colapses in simple gaussian estimation (with prior or not). Under the previous approximations we can estimate each one separately; also, if we have fixed parameters (let's say the coefficients of a regression) we can just estimate them normally. No need to elaborate on this.



### Roll covariance model

Assume there are no features $z$ to predict covariance. At $t-1$, we have

$$
p(\Sigma_{t-1}, \mu_{t-1}\mid \mathcal{F}_{t-1}) = q_{\mu}(\mu_{t-1}\mid\theta_{t-1}) \text{IW}(\Sigma_{t-1} \mid \nu_{t-1},V_{t-1})
$$

where $\text{IW}(\cdot)$ is a Inverse-Wishart distribution; $q_{\mu}$ is left generic.

As before, a reasonable approximation is where we _propagate_ to the same distribution with different parameters

$$
p(\mu_t,\Sigma_t\mid\mathcal{F}_{t-1}) \approx q_\mu(\mu_t\mid \theta_t^-) \text{IW}(\Sigma_{t}\mid\nu_{t}^-,V_{t}^-)
$$

with

$\nu_t^- = \phi(\nu_{t-1}-k-1) + k + 1$

$V_t^- = \phi V_{t-1}$

this preserves expected values and increases variability ($\phi$ is a hyperparameter).

Using the previous approximations, given that Inverse-Wishart is conjugate, one can write

$$
Q_{\Sigma}(\Sigma_t) = \text{IW}(\nu_t, V_t)
$$

with $\nu_t = \nu_t^- + 1$ and $V_t = V_t^- + y_t y_t^T$.


After many iterations

$$
\nu_{\infty} = \frac{1}{1-\phi} + k + 1
$$

$$
V_t = y_{t} y_{t}^T + \phi y_{t-1} y_{t-1}^T + \phi^2 y_{t-2} y_{t-2}^T + \cdots
$$

Which, from expectation of the inverse-whishart, implies

$$
\mathbb{E}[\Sigma_t] = (1-\phi)\left( y_{t} y_{t}^T + \phi y_{t-1} y_{t-1}^T + \phi^2 y_{t-2} y_{t-2}^T + \cdots \right)
$$

which is consistent. Correct formula for a roll estimator of covariance. This justifies the use of exponential weighted covariance estimation. Also

$$
\mathbb{E}[\Sigma_t] = (1-\phi) y_{t} y_{t}^T + \phi \mathbb{E}[\Sigma_{t-1}]
$$





### Static mean with roll covariance

If we aknowledge that covariance changes over time (and it is observed that it is autocorrelated in some sense) then we can ask how to calculate a static mean return. 

At $t-1$, we have

$$
p(\Sigma_{t-1}, \mu_{t-1} \mid \mathcal{F}_{t-1}) = \text{N}(\mu \mid m_{t-1},P_{t-1}) q_{\Sigma}(\Sigma_{t-1}\mid\gamma_{t-1})
$$

A reasonable model for this case is that the mean distribution are the same (after many iterations we expect them to have a distribution that just converges to the mean with decreasing variance with the number of observations) and the covariance model is evolved as before:

$$
p(\mu_t,\Sigma_t\mid\mathcal{F}_{t-1}) \approx \text{N}(\mu_{t} \mid m_{t_-1},P_{t-1}) q_{\Sigma}(\Sigma_{t}\mid\gamma_{t}^-)
$$


Then

$$
\log Q_{\mu}(\mu_t) =  -\frac12 (\mu-m_{t-1})^\top P_{t-1}^{-1} (\mu-m_{t-1}) -\frac12 (y_t-\mu)^\top W_t (y_t-\mu) + \text{const}
$$

with $W_t = \mathbb{E}_{Q\_\Sigma} \left[\Sigma_t^{-1} \right\]$. Therefore,

$$
Q_{\mu}(\mu_t)=N(m_t,P_t)
$$

with

$$
P_t^{-1} = P_{t-1}^{-1}+W_t
$$

and

$$
m_t = P_t \left( P_{t-1}^{-1}m_{t-1} + W_ty_t \right)
$$

This is the sequential posterior update for a fixed unknown mean with time-varying covariance. Repeated substitution gives

$$ 
P_t^{-1} = P_0^{-1} + \sum_{s=1}^{t}W_s
$$

Similarly,

$$
P_t^{-1}m_t = P_0^{-1}m_0 + \sum_{s=1}^{t}W_sy_s
$$

Hence,

$$
P_t = \left( P_0^{-1} + \sum_{s=1}^{t}W_s \right)^{-1}
$$

and

$$
m_t = \left( P_0^{-1} + \sum_{s=1}^{t}W_s \right)^{-1} \left( P_0^{-1}m_0 + \sum_{s=1}^{t}W_sy_s \right)
$$

With a diffuse prior,

$$
P_0^{-1}\rightarrow 0
$$

so

$$
m_t = \left( \sum_{s=1}^{t}W_s \right)^{-1} \left( \sum_{s=1}^{t}W_sy_s \right)
$$

Thus the posterior mean is a precision-weighted average of all observations.



### Roll mean model

Assume there are no features $x$ to predict mean. At $t-1$

$$
p(\Sigma_{t-1}, \mu_{t-1}\mid \mathcal{F}_{t-1}) = \text{N}(\mu_{t-1} \mid m_{t-1},P_{t-1}) q_{\Sigma}(\Sigma_{t-1}\mid\gamma_{t-1})
$$

We now evolve this distribution by preserving its mean and increasing its variance.

$$
p(\mu_t,\Sigma_t\mid\mathcal{F}_{t-1}) \approx \text{N}(\mu_{t} \mid m_{t}^-,P_{t}^-) q_{\Sigma}(\Sigma_{t}\mid\gamma_{t}^-)
$$

with $m_t^- = m_{t-1}$ and $P_t^- = \frac{1}{\psi} P_{t-1}$ ($\psi$ is scalar in $0<\psi<1$).

Let $W_t = \mathbb{E}_{Q\_\Sigma} \left[\Sigma_t^{-1} \right\]$. The mean update is

$$
\log Q_{\mu_t}(\mu_t) = -\frac12 (y_t-\mu_t)^\top W_t (y_t-\mu_t)  -\frac12 (\mu_t-m_{t-1})^\top \psi P_{t-1}^{-1} (\mu_t-m_{t-1}) + \text{const}
$$

which implies

$$
Q_{\mu_t}(\mu_t) = N(m_t,P_t)
$$



$$
P_t^{-1} = \psi P_{t-1}^{-1} + W_t
$$

$$
m_t = P_t \left( \psi P_{t-1}^{-1}m_{t-1} + W_ty_t \right)
$$

We can expand 

$$
P_t^{-1} = \psi^t P_0^{-1} + \sum_{s=1}^t \psi^{t-s} W_s
$$

$$
m_t = P_t \left( \psi^t P_0^{-1}m_0 + \sum_{s=1}^t \psi^{t-s}W_s y_s  \right)
$$


which, after many iterations can be approximated as

$$
m_t \approx \left( \sum_{s=1}^t \psi^{t-s} W_s \right)^{-1} \left( \sum_{s=1}^t \psi^{t-s}W_s y_s\right)
$$

This is an exponentially rolling mean in which every observation receives two weights:

$$
\psi^{t-s}
$$

for its age, and

$$
W_s=\mathbb E[\Sigma_s^{-1}]
$$

for its estimated precision. Also, one can use the approximation $W_s \approx \mathbb E[\Sigma_s]^{-1}$ (in many cases it's a good approximation).

This hints are a simple implementation: compute a rolling variance estimate, multiply that variance by the return observations at that time and then compute a rolling mean of this normalized by the sum of variances in the same window.

#### Using the inverse-Wishart covariance estimate

With

$$ 
\Sigma_t \sim \text{IW}(\nu_t,V_t)
$$

the expected precision is

$$
W_t = \mathbb E[\Sigma_t^{-1}] = \nu_tV_t^{-1}
$$

Therefore,

$$
P_t^{-1} = \psi P_{t-1}^{-1} + \nu_tV_t^{-1}
$$

and

$$
m_t = P_t \left( \psi P_{t-1}^{-1}m_{t-1} + \nu_tV_t^{-1}y_t \right)
$$

In the long-run representation,

$$
m_t \approx \left( \sum_{s=1}^{t} \psi^{t-s}\nu_sV_s^{-1} \right)^{-1} \left( \sum_{s=1}^{t} \psi^{t-s}\nu_sV_s^{-1}y_s \right)
$$


### Linear Regression

#### Univariate

For simplicity, consider we are modellig a single asset return ($k = 1$); at each instant there are some features $x_t$ predicitve of return $y_t$. A simple choice is to consider a linear model for expected value

$$
\mu_t = x_t^\top\beta
$$

Similar to the constant mean case, we are now interested in $\beta$ distribution after a bunch of observations. At $t-1$, we have 

$$
p(\Sigma_{t-1}, \beta_{t-1} \mid \mathcal{F}_{t-1}) = \text{N}(\beta \mid b_{t-1},U_{t-1}) q_{\Sigma}(\Sigma_{t-1}\mid\gamma_{t-1})
$$

Again, a reasonable model for this case is that the beta distribution is the same (after many iterations we expect them to have a distribution that just converges to the mean with decreasing variance with the number of observations) and the covariance model is evolved as before:

$$
p(\beta_t,\Sigma_t\mid\mathcal{F}_{t-1}) \approx \text{N}(\beta_{t} \mid b_{t-1},U_{t-1}) q_{\Sigma}(\Sigma_{t}\mid\gamma_{t}^-)
$$


The covariance model is estimated independently as before,

$$
q_\Sigma(\Sigma_t)
$$

After estimating the covariance, define

$$
w_t=\mathbb E[\sigma_t^{-2}]
$$


The posterior satisfies

$$
\log Q_\beta(\beta_t) = -\frac12(\beta-b_0)^\top U_0^{-1}(\beta-b_0) -\frac12 \sum_{s=1}^{t} w_s (y_s-x_s^\top\beta)^2 +\text{const}
$$

Collecting quadratic terms gives

$$
Q_\beta(\beta_t) = \text{N}(\beta_t \mid b_t,U_t)
$$

with

$$
U_t^{-1} = U_0^{-1} + \sum_{s=1}^{t} w_s x_sx_s^\top
$$

and

$$
b_t = U_t \left( U_0^{-1}b_0 + \sum_{s=1}^{t} w_s x_sy_s \right)
$$

With a diffuse prior, we obtain

$$
b_t = \left( \sum_{s=1}^{t} w_s x_sx_s^\top \right)^{-1} \left( \sum_{s=1}^{t} w_s x_sy_s \right)
$$

which is weighted least squares estimator. Of course one can use a different prior and get a ridge or lasso regression and even optimize hyperparameters with type-II MLE - what matters is that observations are weighted differently according the volatility estimate.







#### Rolling Univariate

A simple extension is to consider

$$
p(\beta_t,\Sigma_t\mid\mathcal{F}_{t-1}) \approx \text{N}(\beta_{t} \mid b_{t}^-,U_{t}^-) q_{\Sigma}(\Sigma_{t}\mid\gamma_{t}^-)
$$

with

$$
b_t^- = b_{t-1}
$$

and

$$
U_t^- = \frac1\psi U_{t-1}
$$

where

$$
0<\psi<1.
$$

Again let

$$
w_t =\mathbb E[\sigma_t^{-2}].
$$



The posterior is


$$
Q_\beta(\beta_t) = \text{N}(\beta_t \| b_t,U_t)
$$

where

$$
U_t^{-1} = \psi U_{t-1}^{-1} + w_tx_tx_t^\top
$$

and

$$
b_t = U_t \left( \psi P_{t-1}^{-1}b_{t-1} + w_tx_ty_t \right)
$$

This is recursive least-squares with exponential forgetting and observation-specific precision weights.


Repeated substitution gives

$$
U_t^{-1} = \psi^tU_0^{-1} + \sum_{s=1}^{t} \psi^{t-s} w_sx_sx_s^\top
$$

and

$$
b_t = U_t \left( \psi^tU_0^{-1}b_0 + \sum_{s=1}^{t} \psi^{t-s} w_sx_sy_s \right)
$$

Ignoring the prior after sufficient observations

$$
b_t = \left( \sum_{s=1}^{t} \psi^{t-s} w_sx_sx_s^\top \right)^{-1} \left( \sum_{s=1}^{t} \psi^{t-s} w_sx_sy_s \right)
$$

#### Multiple regression

To build a linear model with multiple targets variables we need to find a matrix $B$ of coefficients. If covariance is uniform across observations this is known to colapse into a simple regression for each target individually; if (a non diagonal) covariance changes over time things get more complicated as now the model needs to be estimated with all targets at the same time. Since correlations between targets may be unstable as well probably this case is not that interesting to be covered.

