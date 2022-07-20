# Train HMM with multiple observations sequences


The other posts relating to HMM focused on fitting the model to a long and unique sequence. Even the model architectures that were discussed focus on this type of application: any state can be reached from the current state (the transition matrix allows for this).

Let us consider that we want to model a short sequence but we have many observations of this process, i.e, many short sequences. In a market context, we may want to model the weekly/monthly sequences of daily market returns (this idea can be simply extrapolated to other periods and/or data frequencies); in this case we have many short sequences (for example, we may want to build a model that models a full month of daily returns and so we have 12 observations per year. When the month ends, the model restarts). Also, the training sequences may not have all the same lenght.



Figure below shows a HMM where all states can be visited after a finite number of steps.

![png](/images/mshmm/regular_hmm.png)

The objective of this post is to adapt the estimation to train models like the one on the figure below. The states go from left to right and after this it cannot change. This can be a good idea to model short sequences with many observations.

![png](/images/mshmm/lr_hmm.png)

A more realistic example, we may want to model returns durig a period (week, month), and it can make sense to make the model accomodate different behaviours (maybe volatile months have a different model). This is illustrated in the next figure; as the chain is unfolded (i.e, as we get observations) the model may switch to a different state path (which will have different emission properties).

![png](/images/mshmm/lr_hmm2.png)


In theory, the only thing we have to do is to set some transitions probabilities to zero (in matrix $A$) and then, during the re-estimation, these entries will stay zero. This way we can force the state path to go on a certain path. 

As said, this type of model have short sequences, and so we need to change the re-estimation formulas.

## Modification of the algorithm


Consider that we have $L$ sequences, each one with lenght $N_l$. As in the other post, the problem is still to find parameters that maximize $p(D|M)$; since all sequences are independent, we have that:

$p(D\|M)=\Pi_l p(L_l\|M)$

where $L_l$ is the sequence $l$.

In log space (check other post) this means that all quantities will be aditive. For each subsequence (each $l$) we can compute the $\alpha$ and $\beta$ variables independently (there is not change here appart from having $L$ different quantities).

Recall that

$\gamma_t(k)=\frac{\alpha_t(k) \beta_t(k)}{p(X)}$

and

$\xi_t(i,j) = \frac{\alpha_t(i) A_{ij} p(x_{t+1}|j) \beta_{t+1}(j) }{p(X)}$


Now we should have a $\gamma$ and $\xi$ for each subsequence, i.e, $\gamma_t(k)^l$ and $\xi_t(i,j)^l$.


The re-estimation formula for $A$ now becomes:

$A_{ij} = \frac{\sum_l \sum_t \xi_t(i,j)^l}{\sum_l \sum_t \gamma_t(k)^l}$


For the re-estimation of the emission distributions parameters, the formula does not change; is we consider all the subsequences as one long one, we can concatenate the subsequences $\gamma$ and proceed with the re-estimation of the parameters (under this long concatenated sequence). This is easy to see from the formula of $T_3$ in the other post.


### Scaling

As before, we have a problem with underflow. Using the scaled variables for each one of the subsequences we can write:

$A_{ij} = \frac{\sum_l \sum_t \hat \alpha_t(i)^l A_{ij} p(x_{t+1}\|j) \hat \beta_{t+1}(j)^l }{\sum_l \sum_t \hat \alpha_t(i)^l  \hat \beta_{t}(j)^l / c_t}$





## Numerical implementation

Now we will implement the previous ideas. Most of the code follows from the other post; the re-estimation is remade to accomodate the multiple sequences.

First, 30 sequences with different lenghts from a HMM are generated. This process has three states that can only stay withing themselfs or jump to the next one (check matrix $A$ in code).

For the algorithm, the subsequences are concatenated in a single large sequence and the indexes where each one starts is recorded (this is used as a variable in the re-estimation function).



```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def simulate_hmm(n,A,P,mu,scale):
    states=np.arange(A.shape[0],dtype=int)
    z=np.zeros(n,dtype=int)
    x=np.zeros(n)
    z[0]=np.random.choice(states,p=P)
    x[0]=np.random.normal(mu[z[0]],scale[z[0]])
    for i in range(1,n):
        z[i]=np.random.choice(states,p=A[z[i-1]])
        x[i]=np.random.normal(mu[z[i]],scale[z[i]])
    return x,z        
```


```python
# Simulation parameters
A=np.array([[0.9,0.1,0],[0,0.9,0.1],[0,0,1]])
P=np.array([1,0,0])
mu=np.array([1,0,-1])
scale=np.array([0.5,0.1,0.5])
k=30
n_min=5
n_max=15
o=[]
for i in range(k):
    n=np.random.randint(n_min,n_max)
    x,z=simulate_hmm(n,A,P,mu,scale)
    o.append(x)
    plt.plot(x)
plt.grid(True)
plt.show()
```


![png](/images/mshmm/output_12_0.png)



```python

class GaussianEmmission(object):
    '''
    Simple class with a Gaussian emission for an HMM
    '''
    def __init__(self,mu,scale):
        self.mu=mu
        self.scale=scale
    
    def view(self):
        print('Gaussian Emission N(%s,%s)'%(self.mu,self.scale))
    
    def probability(self,x):
        '''
        Compute probability of each observation
        in x given the parameters
        '''
        return np.exp(-0.5*np.power((x-self.mu)/self.scale,2))/(self.scale*np.sqrt(2*np.pi))
    
    def reestimate(self,x,gamma):
        '''
        Re-estimate parameters under gamma
        '''
        d=np.sum(gamma)
        mu=np.dot(x,gamma)
        mu/=d
        var=np.dot(np.power(x-mu,2),gamma)/d
        scale=np.sqrt(var)
        self.mu=mu
        self.scale=scale        

def forward(prob,A,P):
    n_states=A.shape[0]
    n_obs=prob.shape[0]
    alpha=np.zeros((n_obs,n_states),dtype=np.float)
    c=np.zeros(n_obs,dtype=np.float)
    alpha[0]=P*prob[0]
    c[0]=1/np.sum(alpha[0])
    alpha[0]*=c[0]
    for i in range(1,n_obs):
        alpha[i]=np.sum(A*alpha[i-1][:,None],axis=0)*prob[i] 
        c[i]=1/np.sum(alpha[i])
        alpha[i]*=c[i]
    return alpha,c

def backward(prob,A,c):
    n_states=A.shape[0]
    n_obs=prob.shape[0]
    beta=np.zeros((n_obs,n_states),dtype=np.float)
    beta[-1]=np.ones(n_states,dtype=np.float)
    beta[-1]*=c[-1]
    for i in range(n_obs-2,-1,-1):
        beta[i]=np.sum(A*(prob[i+1]*beta[i+1]),axis=1)
        beta[i]*=c[i]
    return beta

def baum_welch_step_multi(probs,A,P,idx=None):
    '''
    Iteration step of the Baum-Welch algorithm
    for multiple sequences
    returns: ml,A,P,gamma
    '''
    if idx is None:
        idx=np.array([0,probs.shape[0]])        
    n_seqs=idx.size-1
    n_states=A.shape[0]
    # for A re-estimation
    aux_a=np.zeros((n_seqs,n_states,n_states))
    aux_p=np.zeros((n_seqs,n_states))
    # for A re-estimation
    aux_gamma=np.zeros((n_seqs,n_states))
    gamma=np.zeros((probs.shape[0],n_states))
    ml=0    
    for l in range(1,n_seqs):
        # forward-backward on each sequence
        prob=probs[idx[l-1]:idx[l]]
        n_obs=prob.shape[0]
        alpha,c=forward(prob,A,P)
        beta=backward(prob,A,c)
        # model likelihood
        ml+=(-1*np.sum(np.log(c)))        
        seq_gamma=alpha*beta
        seq_gamma/=c[:,None]
        gamma[idx[l-1]:idx[l]]=seq_gamma
        A_tensor=np.array([A])
        A_tensor=np.tile(A_tensor,(n_obs-1,1,1))
        xi=(prob[1:]*beta[1:])[:,None]*A_tensor*alpha[:-1][:,:,None]        
        a=np.sum(xi,axis=0)
        gamma_norm=np.sum(seq_gamma[:-1],axis=0)
        aux_a[l-1]=a
        aux_gamma[l-1]=gamma_norm    
        aux_p[l-1]=seq_gamma[0]
    # reestimate A
    a=np.sum(aux_a,axis=0)
    a/=np.sum(aux_gamma,axis=0)[:,None]
    # re-estimate P
    p=np.sum(aux_p,axis=0)
    p/=np.sum(p)
    return ml,a,p,gamma  
```


```python
# Initial train parameters
A=np.array([[0.6,0.4,0],[0,0.6,0.4],[0,0,1]])
P=np.array([0.8,0.2,0])
mu=np.array([0.1,0.5,-0.1])
scale=np.array([0.5,0.2,0.5])

n_states=3
n_iter=50
emissions=[]
for i in range(mu.size):
    emissions.append(GaussianEmmission(mu=mu[i],scale=scale[i]))

print('Initial parameters')
print('P')
print(P)
print('A')
print(A)
print('EMISSIONS')
for e in emissions:
    e.view()

# FROM OBSERVATIONS o BUILD ARRAYS
# now we have a list of observations
# build array with sizes
sizes=np.zeros(len(o)+1,dtype=int)
o_fixed=[]
for i in range(1,len(o)+1):
    o_fixed.append(o[i-1][:,None])
    sizes[i]=o[i-1].shape[0]
idx=np.cumsum(sizes)
x=np.vstack(o_fixed)  

# ITERATE EM ALGORITHM
n_iter=20
n_states=3

# store model likelihood here
ml_history=[]

for i in range(n_iter):
    # evaluate prob
    prob=np.zeros((x.shape[0],n_states))
    for j in range(n_states):
        prob[:,j]=emissions[j].probability(x).ravel()
    
    ml,A,P,gamma=baum_welch_step_multi(prob,A,P,idx)

    x=np.hstack(o)
    # reestimate distributions parameters with gamma
    for j in range(n_states):
        emissions[j].reestimate(x,gamma[:,j])
    ml_history.append(ml)

print()
print()
print('Final parameters')
print('P')
print(P)
print('A')
print(A)
print('EMISSIONS')
for e in emissions:
    e.view()
    
# plot convergence of model probability
plt.title('log P(X|M) convergence')
plt.plot(ml_history,'.-')
plt.grid(True)
plt.show()


```

    Initial parameters
    P
    [0.8 0.2 0. ]
    A
    [[0.6 0.4 0. ]
     [0.  0.6 0.4]
     [0.  0.  1. ]]
    EMISSIONS
    Gaussian Emission N(0.1,0.5)
    Gaussian Emission N(0.5,0.2)
    Gaussian Emission N(-0.1,0.5)
    
    
    Final parameters
    P
    [1.00000000e+00 6.07871171e-50 0.00000000e+00]
    A
    [[0.86951023 0.13048977 0.        ]
     [0.         0.90152824 0.09847176]
     [0.         0.         1.        ]]
    EMISSIONS
    Gaussian Emission N(1.012560556728482,0.4573534013574739)
    Gaussian Emission N(0.002653204629133477,0.09492264902444446)
    Gaussian Emission N(-0.9381922289792377,0.44175877964709603)



![png](/images/mshmm/output_14_1.png)


We can observe that the procedure converged and recovered the parameters used to generate the sequences (as desired).
