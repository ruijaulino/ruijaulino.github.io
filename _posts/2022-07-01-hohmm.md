# Higher Order HMM



An idea to extend the HMM model is to consider that the next state depends on more previous states than just the previous one: 

$p(z_t\|z_1,\cdots,z_{t-1})=p(z_t\|z_{t-\tau},\cdots,z_{t-1})$

As in the ordinary HMM, the observations depend only on the current state: 
$p(x_t\|z_t)$.

This problem now involves a more intrincate _transition matrix_ (or a tensor) and allows to model, for example, a more rich set of state durations distributions as the exponential implied in the first order HMM.

As a initial example, let us consider for now, the case where 
$p(z_t\|z_{t-2},z_{t-1})$, 
i.e, the current state depends on the previous two states (second order HMM); for this model, we need a tensor 
$a_{i,j,k}=p(z_k=k\|z_{t-1}=j,z_{t-2}=i)$ 
to model the transitions probabilities and to reformulate the HMM re-estimation formulas.

To avoid this, there is a nice trick that we can do to estimate a higher order HMM using the first order HMM. This makes the problem quite simple to solve.

First, lets simulate a second order HMM. To help make the simulation, in code, there is a variable 
$Q$ 
with the possible combinations of observations of 
$\{z_{t-2},z_{t-1}\}$; 
this time, the variable (transition matrix) 
$A$ 
is no longer a square matrix, but a matrix where each row represents the probability to go to a state given that we observed the previous states sequence in the corresponding row in 
$Q$.




```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def simulate_hohmm(n,A,P,mu,scale,Q):
    
    n_states=A.shape[1]
    states=np.arange(n_states,dtype=int)
    
    z=np.zeros(n,dtype=int)
    x=np.zeros(n)
    
    order=Q.shape[1]
    
    for i in range(order):
        z[i]=np.random.choice(states,p=P)
        x[i]=np.random.normal(mu[z[i]],scale[z[i]])

    for i in range(order,n):
        # get previous two states in a array
        q=z[i-order:i]
        # get the index of Q where this pattern occured
        idx=np.where((Q==q).all(axis=1))[0][0]
        # the transition probability is in A[idx]
        z[i]=np.random.choice(states,p=A[idx])
        x[i]=np.random.normal(mu[z[i]],scale[z[i]])
    return x,z

# next probability depends on previous two states
Q=np.array([
    [0,0], # 0,0
    [0,1], # 0,1
    [1,0], # 1,0
    [1,1] # 1,1
    ],dtype=int)
# Q contains the meaning of each row in A -> A[i]=p( z[i] | z[i-1],z[i-2] = Q[i] )
A=np.array([
            [0.9,0.1],
            [0.6,0.4],
            [0.3,0.7],
            [0.2,0.8]
            ])

P=np.array([0.5,0.5])
mu=np.array([-1,1])
scale=np.array([0.5,0.5])
n=1000

x,z=simulate_hohmm(n,A,P,mu,scale,Q)
ax=plt.subplot(211)
plt.title('States Sequence')
plt.plot(z,color='r')
plt.xticks([])
ax=plt.subplot(212)
plt.title('Observations Sequence')
plt.plot(x)
plt.show()
  
```


    
![png](/images/hohmm/output_3_0.png)
    


##### Reduction to a conventional HMM

The idea of the trick to solve the high-order HMM using a single order HMM, is consider some artificial states 
$q$, 
that, when the dynamics are expressed in their terms, we have a first order markov chain. 

Let us build a new set of variables/states 
$Q=\{q_1,q_2,\cdots \}$ 
where each 
$q_i$ 
is a combination of two states from 
$Z=\{z_1,z_2,\cdots\}$. 
For the second order HMM (with 
$Z=\{z_1,z_2\}$, 
i.e, there are two distinct states; generatization of this should be easy from here), the states 
$q_i$ 
are:

$q_1=\{ z_1,z_1\}$

$q_2=\{ z_1,z_2\}$

$q_3=\{ z_2,z_1\}$

$q_4=\{ z_2,z_2\}$


To be clear, when we observe the sequence 
$S=z_1,z_1,z_2,z_1,z_2$, 
in 
$Q$ 
terms, this is equivalent to say that we observe 
$S=q_0,q_1,q_2,q_3,q_2$ 
(where the 
$q_0$ 
is there to emphasize that we have the same number of observations but here we put some random state 
$q$).


Now, we can observe that the sequence written in term of variables 
$q_i$ 
is a first order markov chain: if the states 
$z$ 
have two lags of dependency, then, since we defined each 
$q$ 
as two successive 
$z$ 
state values, then 
$p(q_t\|q_1,\cdots,q_{t-1})=p(q_t\|q_{t-1})$. 
The regular HMM written in terms of 
$q$ 
should work.


Now we are left with adjusting the regular algorithm to work under 
$q$. 
In general there will be 
$K^{\tau}$ 
different 
$q$ 
states (
$K$ 
is the number of distinct states in 
$z$ 
and 
$\tau$ 
the number of lags/order); for two states, second order HMM, we have 
$2^2=4$ 
different 
$q$ 
(as is described above for the example). 

On transition matrix 
$A$, 
$A_{ij}$ 
will correspond to 
$p(q_t=q_j\|q_{t-1}=q_i)$, 
but, it is possible to observe that not all transitions are allowed. For example (and considering the previous case), 
$q_1=\{ z_1,z_1\}$, 
can only go to 
$q_1$ 
or 
$q_2$; 
since the next 
$z$ observation can be 
$z_1$ 
or 
$z_2$ 
we are left with 
$\{ z_1,z_1\}$ 
or 
$\{ z_1,z_2\}$ 
(keep the last element of 
$q_1$
) which is 
$q_1$ 
or 
$q_2$. 

This information can be included in the (square) matrix 
$A$ 
by setting to zero impossible transitions. If an entry of 
$A$ 
is zero it's value is not changed on the reestimation procedure.


Again, for this example, the final (converged) 
$A$ 
matrix entries can be interpreted as the second order transition probabilities (this will be made more clear with the code example although it is easy to check). 


Regarding the observations probabilities, we only have as much states as there are in 
$z$. 
We can notice, however, that, the emission distribution for 
$q_1$ 
is the same emission as for 
$z_1$: 
in general, observation distribution for 
$q$ 
is the same as the one of the last state 
$z$ 
represented in 
$q$. 
On the code we do this by replicating/copying the probabilities given 
$z$ 
to the corresponding 
$q$.


Finally, for the reestimation of distribution parameters, we need the 
$\gamma$ 
which means the probabilities of the 
$z$ 
states given the observations. For example, for 
$z_1$, 
it's 
$\gamma$ 
can be obtained by summing the 
$\gamma$ 
for 
$q_1$ 
and 
$q_3$. 

Let us implement the idea to estimate the parameters of the data generated previously.



```python
# REGULAR HMM CODE 
def forward(prob,A,P):
    '''
    Forward Procedure 
    Calculates the probability of the sequence given the model
    alpha[i] = Pr(O1 O2 ... Ot,state @ t = state i | model)
    prob: numpy (n_obs,n_states) array with the probability
        of the observations given each state (each column is this probability)
    A: numpy (n_states,n_states) array with the state transition probability distribution
    P: numpy (n_states,) array with the initial state distribution

    output
        alpha: numpy (n_obs,n_states) array with the
            renormalized alphas at each period 
        c: numpy (n_obs,) array with the renormalization
            factor at each time stamp
    '''
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
    '''
    Backward Procedure 
    Calculates the probability 

    beta[i] = Pr(Ot+1 Ot+2 ... OT | state @ t = state i , model)

    prob: numpy (n_obs,n_states) array with the probability
        of the observations given each state (each column is this probability)
    A: numpy (n_states,n_states) array with the state transition probability distribution
    c: numpy (n_obs,) array with the renormalization factor at each time stamp
        ** from forward() **
    output
        beta: numpy (n_obs,n_states) array with the
            betas at each period
    '''
    n_states=A.shape[0]
    n_obs=prob.shape[0]
    beta=np.zeros((n_obs,n_states),dtype=np.float)
    beta[-1]=np.ones(n_states,dtype=np.float)
    beta[-1]*=c[-1]
    for i in range(n_obs-2,-1,-1):
        beta[i]=np.sum(A*(prob[i+1]*beta[i+1]),axis=1)
        beta[i]*=c[i]
    return beta

def baum_welch_step(prob,A,P):
    '''
    Iteration step of the Baum-Welch algorithm
    returns: ml,A,P,gamma
    '''
    n_states=A.shape[0]
    n_obs=prob.shape[0]
    alpha,c=forward(prob,A,P)
    beta=backward(prob,A,c)
    gamma=alpha*beta
    gamma/=c[:,None]
    ml=np.sum(gamma[0])
    gamma/=ml
    A_tensor=np.array([A])
    A_tensor=np.tile(A_tensor,(n_obs-1,1,1))
    xi=(prob[1:]*beta[1:])[:,None]*A_tensor*alpha[:-1][:,:,None]
    xi/=ml
    p=gamma[0]
    A=np.sum(xi,axis=0)/np.sum(gamma[:-1],axis=0)[:,None]
    ml=-1*np.sum(np.log(c))
    return ml,A,P,gamma

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
```


```python
# SOLVE SECOND ORDER HMM BY REDUCTION TO A FIRST ORDER HMM

# z states are {0,1}

# define model in terms of states q
q=np.array([
    [0,0], 
    [0,1], 
    [1,0], 
    [1,1] 
    ],dtype=int)

# transition matrix does not allow for all transitions
# if they are set to zero then they will stay zero throughout all iteration of EM
# initialize non zero entries at some random value
A_est=np.array([
                [0.6,0.4,0,0],
                [0,0,0.4,0.6],
                [0.4,0.6,0,0],
                [0,0,0.4,0.6],
                ])
# intialize P with equal prob
P_est=np.array([0.25,0.25,0.25,0.25])
emissions=[]
# initialize first emission
emissions.append(GaussianEmmission(mu=-0.1,scale=0.1))
emissions.append(GaussianEmmission(mu=0.1,scale=0.1))

print('Initial parameters')
print('P')
print(P_est)
print('A')
print(A_est)
print('EMISSIONS')
for e in emissions:
    e.view()

# ITERATE EM ALGORITHM
n_iter=50
n_states=2

# store model likelihood here
ml_history=[]
for i in range(n_iter):
    # evaluate prob
    prob=np.zeros((x.size,n_states))
    for j in range(n_states):
        prob[:,j]=emissions[j].probability(x)
    # duplicate prob to attribute to corresponding q
    prob=prob[:,q[:,1]] 
    # first order HMM Baum Welch
    ml,A_est,P_est,gamma=baum_welch_step(prob,A_est,P_est)
    # reestimate distributions parameters with gamma summed for corresponding q
    for j in range(n_states):
        gamma_z=np.sum(gamma[:,np.where(q[:,1]==j)[0]],axis=1)
        emissions[j].reestimate(x,gamma_z)
    ml_history.append(ml)

print()
print()
print('Final parameters')
print('P')
print(P_est)
print('A')
print(np.round(A_est,3) )
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
    [0.25 0.25 0.25 0.25]
    A
    [[0.6 0.4 0.  0. ]
     [0.  0.  0.4 0.6]
     [0.4 0.6 0.  0. ]
     [0.  0.  0.4 0.6]]
    EMISSIONS
    Gaussian Emission N(-0.1,0.1)
    Gaussian Emission N(0.1,0.1)
    
    
    Final parameters
    P
    [0.25 0.25 0.25 0.25]
    A
    [[0.885 0.115 0.    0.   ]
     [0.    0.    0.659 0.341]
     [0.245 0.755 0.    0.   ]
     [0.    0.    0.182 0.818]]
    EMISSIONS
    Gaussian Emission N(-1.0231418012430926,0.4805971781036449)
    Gaussian Emission N(1.0052451496372294,0.4966806826849338)
    


    
![png](/images/hohmm/output_6_1.png)
    


##### Comments

We can see that the algorithm found the original state transitions for the second order HMM. We need to interpret matrix 
$A$ 
row by row and removing the zero entries (which we set to zero at the begining of the iterations): then this matrix 
$A$ 
matches the one that was used to generate the observations. Also, the emissions have similar parameters. 

For the general case, it is usefull to have a code to build the 
$Q$ 
variables for any number of states in 
$z$ 
and any number of lags. After that the algorithm is exactly the same. Below are some utilities to do this.



```python
# UTILITIES FOR THE GENERAL CASE 
# not quite efficient code but does the job
# this only have to be done once anyway

# INITIALIZE A
n_states=3
z_lags=2



def create_q(n_states,z_lags):
    '''
    create q array with the combinations of states and lags
    ** these are the new states to solve the first order HMM problem **
    '''
    def aux(n_states,obj):
        q=[]
        for m in range(n_states):
            if len(obj)==0:
                q.append(m)
            for e in obj:
                if not isinstance(e,list):
                    q.append([e,m])
                else:
                    q.append(e+[m])            
        return q
    q=[]
    for lag in range(z_lags):
        q=aux(n_states,q)
    q=np.array(q,dtype=int)
    q=q[:,::-1]
    return q

def A_init(q,n_states):
    '''
    Initialize the A (square) matrix with the
    suitable entries set to zero
    ** the initialization is random **
    '''
    # INITIALIZE A (transitions matrix for q)
    A=np.zeros((q.shape[0],q.shape[0]))
    for i in range(q.shape[0]):
        for j in range(n_states):
            next_q=np.hstack((q[i][1:],j))
            a_idx=np.where( (q==next_q).all(axis=1) )[0][0]
            A[i,a_idx]=np.abs(np.random.normal())
    A/=np.sum(A,axis=1)[:,None]
    return A

def compressA(A,q,n_states):
    '''
    "compress" A matrix from re-estimation procedure to be interpreted as the original transitions
    ** not really necessary **
    '''
    # use q and n_states to generate a random A to avoid numerical issues
    A_tmp=A_init(q,n_states)
    T=[]
    for i in range(A.shape[0]):
        T.append(A[i][np.where(A_tmp[i]!=0)[0]])
    T=np.array(T)
    return T


# EXAMPLE OF USE
# it should be easy to see how this can be used along with the previous code
# ** the variable names are kept consistent **
q=create_q(n_states,z_lags)
print('** CHANGE PARAMETERS AT WILL TO CHECK WHAT HAPPENS **')
print('Number of z states = ', n_states)
print('Number of lags = ', z_lags)
print('** q states to solve the problem in terms of z states **')
print(q)
print('There are %s new states'%q.shape[0])

print('** possible initialization of A matrix (notice the zero entries) **')
A=A_init(q,n_states)
print(np.round(A,3))

print('** AFTER THE RE-ESTIMATION IS DONE **')
print('this can be used to interpret in a simpler way the estimated A matrix')
print('each row represents the probability to go a given z states (columns) after we observe a q state')
print('(this is just the previous matrix without the zeros)')
print(compressA(A,q,n_states))

```

    ** CHANGE PARAMETERS AT WILL TO CHECK WHAT HAPPENS **
    Number of z states =  3
    Number of lags =  2
    ** q states to solve the problem in terms of z states **
    [[0 0]
     [0 1]
     [0 2]
     [1 0]
     [1 1]
     [1 2]
     [2 0]
     [2 1]
     [2 2]]
    There are 9 new states
    ** possible initialization of A matrix (notice the zero entries) **
    [[0.109 0.845 0.046 0.    0.    0.    0.    0.    0.   ]
     [0.    0.    0.    0.546 0.279 0.176 0.    0.    0.   ]
     [0.    0.    0.    0.    0.    0.    0.015 0.821 0.164]
     [0.209 0.306 0.485 0.    0.    0.    0.    0.    0.   ]
     [0.    0.    0.    0.063 0.301 0.636 0.    0.    0.   ]
     [0.    0.    0.    0.    0.    0.    0.547 0.34  0.114]
     [0.642 0.139 0.219 0.    0.    0.    0.    0.    0.   ]
     [0.    0.    0.    0.17  0.318 0.512 0.    0.    0.   ]
     [0.    0.    0.    0.    0.    0.    0.486 0.281 0.232]]
    ** AFTER THE RE-ESTIMATION IS DONE **
    this can be used to interpret in a simpler way the estimated A matrix
    each row represents the probability to go a given z states (columns) after we observe a q state
    (this is just the previous matrix without the zeros)
    [[0.10923417 0.84475999 0.04600584]
     [0.54573887 0.2785992  0.17566193]
     [0.01470497 0.82138609 0.16390894]
     [0.20905456 0.30620169 0.48474375]
     [0.06316007 0.30113876 0.63570117]
     [0.54654924 0.33951845 0.11393231]
     [0.64157826 0.13945684 0.21896491]
     [0.16967744 0.31844578 0.51187678]
     [0.4864303  0.28135635 0.23221334]]
    


```python

```
