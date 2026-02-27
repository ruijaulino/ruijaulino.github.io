# Approximations to allocation

Estimation of parameters and numerical problems make the optimal allocation problem quite difficult to solve. For practical applications it is logical to make approximations to have more robust solutions (ignore correlations for example). The objective here is to show some approximations that can be made to produce usefull and intuitive formulas. 

The setup is one where we have some features to make predictions on future returns. As discussed in other posts, when presented with new information/features $x \in \mathbb{R}^q$, the optimal allocation on returns $y \in \mathbb{R}^p$ is

$w = C_{y\|x}^{-1} \mu_{y\|x}$


Consider a joint gaussian distribution between assets and features $y, x$. In this case, the conditional distribution is

$p(y\|x) = N \left( \mu_y + C_{yx}C_{xx}^{-1}(x-\mu_x), C_{yy} - C_{yx}C_{xx}^{-1}C_{xy} \right)$

#### Some notation

For a generic part of the joint covariance matrix $C$ we can always write $C_{ab} = S_{ab}R_{ab}S_{ab}$ where $S$ is a matrix with the scale and $R$ is a matrix with the correlations (may not be square matrices, dimensions just have to match).

In general, a (square) correlation can be written as

$R = I + \epsilon (R-I) =: I + \epsilon r$

when $\epsilon \rightarrow 1$ we recover $R$. This represents some (necessary) regularizationn that we can do to correlation to aleviate estimation issues. 

Also, we can approximate

$R^{-1} \approx I - \epsilon r + \epsilon^2 r^2 = I - \epsilon r (I-\epsilon r)$

We will use the notation 
- $\rho$ to refer to the the $p \times q$ matrix $R_{yx}$ (feature/target correlations)
- $r$ to refer to the $p \times p$ matrix $r_{yy}$ (correlation matrix between targets with zeros on the diagonal) 
- $\pi$ to refer to the $q \times q$ matrix $r_{xx}$ (correlation matrix between features with zeros on the diagonal)


$\blacksquare$

Under this definitions we can write the following

$C_{y\|x} = S_y \left( R_{yy} - R_{yx} R_{xx}^{-1} R_{xy} \right) S_y$

and

$\mu_{y\|x} = \mu_y + S_y R_{yx} R_{xx}^{-1} S_x^{-1} (x-\mu_x)$

Let us make approximations such that we can generate intuition about what determines the allocation.

The typical case is that $R_{yx}$ is small as correlation with the targets $y$ are small (low signal) but $R_{yy}$ and $R_{xx}$ may be large (similar features, correlated targets) and possibly second order effects exist.


For the conditional covariance, since the product $R_{yx} R_{xx}^{-1} R_{xy}$ is always second order small:

$C_{y\|x} \approx S_{y} R_{yy} S_{y} = C_{yy}$

and 
$C_{y\|x}^{-1} \approx I - \epsilon r (I - \epsilon r )$





For the conditional expected value

$\mu_{y\|x} \approx \mu_y + S_y R_{yx} \left( I - \epsilon \pi (I - \epsilon \pi)  \right) z_x$

where $z_x = S_x^{-1} (x-\mu_x)$ is the normalized/standardized feature value.



Putting together the terms, optimal weight is approximated up to second order as:

$w \approx S_y^{-1} \left[ I - \epsilon_r r (I-\epsilon_r r)\right] S_y^{-1} \left[ \mu_y + S_y R_{yx} \left( I - \epsilon_\pi \pi (I-\epsilon_\pi \pi) \right) z_x \right]$

We can study this by abstracting the expected value part writting


$w \approx S_y^{-1} \left[ I - \epsilon_r r (I-\epsilon_r r)\right] S_y^{-1} m$

with $m = \mu_y + S_y R_{yx} \left( I - \epsilon_\pi \pi (I-\epsilon_\pi \pi) \right) z_x $


### Target correlation effect

The first level of complexity that we can go into after considering the zeroth order $w_i = \frac{m_i}{\sigma_i^2}$ is

$w \approx S_y^{-1}\left( I - \epsilon_r r \right) S_y^{-1} m$

Another to write this is:

$w_i^{\text{first}} \approx \frac{1}{\sigma_i} \left(s_i - \epsilon_r \sum_{j \neq i} r_{ij}s_j \right)$

with $s_i = \frac{\mu_i}{\sigma_i}$ the sharpe ratio of asset $i$. This can be interpreted as a the optimal individual solution penalized with the sharpe-weighted correlation with the other assets; if an asset has positive correlation with the others (expected values have something to say here as a negative expected value on a positive correlated asset has the effect of a negative correlated asset) then it's weight get smaller. By itself, this result can be seen as a numerically stable way to solve allocation between assets. In practice, we can set up the regularization $\epsilon_r$ such that weights do not go against expected values ($\epsilon_r$ must be verify $\epsilon_r < \frac{\|s_i\|}{\|\sum_{j \neq i} r_{ij} s_j\|}$ for all $i$). This ensures that this is effectivelly a _correction_ term.

#### Equal Sharpes

To make a connection with risk parity solutions, let us consider all sharpe ratios $s$ equal. Then:

$w_i \propto \frac{1-\epsilon \sum_{j \neq i}r_{ij}}{\sigma_i}$

This is similar to risk parity but we make a small correction due to correlation. 

#### Second Order

We can use the second order approximation:

$w^{\text{second}} \approx S_y^{-1} \left[ I - \epsilon_r r \left( I - \epsilon_r r \right) \right] S_y^{-1}\mu_y$

Using the previous result for the first order approximation


$w_i^{\text{second}} \approx \frac{1}{\sigma_i} \left(s_i - \epsilon_r \sum_{k \neq i} r_{ik} \sigma_k w_k^{\text{first}} \right)$

which is similar to the first order solution but using a _sharpe ratio_ ($\sigma_k w_k^{\text{first}}$ is similar to a sharpe as $w \sim \mu/\sigma^2$) implied by the first order solution. Second order correction can be seen as adding back some weight: in first order we reduced allocation on $A$ because it is correlated with $B$ and reduced allocation on $B$ because it is correlated with $A$; now we recognize what weights were penalized too much and we add back some.



### Feature correlation effect

Now we can focus on $m = \mu_y + S_y R_{yx} \left[ I - \epsilon \pi (I-\epsilon_\pi \pi) \right] z_x $


The zero order approximation ($\epsilon_\pi = 0$) to $m$ is $m_i = \mu_{y_i} + \sigma_i \sum_{j = 1}^q \rho_{ij} z_{x_j}$; to make a prediction we just multiply each standardized feature by it's correlation with the target and sum the values - then it multiplies by the scale of the target to get an appropriate value. 

Now, let 

$t^{\text{first}} = (I-\epsilon_\pi \pi) z_x \rightarrow t^{\text{first}}\_i = z_{x_i} - \epsilon_\pi \sum_{j \neq i} \pi_{ij} z_{x_j}$ 

This operation is the expected value $z_x$ penalized by how the features are correlated: if there are other features that are correlated with this one, then it's _signal_ is reduced. As before, we can set up the regularization $\epsilon_\pi$ such that the final value do not go against the initial expected value ($\epsilon_\pi$ must be verify $\epsilon_\pi < \frac{\|z_{x_i}\|}{\|\sum_{j \neq i} \pi_{ij} z_{x_j}\|}$ for all $i$). This ensures that this is effectivelly a _correction_ term.

Then first order approximation to expected value is

$m_i = \mu_{y_i} + \sigma_i \sum_{j = 1}^q \rho_{ij} t^{\text{first}}_j$

To make a prediction we proceed as before but use a _penalized_ feature value. This is a interesting way to think about it: we go from summing individual regressions to sum penalized individual regressions.


#### Second Order

In a similar way as before, we can write a recursive formula:

$t^{\text{second}} = z_x -\epsilon_\pi \pi t^{\text{first}}$

this yields the expected value:

$m_i = \mu_{y_i} + \sigma_i \sum_{j = 1}^q \rho_{ij} t^{\text{second}}_j$

which is similar to the first order solution but with a different expected value vector. The interpretation is as before: adding back some values that were penalized too much.





## Heuristics

Although one could use the formulas directly it is more interesting to derive heuristics in how one should think:

- When allocating between correlated assets we should reduce the weight on correlated groups.
- To make predictions we should reduce the predictions derived from correlated groups.

For sure all of these recipes are known; correcting or not adding signals for being too correlated and/or reducing positions in correlated assets may seem _ad hoc_ but can be justified.


### Equivalent variables analogy

Lets consider the first order approximations for weight and expected value:

$w \approx S_y^{-1} \left\[I-\epsilon r \right]S_y^{-1} m$

and

$m \approx \mu_y + S_y R_{yx} \left\[ I - \epsilon \pi \right] z_x$

A common feature that they have is a calculation of the type $\left(I - \epsilon h\right) v$. Consider the existence of a group $G$ such that correlations between elements in it are high and equal to $q$ (the expansion in $\epsilon$ is not valid - need more terms - but we are still using it); $r_{ij} \approx q$. For any $i \in G$ we can write the component $i$ of the above vector as

$\left\[\left(I - \epsilon r\right) v \right]_i = v_i - \epsilon \left(q \sum_{j \neq i, j \in G} v_j + \sum_{k \neq i, k \notin G} r_{ik}v_k \right)$

also, the terms with correlation $q$ dominate the others as this correlation is much higher:

$\left\[\left(I - \epsilon r\right) v \right\]\_i \approx v_i - \epsilon \left(q \sum_{j \neq i, j \in G} v_j \right) = v_i (1+q\epsilon) - q\epsilon \sum_{j \in G} v_j$


with small $\epsilon$, $q \rightarrow 1$ and $\bar{v_G} = \frac{1}{\|G\|}\sum_{j \in G} v_j$

$\left\[\left(I - \epsilon r\right) v \right]_i \approx v_i - \epsilon \|G\|\bar{v_G}$

For practical reasons, $\epsilon$ must be proportional to $\frac{1}{n}$ and so the term $\epsilon \|G\|$ is of the order of magnitude to the ratio of number of elements in $G$ to the whole set - call it $\phi$. Finally

$v_i - \epsilon \|G\|\bar{v_G} \approx v_i - \phi \bar{v_G} $

This analysis shows explicitly how the approximation separates two components inside a correlated group:

- the common component, proportional to the group average, corresponding to the dominant eigenvector;
- the deviation components, corresponding to eigenvectors with small eigenvalues.

These deviations correspond to directions with small eigenvalues, which represent small differences between highly correlated variables. These directions are more sensitive to estimation error and are therefore often dominated by noise. This can be seen explicitly from the structure of the correlation matrix. A group of highly correlated variables produces one eigenvector proportional to the vector of ones on the group, representing their common component, and additional eigenvectors representing deviations inside the group. The operation above subtracts the projection onto this common direction and leaves only the deviations. Replacing the group by its average keeps only the common eigenvector and discards the deviation eigenvectors, which correspond to small eigenvalues.

In particular, for the weight case ($s \leftarrow v, r \leftarrow h, i \in G$)

$w_i \approx \frac{1}{\sigma_i }\left( s_i - \phi \bar{s_G} \right)$

which is interpreted a relative value allocation around the mean of the group: here it may make sense to consider an equivalent variable (and discard the relative allocation) or not depending on the application - signals may be designed to trade in spreads.

For the expected value case, ($z_x \leftarrow v, \pi \leftarrow h, i \in G$), it it not clear what a relative signal may mean and most likely is just noise - considering a equivalent variable seems the correct interpretation.

