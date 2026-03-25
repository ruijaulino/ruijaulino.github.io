# Intuitions from a Gaussian model

Consider returns $y \in \mathbb{R}^p$ and features $x \in \mathbb{R}^q$ to be joint Gaussian:

$$p(y, x) = N(\mu, C)$$

Conditioning $y$ on $x$ yields a linear regression. In a financial context, we generally assume $R_{xy}$ (cross-correlation) is small — low signal to noise — while $R_{yy}$ and $R_{xx}$ (internal correlations) may be large due to market beta and factor exposures.


## Performance is sum of squared canonical correlations

If we define the optimal weight vector for a strategy and assume small feature/target correlations with zero mean, we get:

$$w = C_{y\|x}^{-1} \mu_{y\|x} \approx C_{yy}^{-1} C_{yx} C_{xx}^{-1} x$$

This yields a strategy $s = w^T y$. The quantity $g = \mathbf{E}[s] = \mathbf{E}[s^2]$ is

$$g = \int \int w^T y p(x, y) \text{d}y \text{d}x = \int w^T \mu_{y\|x} p(x) \text{d}x = \mathbf{E}_x \left[ x^T C_{xx}^{-1} C_{xy} C_{yy}^{-1} C_{yx} C_{xx}^{-1} x \right] = \text{tr} \left[ C_{yy}^{-1} C_{yx} C_{xx}^{-1} C_{xy}\right]$$

Let $M = C_{yy}^{-1} C_{yx} C_{xx}^{-1} C_{xy}$. The trace $\text{tr}(M) = \sum_i \lambda_i$ represents the sum of the eigenvalues of $M$. We can associate $\lambda_i$ as the square of a canonical correlation: the correlation between the variables $a^Ty$ and $b^Tx$ where $a$ and $b$ are unit norm vectors such that $\rho(a^Ty, b^Tx)$ is maximized.This result generalizes the one-dimensional case (one feature, one target) where $g = \rho(x, y)^2$. It suggests that performance is essentially the cumulative "information bandwidth" captured across all independent dimensions of the feature-target relationship.





## Averaging models is doing regularization

We often face a choice: do we build one large model that accounts for every asset and feature, or an ensemble of many small, independent strategies? How both precedures compare?

Consider that we build strategies on a subset of targets $y_k$ using a subset of features $x_k$. As we add new strategies over time, we average them. 

If $S$ is a selection matrix for $y$ and $P$ for $x$, then the weight vector is

$$w_k = S^T \left( S C_{yy} S^T \right)^{-1} S C_{yx} P^T \left( P C_{xx} P^T \right)^{-1} P x$$

this $w_k$ is just the optimal weight vector that we get when we only consider the distribution of a subset of $y$ conditioned on a subset of $x$ with zeros padded on the other excluded variables to keep the same original dimensions - this is a good prototype of what one would consider a strategy.

Now we can look at what happens when we average many _strategies_:

$$\bar{w} = \frac{1}{n} \sum_k w_k$$

if the selection of subsets on $y$ and $x$ is independent then

$$\bar{w} \rightarrow \mathbf{E}\left[ S^T \left( S C_{yy} S^T \right)^{-1} S \right] C_{yx} \mathbf{E}\left[ P^T \left( P C_{xx} P^T \right)^{-1} P \right] x$$

### Heuristics

How does $\mathbf{E}\left[ L^T \left( L C L^T \right)^{-1} L \right]$ behave? Let us start to show that this can be associated with regularization of the covariance. For a single selection $L$, rearange the elements in $C$ such that

$$C = \begin{bmatrix} C_{II} & C_{IO} \\\\ C_{OI} & C_{OO} \end{bmatrix}$$

$L$ picks out $C_{II}$. The expression $L^T(LCL^T)^{-1}L$ results in:

$$M_L = \begin{bmatrix} C_{II}^{-1} & 0 \\\\ 0 & 0 \end{bmatrix}$$

Now, compare this to the full inverse of $C$ using the Schur complement $S_{OO} = C_{OO} - C_{OI}C_{II}^{-1}C_{IO}$:

$$C^{-1} = \begin{bmatrix} C_{II}^{-1} + C_{II}^{-1}C_{IO}S_{OO}^{-1}C_{OI}C_{II}^{-1} & -C_{II}^{-1}C_{IO}S_{OO}^{-1} \\\\ -S_{OO}^{-1}C_{OI}C_{II}^{-1} & S_{OO}^{-1} \end{bmatrix}$$

Notice the difference

- Full Model ($C^{-1}$): Uses the cross-correlations ($C_{IO}$) to adjust the weights.

- Subset Model ($M_L$): ignores $C_{IO}$. We are forcing off-diagonal interactions to zero.

When we average many subsets we can see that cross correlations count less for the inverse, a process similar to regularization. To make this a bit more formal, consider the matrix:

$$\Sigma = \text{lim}_{\tau \rightarrow \infty} \begin{bmatrix} 0 & 0 \\\\ 0 & \tau I \end{bmatrix}$$

Then, from the formulas to a block matrix inverse we can write

$$\left(C+\Sigma \right)^{-1} = \begin{bmatrix} C_{II}^{-1} & 0 \\\\ 0 & 0 \end{bmatrix}$$

and, consequentially

$$L^T \left( L C L^T \right)^{-1} L = \text{lim}_{\tau \rightarrow \infty} \left(C+\Sigma\right)^{-1}$$

Taking the expected value

$$\mathbf{E}\left[ L^T \left( L C L^T \right)^{-1} L \right] = \mathbf{E}\left[ \left(C+\Sigma\right)^{-1} \right] > \left(C + \mathbf{E}\left[\Sigma \right] \right)^{-1}$$

If the subsets we choose have size $k<d$, by the definition of $\Sigma$, $\mathbf{E}\left[\Sigma \right] = \tau \frac{k}{d}I$. A interesting heuristic is

$$\left(C + \mathbf{E}\left[\Sigma \right] \right)^{-1} = \left(C + \lambda I \right)^{-1}$$

which can be seen as a ridge regularization of the covariance. Recall that this is a heuristic to understand the effect of averaging the inverse of many random subsets of the covariance; from the inequality we can see that the covariance that is generated is _less_ regularized than the ridge version and, as $\tau \rightarrow \infty$ we cannot use the proxy anymore. 


Finally, the average weight becomes:

$$\bar{w} \rightarrow \left(C_{yy} + \lambda_y I\right)^{-1} C_{yx} \left(C_{xx} + \lambda_x I\right)^{-1} x $$

which is similar to estimate a regularized large model. In practice, if the model grows we have to regularize it anyway as data becomes insufficient and so, it is likely that both procedures will produce similar results.

##### Appendix: Selection matrix

To give an example, let $L$ be a selection matrix and $z$ be a $4 \times 1$ vector:

$$z = \begin{bmatrix} z_1 \\\\ z_2 \\\\ z_3 \\\\ z_4 \end{bmatrix}$$

Suppose we want to create a new vector that contains only the first and third elements of $z$. A selection matrix $L$ would be:

$$L = \begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 0 & 1 & 0 \end{bmatrix}$$

When you multiply $L$ by $z$:

$$Lz = \begin{bmatrix} z_1 \\\\ z_3 \end{bmatrix}$$



