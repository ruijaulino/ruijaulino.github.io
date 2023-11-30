
# Small remark on backtesting

The most important aspect in any systematic trading model is the quantification of future performance - or, in other words, out-of-sample performance.

In general, a model consists in the definition of the distribution of the target variable given some feature variables (that can include past observations of the target), $p(y\|x)$; this distribution can be specified under some parameters $\theta$ that need to be _fitted_(or _learned_) from data.

A common way to make this evaluation is to run a walk-forward simulation; we follow the data in time and, as new data comes, we retrain the model; then we evaluate the model in the next data points. This is simple to interpret as it resembles the actual trading process. This is illustrated in the next figure.


    
![png](/images/cv_remarks/output_2_0.png)
    


Another variation of this is to ditch _old_ data for training; this looks like


    
![png](/images/cv_remarks/output_4_0.png)
    


There are some limitations on this type of procedure; one only gets a single historical path, it depends on the order of the events that took place (unless we made that assumption explicitly, there is not reason to assume that the same data distribution will precede some events) and it uses the data in a ineficient manner.


A less consensual way of estimating out-of-sample performance is to do it in a reverse order (of course provided that there is no _superposition_ between the training and test sets; this can happen if the features are generated with past data - in this situation it is useful to leave some data out between the training and testing sets) or using from period before and after the testing set (more similar to cross-validation). Please note that I am not talking about testing on the reversed return sequence; what is meant is to train on return sequence that was generated after the test return sequence happened.





    
![png](/images/cv_remarks/output_7_0.png)
    



    
![png](/images/cv_remarks/output_7_1.png)
    



    
![png](/images/cv_remarks/output_7_2.png)
    


Using this method is not consentual because people may argue that _future_ information is being used. Provided that we eliminated any superposition between training and testing and that we are fitting a model, is it correct to discard this method? 

For example, consider the dataset divided into three disjoint sets $S_1,S_2,S_3$. The regular walk forward would estimate with:



    
![png](/images/cv_remarks/output_9_0.png)
    


Now, let us say that the data in set $S_i$ has distribution $\Omega_i$. Since we assume that the previous construction is valid (hey, it is on the correct time order), then we are saying/assuming that $\Omega_1=\Omega_2$ and $\Omega_2=\Omega_3$ (if this is not the case, it does not make sense to estimate the model on a distribution and not assume that the test data has the same one). By consequence, $\Omega_1=\Omega_3$; this means that, in principle, there is no great justification to ditch _use data from set_ $S_3$ _to evaluate the model in_ $S_1$ (and to use $S_2$ to evaluate in $S_1$). Again, what we did here was to assume that the walk forward procedure is correct, then, since the sets overlap (the next train is the previous test) imply that the distribution has to be the same throughout the whole dataset. 

In the end this has to be valid because we are modelling the relation between the features and the targets (probably financial returns) with some model and we need that model to be true all the time (of course, the model can include regime/parameter changes; does not need to be a fixed relation). The worst that can happen is that the market sequence _arbed_ out something that could have been present in the past data and so, if this is the case, we can use this _future_ data to learn the model because it will have zero relation with the past - meaning that there is no positive bias in doing this. 


When using just the most recent data to train the model, many times the argument is that the parameters change in time; first, it should be clear that this change must be _slow_ (this means that the estimated parameters are valid in a near future) and, second, this does not invalidate the previous reasoning because we can always subordinate that case by devising a model with a difusion on the parameters and the time scale of parameter change can be infered from training data (from the _future_ or not) and so, again, this is no justification to ditch using data from the _future_.

A more correct case, would be to consider a setup like the next figure, where we leave some data out between training as testing sets to make sure that there are not correlations between them (this yes can induce a bias).




    
![png](/images/cv_remarks/output_11_0.png)
    


As a final remark, consider the data divided into three disjoint sets $S_1,S_2,S_3$ and, also, consider each set divided into a train and a test subset.


![png](/images/cv_remarks/output_13_0.png)
    


If we assume that all distributions $\Omega_i$ are different and different from the distribution of future data, then we cannot conclude nothing about future performance of the algorithm (no free-lunch). We always need to make assumptions to fit models; if we consider that a walk-forward is correct then there is no reason to assume that a procedure like cross-validation (that can use data generated in time _after_ the testing observations) is not valid and/or will induce a positive bias in the results (provided that we ensure the train and test are properly uncorrelated).
