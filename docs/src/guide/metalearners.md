# Metalearners
Instead of knowing the average cuasal effect, we might want to know which units benefit and 
which units lose by being exposed to a treatment. For example, a cash transfer program might 
motivate some people to work harder and incentivize others to work less. Thus, we might want 
to know how the cash transfer program affects individuals instead of it average affect on 
the population. To do so, we can use metalearners. Depending on the scenario, we may want to 
use an S-learner, a T-learner, an X-learner, or an R-learner. The basic steps to use all 
three metalearners are below. The difference between the metalearners is how they estimate 
the CATE and what types of variables they can handle. In the case of S, T, and X learners, 
they can only handle binary treatments. On the other hand, R-learners can handle binary, 
categorical, count, or continuous treatments but only supports continuous outcomes.

!!! note
    For a deeper dive on S-learning, T-learning, and X-learning see:
    
        Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
        estimating heterogeneous treatment effects using machine learning." Proceedings of the 
        national academy of sciences 116, no. 10 (2019): 4156-4165.


    To learn more about R-learning see:
    
        Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
        effects." Biometrika 108, no. 2 (2021): 299-319.

!!! note
    When estimating the CATE on discrete outcomes with an R-learner, it is theoretically 
    possible for the estimated outcomes to fall outside the range of actual values in the 
    dataset, similar to estimating a lienar probability model.

# Initialize a Metalearner
S-learners, T-learners, and X-learners all take at least three arguments: an array of 
covariates, a vector of outcomes, and a vector of treatment statuses. 

!!! tip
    Additional options can be specified for each type of metalearner using its keyword arguments.
```julia
# Generate data to use
X, Y, T =  rand(1000, 5), rand(1000), [rand()<0.4 for i in 1:1000]

# We could also use DataFrames
# using DataFrames
# X = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:1000]), DataFrame(y=rand(1000))

s_learner = SLearner(X, Y, T)
t_learner = TLearner(X, Y, T)
x_learner = XLearner(X, Y, T)
r_learner = RLearner(X, Y, T)
```

# Estimate the CATE
We can estimate the CATE for all the models by passing them to estimate_causal_effect!.
```julia
estimate_causal_effect!(s_learner)
estimate_causal_effect!(t_learner)
estimate_causal_effect!(x_learner)
estimate_causal_effect!(r_learner)
```

# Get a Summary
We can get a summary of the models that includes p0values and standard errors for the 
average treatment effect by passing the models to the summarize method.

Calling the summarize methodd returns a dictionary with the estimator's task (regression or 
classification), the quantity of interest being estimated (CATE), whether the model 
uses an L2 penalty, the activation function used in the model's outcome predictors, whether 
the data is temporal, the validation metric used for cross validation to find the best 
number of neurons, the number of neurons used in the ELMs used by the estimator, the number 
of neurons used in the ELM used to learn a mapping from number of neurons to validation 
loss during cross validation, the causal effect, standard error, and p-value for the ATE.
```julia
summarize(s_learner)
summarize(t_learner)
summarize(x_learner)
summarize(r_learner)
```

## Step 4: Validate the Model
We can validate the model by examining the plausibility that the main assumptions of causal 
inference, counterfactual consistency, exchangeability, and positivity, hold. It should be 
noted that consistency and exchangeability are not directly testable, so instead, these 
tests do not provide definitive evidence of a violation of these assumptions. To probe the 
counterfactual consistency assumption, we assume there were multiple levels of treatments 
and find them by binning the dependent vairable for treated observations using Jenks breaks. 
The optimal number of breaks between 2 and num_treatments is found using the elbow method. 
Using these hypothesized treatment assignemnts, this method compares the MSE of linear 
regressions using the observed and hypothesized treatments. If the counterfactual 
consistency assumption holds then the difference between the MSE with hypothesized 
treatments and the observed treatments should be positive because the hypothesized 
treatments should not provide useful information. If it is negative, that indicates there 
was more useful information provided by the hypothesized treatments than the observed 
treatments or that there is an unobserved confounder. Next, this methods tests the model's 
sensitivity to a violation of the exchangeability assumption by calculating the E-value, 
which is the minimum strength of association, on the risk ratio scale, that an unobserved 
confounder would need to have with the treatment and outcome variable to fully explain away 
the estimated effect. Thus, higher E-values imply the model is more robust to a violation of 
the exchangeability assumption. Finally, this method tests the positivity assumption by 
estimating propensity scores. Rows in the matrix are levels of covariates that have a zero 
probability of treatment. If the matrix is empty, none of the observations have an estimated 
zero probability of treatment, which implies the positivity assumption is satisfied.

!!! tip
    One can also specify the maxium number of possible treatments to consider for the causal 
    consistency assumption and the minimum and maximum probabilities of treatment for the 
    positivity assumption with the num\_treatments, min, and max keyword arguments.

!!! danger
    Obtaining correct estimates is dependent on meeting the assumptions for interrupted time 
    series estimation. If the assumptions are not met then any estimates may be biased and 
    lead to incorrect conclusions.

!!! note
    For a thorough review of casual inference assumptions see:

        Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
        Francis, 2024. 

    For more information on the E-value test see:

        VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
        introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

```julia
validate(s_learner)
validate(t_learner)
validate(x_learner)
validate(r_learner)
```