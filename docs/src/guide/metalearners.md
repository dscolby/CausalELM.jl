# Metalearners
Instead of knowing the average causal effect, we might want to know which units benefit and 
which units lose by being exposed to a treatment. For example, a cash transfer program might 
motivate some people to work harder and incentivize others to work less. Thus, we might want 
to know how the cash transfer program affects individuals instead of it average affect on 
the population. To do so, we can use metalearners. Depending on the scenario, we may want to 
use an S-learner, T-learner, X-learner, R-learner, or doubly robust learner. The basic steps 
to use all five metalearners are below. The difference between the metalearners is how they 
estimate the CATE and what types of variables they can handle. In the case of S, T, X, and 
doubly robust learners, they can only handle binary treatments. On the other hand, 
R-learners can handle binary, categorical, count, or continuous treatments but only supports 
continuous outcomes.

!!! note
    For a deeper dive on S-learning, T-learning, and X-learning see:
    
        Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
        estimating heterogeneous treatment effects using machine learning." Proceedings of the 
        national academy of sciences 116, no. 10 (2019): 4156-4165.


    To learn more about R-learning see:
    
        Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
        effects." Biometrika 108, no. 2 (2021): 299-319.


    To see the details out doubly robust estimation implemented in CausalELM see:

        Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
        effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

# Initialize a Metalearner
S-learners, T-learners, X-learners, R-learners, and doubly robust estimators all take at 
least three arguments—covariates, treatment statuses, and outcomes, all of which can be 
either an array or any struct that implements the Tables.jl interface (e.g. DataFrames). S, 
T, X, and doubly robust learners support binary treatment variables and binary, continuous, 
count, or time to event outcomes. The R-learning estimator supports binary, continuous, or 
count treatment variables and binary, continuous, count, or time to event outcomes.

!!! note
    Non-binary categorical outcomes are treated as continuous.

!!! tip
    You can also specify the the number of folds to use for cross-fitting, the number of 
    extreme learning machines to incorporate in the ensemble, the number of features to 
    consider for each extreme learning machine, the activation function to use, the number 
    of observations to bootstrap in each extreme learning machine, and the number of neurons 
    in each extreme learning machine. These arguments are specified with the folds, 
    num\_machines, num\_features, activation, sample\_size, and num\_neurons keywords.

```julia
# Generate data to use
X, Y, T =  rand(1000, 5), rand(1000), [rand()<0.4 for i in 1:1000]

# We could also use DataFrames or any other package that implements the Tables.jl API
# using DataFrames
# X = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:1000]), DataFrame(y=rand(1000))
s_learner = SLearner(X, Y, T)
t_learner = TLearner(X, Y, T)
x_learner = XLearner(X, Y, T)
r_learner = RLearner(X, Y, T)
dr_learner = DoublyRobustLearner(X, T, Y)
```

# Estimate the CATE
We can estimate the CATE for all the models by passing them to estimate_causal_effect!.
```julia
estimate_causal_effect!(s_learner)
estimate_causal_effect!(t_learner)
estimate_causal_effect!(x_learner)
estimate_causal_effect!(r_learner)
estimate_causal_effect!(dr_lwarner)
```

# Get a Summary
We can get a summary of the model by pasing the model to the summarize method.

!!!note
    To calculate the p-value, standard error, and confidence interval for the treatment 
    effect, you can set the inference keyword to false. However, these values are calculated 
    via randomization inference, which will take a long time. This can be greatly sped up by 
    launching Julia with more threads and lowering the number of iterations using the n 
    keyword (at the expense of accuracy).

```julia
summarize(s_learner)
summarize(t_learner)
summarize(x_learner)
summarize(r_learner)
summarize(dr_learner)
```

## Step 4: Validate the Model
We can validate the model by examining the plausibility that the main assumptions of causal 
inference, counterfactual consistency, exchangeability, and positivity, hold. It should be 
noted that consistency and exchangeability are not directly testable, so instead, these 
tests do not provide definitive evidence of a violation of these assumptions. To probe the 
counterfactual consistency assumption, we simulate counterfactual outcomes that are 
different from the observed outcomes, estimate models with the simulated counterfactual 
outcomes, and take the averages. If the outcome is continuous, the noise for the simulated 
counterfactuals is drawn from N(0, dev) for each element in devs and each outcome, 
multiplied by the original outcome, and added to the original outcome. For discrete 
variables, each outcome is replaced with a different value in the range of outcomes with 
probability ϵ for each ϵ in devs, otherwise the default is 0.025, 0.05, 0.075, 0.1. If the 
average estimate for a given level of violation differs greatly from the effect estimated on 
the actual data, then the model is very sensitive to violations of the counterfactual 
consistency assumption for that level of violation. Next, this method tests the model's 
sensitivity to a violation of the exchangeability assumption by calculating the E-value, 
which is the minimum strength of association, on the risk ratio scale, that an unobserved 
confounder would need to have with the treatment and outcome variable to fully explain away 
the estimated effect. Thus, higher E-values imply the model is more robust to a violation of 
the exchangeability assumption. Finally, this method tests the positivity assumption by 
estimating propensity scores. Rows in the matrix are levels of covariates that have a zero 
or near zero probability of treatment. If the matrix is empty, none of the observations have 
an estimated zero probability of treatment, which implies the positivity assumption is 
satisfied.

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
validate(dr_learner)
```