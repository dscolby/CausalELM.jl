# Double Machine Learning
Double machine learning, also called debiased or orthogonalized machine learning, enables
estimating causal effects when the dimensionality of the covariates is too high for linear 
regression or the treatment or outcomes cannot be easily modeled parametrically. Double 
machine learning estimates models of the treatment assignment and outcome and then combines 
them in a final model. This is a semiparametric model in the sense that the first stage 
models can take on any functional form but the final stage model is a linear combination of 
the residuals from the first stage models.

!!! note
    For more information see:

    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.

## Step 1: Initialize a Model
The DoubleMachineLearning constructor takes at least three arguments—covariates, a 
treatment statuses, and outcomes, all of which may be either an array or any struct that 
implements the Tables.jl interface (e.g. DataFrames). This estimator supports binary, count, 
or continuous treatments and binary, count, continuous, or time to event outcomes.

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
# Create some data with a binary treatment
X, T, Y, W = rand(100, 5), [rand()<0.4 for i in 1:100], rand(100), rand(100, 4)

# We could also use DataFrames or any other package implementing the Tables.jl API
# using DataFrames
# X = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100), x5=rand(100))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:100]), DataFrame(y=rand(100))
dml = DoubleMachineLearning(X, T, Y)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we call estimate_causal_effect! on the model above.
```julia
# we could also estimate the ATT by passing quantity_of_interest="ATT"
estimate_causal_effect!(dml)
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
# Can also use the British spelling
# summarise(dml)
summarize(dml)
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
    Obtaining correct estimates is dependent on meeting the assumptions for double machine 
    learning. If the assumptions are not met then any estimates may be biased and lead to 
    incorrect conclusions.

!!! note
    For a thorough review of casual inference assumptions see:

        Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
        Francis, 2024. 

    For more information on the E-value test see:
  
        VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
        introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.


```julia
validate(g_computer)
```