# G-Computation
In some cases, we may want to know the causal effect of a treatment that varies and is 
confounded over time. For example, a doctor might want to know the effect of a treatment 
given at multiple times whose status depends on the health of the patient at a given time. 
One way to get an unbiased estimate of the causal effect is to use G-computation. The basic 
steps for using G-computation in CausalELM are below.

!!! note 
    For a good overview of G-Computation see:
    
        Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
        Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
        Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
        estimator for causal inference with different covariates sets: a comparative simulation 
        study." Scientific reports 10, no. 1 (2020): 9219.

## Step 1: Initialize a Model
The GComputation method takes at least three arguments: an array of covariates, a vector of 
treatment statuses, and an outcome vector. It can support binary treatments and binary, 
continuous, time to event, and count outcome variables.

!!! tip
    You can also specify the causal estimand, which activation function to use, whether the 
    data is of a temporal nature, the number of extreme learning machines to use, the 
    number of features to consider for each extreme learning machine, the number of 
    bootstrapped observations to include in each extreme learning machine, and the number of 
    neurons to use during estimation. These options are specified with the following keyword 
    arguments: quantity\_of\_interest, activation, temporal, num_machines, num_feats, 
    sample_size, and num\_neurons.

!!! note
    Internally, the outcome model is treated as a regression since extreme learning machines 
    minimize the MSE. This means that predicted outcomes under treatment and control groups 
    could fall outside [0, 1], although this is not likely in practice. To deal with this, 
    predicted binary variables are automatically clipped to [0.0000001, 0.9999999]. This also 
    means that count outcomes will be predicted as continuous variables.

```julia
# Create some data with a binary treatment
X, T, Y =  rand(1000, 5), [rand()<0.4 for i in 1:1000], rand(1000)

# We could also use DataFrames or any other package that implements the Tables.jl API
# using DataFrames
# X = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:1000]), DataFrame(y=rand(1000))

g_computer = GComputation(X, T, Y)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we pass the model above to estimatecausaleffect!.
```julia
# Note that we could also estimate the ATT by setting quantity_of_interest="ATT"
estimate_causal_effect!(g_computer)
```

## Step 3: Get a Summary
We get a summary of the model that includes a p-value and standard error estimated via 
asymptotic randomization inference by passing our model to the summarize method.

Calling the summarize method returns a dictionary with the estimator's task (regression or 
classification), the quantity of interest being estimated (ATE), whether the model uses an 
L2 penalty (always true for DML), the activation function used in the model's outcome 
predictors, whether the data is temporal, the number of neurons used in the ELMs used by the 
estimator, the causal effect, standard error, and p-value. Due to long running times, 
calculation of the p-value and standard error is not conducted and set to NaN unless 
inference is set to true.
```julia
summarize(g_computer)
```

## Step 4: Validate the Model
We can validate the model by examining the plausibility that the main assumptions of causal 
inference, counterfactual consistency, exchangeability, and positivity, hold. It should be 
noted that consistency and exchangeability are not directly testable, so instead, these 
tests do not provide definitive evidence of a violation of these assumptions. To probe the 
counterfactual consistency assumption, we simulate counterfactual outcomes that are 
different from the observed outcomes, estimate models with the simulated counterfactual 
outcomes, and take the averages. If the outcome is continuous, the noise for the simulated 
counterfactuals is drawn from N(0, dev) for each element in devs, otherwise the default is 
0.25, 0.5, 0.75, and 1.0 standard deviations from the mean outcome. For discrete variables, 
each outcome is replaced with a different value in the range of outcomes with probability ϵ 
for each ϵ in devs, otherwise the default is 0.025, 0.05, 0.075, 0.1. If the average 
estimate for a given level of violation differs greatly from the effect estimated on the 
actual data, then the model is very sensitive to violations of the counterfactual 
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
    Obtaining correct estimates is dependent on meeting the assumptions for G-computation. 
    If the assumptions are not met then any estimates may be biased and lead to incorrect 
    conclusions.

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