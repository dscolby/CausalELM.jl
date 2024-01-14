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
treatment statuses, and an outcome vector. 

You can also specify the causal estimand, whether to employ L2 regularization, which 
activation function to use, whether the data is of a temporal nature, the metric to use when 
using cross validation to find the best number of neurons, the minimum number of neurons to 
consider, the maximum number of neurons to consider, the number of folds to use during cross 
caidation, and the number of neurons to use in the ELM that learns a mapping from number of 
neurons to validation loss. These are options are specified with the following keyword 
arguments: quantity_of_interest, regularized, activation, temporal, validation_metric, 
min_neurons, max_neurons, folds, iterations, and approximator_neurons.
```julia
# Create some data with a binary treatment
X, T, Y =  rand(1000, 5), [rand()<0.4 for i in 1:1000], rand(1000)

# We could also use DataFrames
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

Calling the summarize methodd returns a dictionary with the estimator's task (regression or 
classification), the quantity of interest being estimated (ATE or ATT), whether the model 
uses an L2 penalty, the activation function used in the model's outcome predictors, whether 
the data is temporal, the validation metric used for cross validation to find the best 
number of neurons, the number of neurons used in the ELMs used by the estimator, the number 
of neurons used in the ELM used to learn a mapping from number of neurons to validation 
loss during cross validation, the causal effect, standard error, and p-value.
```julia
summarize(g_computer)
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

One can also specify the maxium number of possible treatments to consider for the causal 
consistency assumption and the minimum and maximu probabilities of treatment for the 
positivity assumption with the num_treatments, min, and max keyword arguments.


For a thorough review of casual inference assumptions see:
    Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
    Francis, 2024. 

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.
```julia
validate(g_computer)
```