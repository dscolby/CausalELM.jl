# G-Computation
In some cases, we may want to know the causal effect of a treatment that varies and is 
confounded over time. For example, a doctor might want to know the effect of a treatment 
given at multiple times whose status depends on the health of the patient at a given time. 
One way to get an unbiased estimate of the causal effect is to use G-computation. The basic 
steps for using G-computation in CausalELM are below.

For a good overview of G-Computation see:
    Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
    Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
    Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
    estimator for causal inference with different covariates sets: a comparative simulation 
    study." Scientific reports 10, no. 1 (2020): 9219.

## Generate Data
```julia
# Create some data with a binary treatment
X, Y, T =  rand(1000, 5), rand(1000), [rand()<0.4 for i in 1:1000]
```

## Step 1: Initialize a Model
The GComputation method takes three arguments: an array of covariates, a vector of 
outcomes, and a vector of treatment statuses.
```julia
g_computer = GComputation(X, Y, T)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we pass the model above to estimatecausaleffect!.
```julia
# Note that we could also estimate the ATT by setting quantity_of_interest="ATT"
estimatecausaleffect!(g_computer)
```

## Step 3: Get a Summary
We get a summary of the model that includes a p-value and standard error estimated via 
asymptotic randomization inference by passing our model to the summarize method.
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


For a thorough review of casual inference assumptions see:
    Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
    Francis, 2024. 

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.
```julia
validate(g_computer)
```