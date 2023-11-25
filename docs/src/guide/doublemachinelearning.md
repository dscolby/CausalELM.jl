# Double Machine Learning
Doubly robust estimation estimates separate models for the treatment and outcome variables 
and weights the outcome estimates by the treatment estimates. This allows one to model more 
complex, nonlinear relationships between the treatment and outcome variables. Additonally, 
double machine learning is doubly robust, which meants that only one of the models has to be 
specified correctly to produce an unbiased estimate of the causal effect. This 
implementation also uses cross fitting to avoid regularization bias. The main steps for 
using doubly robust estimation in CausalELM are below.

For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.

## Generate Data
```julia
# Create some data with a binary treatment
X, Xₚ, Y, T =  rand(100, 5), rand(100, 4), rand(100), [rand()<0.4 for i in 1:100]
```

## # Step 1: Initialize a Model
The DoubleMachineLearning constructor takes four arguments, an array of covariates for the 
outcome model, an array of covariates for the treatment model, a vector of outcomes, and a 
vector of treatment statuses.
```julia
dml = DoubleMachineLearning(X, Xₚ, Y, T)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we call estimatecausaleffect! on the model above.
```julia
# we could also estimate the ATT by passing quantity_of_interest="ATT"
estimatecausaleffect!(dml)
```

# Get a Summary
We can get a summary that includes a p-value and standard error estimated via asymptotic 
randomization inference by passing our model to the summarize method.
```julia
summarize(dml)
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