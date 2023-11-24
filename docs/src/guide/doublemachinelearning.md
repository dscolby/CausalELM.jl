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
m1 = DoubleMachineLearning(X, Xₚ, Y, T)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we call estimatecausaleffect! on the model above.
```julia
# we could also estimate the ATT by passing quantity_of_interest="ATT"
estimatecausaleffect!(m1)
```

# Get a Summary
We can get a summary that includes a p-value and standard error estimated via asymptotic 
randomization inference by passing our model to the summarize method.
```julia
summarize(m1)
```