# Interrupted Time Series Analysis
Sometimes we want to know how an outcome variable for a single unit changed after an event 
or intervention. For example, if regulators announce sanctions against company A, we might 
want to know how the price of stock A changed after the announcement. Since we do not know
what the price of Company A's stock would have been if the santions were not announced, we
need some way to predict those values. An interrupted time series analysis does this by 
using some covariates that are related to the oucome variable but not related to whether the 
event happened to predict what would have happened. The estimated effects are the 
differences between the predicted post-event counterfactual outcomes and the observed 
post-event outcomes, which can also be aggregated to mean or cumulative effects. 
Estimating an interrupted time series design in CausalELM consists of three steps.

For a deeper dive see:
    Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
    regression for the evaluation of public health interventions: a tutorial." International 
    journal of epidemiology 46, no. 1 (2017): 348-355.

## Generate Data
```julia
X₀, Y₀, X₁, Y₁ =  rand(1000, 5), rand(1000), rand(100, 5), rand(100)
```

## Step 1: Initialize an interrupted time series estimator
The InterruptedTimeSeries method takes four agruments: an array of pre-event covariates, a 
vector of pre-event outcomes, an array of post-event covariates, and a vector of post-event 
outcomes.
```julia
m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
```

## Step 2: Estimate the Treatment Effect
Estimating the treatment effect only requires one argument: an InterruptedTimeSeries struct.
```julia
# We can also estimate the ATT by passing quantity_of_interest="ATT"
estimatecausaleffect!(m1)
```

## Step 3: Get a Summary
We can get a summary of the model, including a p-value and statndard via asymptotic 
randomization inference, by pasing the model to the summarize method.
```julia
summarize(m1)
```