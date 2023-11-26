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

## Step 1: Initialize an interrupted time series estimator
The InterruptedTimeSeries method takes four agruments: an array of pre-event covariates, a 
vector of pre-event outcomes, an array of post-event covariates, and a vector of post-event 
outcomes.
```julia
# Generate some data to use
X₀, Y₀, X₁, Y₁ =  rand(1000, 5), rand(1000), rand(100, 5), rand(100)

its = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
```

## Step 2: Estimate the Treatment Effect
Estimating the treatment effect only requires one argument: an InterruptedTimeSeries struct.
```julia
# We can also estimate the ATT by passing quantity_of_interest="ATT"
estimate_causal_effect!(its)
```

## Step 3: Get a Summary
We can get a summary of the model, including a p-value and statndard via asymptotic 
randomization inference, by pasing the model to the summarize method.
```julia
summarize(its)
```

## Step 4: Validate the Model
For an interrupted time series design to work well we need to be able to get an unbiased 
prediction of the counterfactual outcomes. If the event or intervention effected the 
covariates we are using to predict the counterfactual outcomes, then we will not be able to 
get unbiased predictions. We can verify this by conducting a Chow Test on the covariates. An
ITS design also assumes that any observed effect is due to the hypothesized intervention, 
rather than any simultaneous interventions, anticipation of the intervention, or any 
intervention that ocurred after the hypothesized intervention. We can use a Wald supremum 
test to see if the hypothesized intervention ocurred where there is the largest structural 
break in the outcome or if there was a larger, statistically significant break in the 
outcome that could confound an ITS analysis. The covariates in an ITS analysis should be 
good predictors of the outcome. If this is the case, then adding irrelevant predictors 
should not have much of a change on the results of the analysis. We can conduct all these 
tests in one line of code.
```julia
validate(its)
```