# Interrupted Time Series Analysis
Sometimes we want to know how an outcome variable for a single unit changed after an event 
or intervention. For example, if regulators announce sanctions against company A, we might 
want to know how the price of company A's stock changed after the announcement. Since we do 
not know what the price of Company A's stock would have been if the santions were not 
announced, we need some way to predict those values. An interrupted time series analysis 
does this by using some covariates that are related to the outcome but not related to 
whether the event happened to predict what would have happened. The estimated effects are 
the differences between the predicted post-event counterfactual outcomes and the observed 
post-event outcomes, which can also be aggregated to mean or cumulative effects. 
Estimating an interrupted time series design in CausalELM consists of three steps.

!!! note
    For a general overview of interrupted time series estimation see:
    
        Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
        regression for the evaluation of public health interventions: a tutorial." International 
        journal of epidemiology 46, no. 1 (2017): 348-355.

!!! note
    The flavor of interrupted time series implemented here is similar to the variant proposed 
    in:

        Brodersen, Kay H., Fabian Gallusser, Jim Koehler, Nicolas Remy, and Steven L. Scott. 
        "Inferring causal impact using Bayesian structural time-series models." (2015): 247-274.

    in that, although it is not Bayesian, it uses a nonparametric model of the pre-treatment 
    period and uses that model to forecast the counterfactual in the post-treatment period, as 
    opposed to the commonly used segment linear regression.

## Step 1: Initialize an interrupted time series estimator
The InterruptedTimeSeries constructor takes at least four agruments: pre-event covariates, 
pre-event outcomes, post-event covariates, and post-event outcomes, all of which can be 
either an array or any data structure that implements the Tables.jl interface (e.g. 
DataFrames). The interrupted time series estimator assumes outcomes are either continuous, 
count, or time to event variables.

!!! note
    Non-binary categorical outcomes are treated as continuous.

!!! tip
    You can also specify which activation function to use, the number of extreme learning 
    machines to use, the number of features to consider for each extreme learning machine, 
    the number of bootstrapped observations to include in each extreme learning machine, and 
    the number of neurons to use during estimation. These options are specified with the 
    following keyword arguments: activation, num_machines, num_feats, sample_size, and 
    num\_neurons.

```julia
# Generate some data to use
X₀, Y₀, X₁, Y₁ =  rand(1000, 5), rand(1000), rand(100, 5), rand(100)

# We could also use DataFrames or any other package that implements the Tables.jl interface
# using DataFrames
# X₀ = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# X₁ = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# Y₀, Y₁ = DataFrame(y=rand(1000)), DataFrame(y=rand(1000))
its = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
```

## Step 2: Estimate the Treatment Effect
Estimating the treatment effect only requires one argument: an InterruptedTimeSeries struct.

```julia
estimate_causal_effect!(its)
```

## Step 3: Get a Summary
We can get a summary of the model by pasing the model to the summarize method.

!!!note
    To calculate the p-value and standard error for the treatmetn effect, you can set the 
    inference argument to false. However, p-values and standard errors are calculated via 
    randomization inference, which will take a long time. But can be sped up by launching 
    Julia with a higher number of threads.

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

!!! tip
    One can also specify the number of simulated confounders to generate to test the sensitivity 
    of the model to confounding and the minimum and maximum proportion of data to use in the 
    Wald supremum test by including the n, low, and high keyword arguments.

!!! danger
    Obtaining correct estimates is dependent on meeting the assumptions for interrupted time 
    series estimation. If the assumptions are not met then any estimates may be biased and 
    lead to incorrect conclusions.

!!! note
    For a review of interrupted time series identifying assumptions and robustness checks, see:

        Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
        interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

```julia
validate(its)
```