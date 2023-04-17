# G-Computation
In some cases, we may want to know the causal effect of a treatment that varies and is 
confounded over time. For example, a doctor might want to know the effect of a treatment 
given at multiple times whose status depends on the health of the patient. One way to get an 
unbiased estimate of the causal effect is to use G-computation. The basic steps for using 
G-computation in CausalELM are below.

## Generate Data
```julia
# Create some data with a binary treatment
X, Y, T =  rand(1000, 5), rand(1000), [rand()<0.4 for i in 1:1000]
```

## Step 1: Initialize a Model
The GComputation method takes three arguments: an array of covariates, a vector of 
outcomes, and a vector of treatment statuses.
```julia
m1 = GComputation(X, Y, T)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we pass the model above to estimatecausaleffect!.
```julia
# Note that we could also estimate the ATT by setting quantity_of_interest="ATT"
estimatecausaleffect!(m1)
```

## Step 3: Get a Summary
We get a summary of the model that includes a p-value and standard error estimated via 
asymptotic randomization inference by passing our model to the summarize method.
```julia
summarize(m1)
```