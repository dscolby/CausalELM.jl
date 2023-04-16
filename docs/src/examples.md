# Examples
Below are some small examples for estimating causal quantities of interest with CausalELM.
Regardless of the estimator, the workflow is the same; get some data, initialize an 
estimator, estimate the causal effect of interest, and get a summary of the model.



# T-Learning
```julia
# Generate data
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

# Initialize an S-Learner
m1 = TLearner(X, Y, T)

# Estimate the CATE
estimatecausaleffect!(m1)

# Get a summary that includes a p-value and standard error via randomization inference
summarize(m1)
```

# X-Learning
```julia
# Generate data
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

# Initialize an S-Learner
m1 = XLearner(X, Y, T)

# Estimate the CATE
estimatecausaleffect!(m1)

# Get a summary that includes a p-value and standard error via randomization inference
summarize(m1)
```
