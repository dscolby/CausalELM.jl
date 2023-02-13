# Examples
Below are some small examples for estimating causal quantities of interest with CausalELM.
Regardless of the estimator, the workflow is the same; get some data, initialize an 
estimator, estimate the causal effect of interest, and get a summary of the model.

# Event Study Estimation
```julia
# Generate some data
X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)

# Initialize an event study estimator
m1 = EventStudy(X₀, Y₀, X₁, Y₁)

# Estimate the average treatment effect
# We can also estimate the ATT of ITE
estimatecausaleffect!(m1)

# Get a summary
summarize(m1)
```

# G-Computation
```julia
# Create some data with a binary treatment
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

# Initialize a model
m1 = GComputation(X, Y, T)

# Estimate the ATE
# Note that we could also estimate the ATT or ITE
estimatecausaleffect!(m1)

# Get a summary
summarize(m1)
```

# Doubly Robust Estimation
```julia
# Create some data with a binary treatment
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

# Initialize a model
m1 = DoublyRobust(X, Y, T)

# Estimate the ATE
# Note that we could also estimate the ATT or ITE
estimatecausaleffect!(m1)

# Get a summary
summarize(m1)
```

# S-Learning
```julia
# Generate data
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

# Initialize an S-Learner
m1 = SLearner(X, Y, T)

# Estimate the CATE
estimatecausaleffect!(m1)

# Get a summary
summarize(m1)
```

# T-Learning
```julia
# Generate data
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

# Initialize an S-Learner
m1 = TLearner(X, Y, T)

# Estimate the CATE
estimatecausaleffect!(m1)

# Get a summary
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

# Get a summary
summarize(m1)
```
