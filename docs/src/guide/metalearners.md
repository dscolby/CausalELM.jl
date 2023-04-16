# Metalearners
Instead of knowing the avereage cuasal effect, we might want to know which units benefit and 
which units lose by being exposed to a treatment. For example, a cahs transfer program might 
motivate some people to work harder and incentivize tohers to work less. Thus, we might want 
to know how the cash transfer program affects individuals isntead of it average affect on 
the population. To do so, we can use metalearners. Depending on the scenario, we may want to 
use an S-learners, a T-learner, or an X-learner. The basic steps to use all theree 
metalearners are below.

## Generate Some data
```julia
X, Y, T =  rand(1000, 5), rand(1000), [rand()<0.4 for i in 1:1000]
```

# Initialize a Metalearner
S-learners, T-learners, and X-learners all take three arguments: an array of covariates, a 
vector of outcomes, and a vector of treatment statuses.
```julia
m1 = SLearner(X, Y, T)
m2 = TLearner(X, Y, T)
m3 = XLearner(X, Y, T)
```

# Estimate the CATE
We can estimate the CATE for all the models by passing them to estimatecausaleffect!.
```julia
estimatecausaleffect!(m1)
estimatecausaleffect!(m2)
estimatecausaleffect!(m3)
```

# Get a Summary
We can get a summary of the models that includes p0values and standard errors for the 
average treatment effect by passing the models to the summarize method.
```julia
summarize(m1)
```