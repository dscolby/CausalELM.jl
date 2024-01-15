"""Abstract type for metalearners"""
abstract type Metalearner end

"""S-Learner for CATE estimation."""
mutable struct SLearner <: Metalearner
    g::GComputation
    """The effect of exposure or treatment"""
    causal_effect::Array{Float64}

"""
    SLearner(X, T, Y; <keyword arguments>)

Initialize a S-Learner.

For an overview of S-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
- `task::String`: either regression or classification.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of folds to use for cross validation.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.
...

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = SLearner(X, T, Y)
julia> m2 = SLearner(X, T, Y; task="regression")
julia> m3 = SLearner(X, T, Y; task="regression", regularized=true)
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = SLearner(x_df, t_df, y_df)
julia> estimate_causal_effect!(m4)
```
"""
    function SLearner(X, T, Y; task="regression", regularized=true, activation=relu, 
        validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
        iterations=round(size(X, 1)/10), approximator_neurons=round(size(X, 1)/10))

        new(GComputation(X, T, Y, task=task, quantity_of_interest="ATE", 
            regularized=regularized, activation=activation, temporal=false, 
            validation_metric=validation_metric, min_neurons=min_neurons, 
            max_neurons=max_neurons, folds=folds, iterations=iterations, 
            approximator_neurons=approximator_neurons))
    end
end

"""T-Learner for CATE estimation."""
mutable struct TLearner <: Metalearner
    """Covariates"""
    X::Array{Float64}
    """Treatment statuses"""
    T::Array{Float64}
    """Outome variable"""
    Y::Array{Float64}
    """Either regression or classification"""
    task::String
    """Whether to use L2 regularization"""
    regularized::Bool
    """Activation function to apply to the outputs from each neuron"""
    activation::Function
    """Validation metric to use when tuning the number of neurons"""
    validation_metric::Function
    """Minimum number of neurons to test in the hidden layer"""
    min_neurons::Int64
    """Maximum number of neurons to test in the hidden layer"""
    max_neurons::Int64
    """Number of cross validation folds"""
    folds::Int64
    """Number of iterations to perform cross validation"""
    iterations::Int64
    """Number of neurons in the hidden layer of the approximator ELM for cross validation"""
    approximator_neurons::Int64
    """This will always be CATE and is used by summarize methods"""
    quantity_of_interest::String
    """This will always be set to false and is only accessed by summarize methods"""
    temporal::Bool
    """Number of neurons in the ELM used for estimating the abnormal returns"""
    num_neurons::Int64
    """The effect of exposure or treatment"""
    causal_effect::Array{Float64}
    """Extreme Learning Machine used for the first stage of estimation"""
    μ₀::ExtremeLearningMachine
    """Extreme Learning Machine used for the first stage of estimation"""
    μ₁::ExtremeLearningMachine

    function TLearner(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
        task="regression", regularized=false, activation=relu, validation_metric=mse, 
        min_neurons=1, max_neurons=100, folds=5, iterations=round(size(X, 1)/10), 
        approximator_neurons=round(size(X, 1)/10))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        new(Float64.(X), Float64.(T), Float64.(Y), task, regularized, activation,  
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, "CATE", false, 0)
    end
end

"""
    TLearner(X, T, Y; <keyword arguments>)

Initialize a T-Learner.

For an overview of T-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
- `task::String`: either regression or classification.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of folds to use for cross validation.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.
...

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = TLearner(X, T, Y)
julia> m2 = TLearner(X, T, Y; task="regression")
julia> m3 = TLearner(X, T, Y; task="regression", regularized=true)
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = TLearner(x_df, t_df, y_df)
julia> estimate_causal_effect!(m4)
```
"""
function TLearner(X, T, Y; task="regression", regularized=false, activation=relu, 
    validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
    iterations=round(size(X, 1)/10), approximator_neurons=round(size(X, 1)/10))

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    TLearner(X, T, Y; task=task, regularized=regularized, activation=activation, 
        validation_metric=validation_metric, min_neurons=min_neurons, 
        max_neurons=max_neurons, folds=folds, iterations=iterations, 
        approximator_neurons=approximator_neurons)
end

"""X-Learner for CATE estimation."""
mutable struct XLearner <: Metalearner
    """Covariates"""
    X::Array{Float64}
    """Treatment statuses"""
    T::Array{Float64}
    """Outome variable"""
    Y::Array{Float64}
    """Either regression or classification"""
    task::String
    """Whether to use L2 regularization"""
    regularized::Bool
    """Activation function to apply to the outputs from each neuron"""
    activation::Function
    """Validation metric to use when tuning the number of neurons"""
    validation_metric::Function
    """Minimum number of neurons to test in the hidden layer"""
    min_neurons::Int64
    """Maximum number of neurons to test in the hidden layer"""
    max_neurons::Int64
    """Number of cross validation folds"""
    folds::Int64
    """Number of iterations to perform cross validation"""
    iterations::Int64
    """Number of neurons in the hidden layer of the approximator ELM for cross validation"""
    approximator_neurons::Int64
    """This will always be CATE and is used by summarize methods"""
    quantity_of_interest::String
    """This will always be set to false and is only accessed by summarize methods"""
    temporal::Bool
    """Number of neurons in the ELM used for estimating the abnormal returns"""
    num_neurons::Int64
    """Extreme Learning Machine used for the first stage of estimation"""
    μ₀::ExtremeLearningMachine
    """Extreme Learning Machine used for the first stage of estimation"""
    μ₁::ExtremeLearningMachine
    """Individual propensity scores"""
    ps::Array{Float64}
    """The effect of exposure or treatment"""
    causal_effect::Array{Float64}

    function XLearner(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
        task="regression", regularized=false, activation=relu, validation_metric=mse, 
        min_neurons=1, max_neurons=100, folds=5, iterations=round(size(X, 1)/10), 
        approximator_neurons=round(size(X, 1)/10))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        new(Float64.(X), Float64.(T), Float64.(Y), task, regularized, activation,  
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, "CATE", false, 0)
    end
end

"""
    XLearner(X, T, Y; <keyword arguments>)

Initialize an X-Learner.

For an overview of X-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
- `task::String`: either regression or classification.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of folds to use for cross validation.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.
...

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> m2 = XLearner(X, T, Y; task="regression")
julia> m3 = XLearner(X, T, Y; task="regression", regularized=true)
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = XLearner(x_df, t_df, y_df)
julia> estimate_causal_effect!(m4)
```
"""
function XLearner(X, T, Y; task="regression", regularized=false, activation=relu, 
    validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
    iterations=round(size(X, 1)/10), approximator_neurons=round(size(X, 1)/10))

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    XLearner(X, T, Y; task=task, regularized=regularized, activation=activation, 
        validation_metric=validation_metric, min_neurons=min_neurons, 
        max_neurons=max_neurons, folds=folds, iterations=iterations, 
        approximator_neurons=approximator_neurons)
end

"""Container for the results of an R-learner"""
mutable struct RLearner <: Metalearner
    dml::DoubleMachineLearning
    """The effect of exposure or treatment"""
    causal_effect::Array{Float64}

"""
    RLearner(X, T, Y; <keyword arguments>)

Initialize an R-Learner.

For an explanation of R-Learner estimation see:
    Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
    effects." Biometrika 108, no. 2 (2021): 299-319.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
- `t_cat::Bool=false`: whether the treatment is categorical.
- `task::String`: either regression or classification.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of folds to use for cross validation.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.
...

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = RLearner(X, T, Y)
julia> m2 = RLearner(X, T, Y; t_cat=true)
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = RLearner(x_df, t_df, y_df)
julia> estimate_causal_effect!(m4)
```
"""
    function RLearner(X, T, Y; t_cat=false, activation=relu, validation_metric=mse, 
        min_neurons=1, max_neurons=100, folds=5, iterations=round(size(X, 1)/10), 
        approximator_neurons=round(size(X, 1)/10))

        new(DoubleMachineLearning(X, T, Y; t_cat=t_cat, regularized=true, 
            activation=activation, validation_metric=validation_metric, 
            min_neurons=min_neurons, max_neurons=max_neurons, folds=folds, 
            iterations=iterations, approximator_neurons=approximator_neurons))
    end
end

"""
    estimate_causal_effect!(s)

Estimate the CATE using an S-learner.

For an overview of S-learning see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m4 = SLearner(X, T, Y)
julia> estimate_causal_effect!(m4)
100-element Vector{Float64}
 0.20729633391630697
 0.20729633391630697
 0.20729633391630692
 ⋮
 0.20729633391630697
 0.20729633391630697
 0.20729633391630697
```
"""
function estimate_causal_effect!(s::SLearner)
    estimate_causal_effect!(s.g)
    Xₜ, Xᵤ= hcat(s.g.X, ones(size(s.g.T, 1))), hcat(s.g.X, zeros(size(s.g.T, 1)))
    s.causal_effect = vec(predict(s.g.learner, Xₜ) - predict(s.g.learner, Xᵤ))

    return s.causal_effect
end

"""
    estimate_causal_effect!(t)

Estimate the CATE using an T-learner.

For an overview of T-learning see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m5 = TLearner(X, T, Y)
julia> estimate_causal_effect!(m5)
100-element Vector{Float64}
 0.0493951571746305
 0.049395157174630444
 0.0493951571746305
 ⋮ 
 0.049395157174630444
 0.04939515717463039
 0.049395157174630444
```
"""
function estimate_causal_effect!(t::TLearner)
    x₀, x₁, y₀, y₁ = t.X[t.T .== 0,:], t.X[t.T .== 1,:], t.Y[t.T .== 0], t.Y[t.T .== 1]

    # Only search for the best number of neurons once and use the same number for inference
    if t.num_neurons === 0
        t.num_neurons = best_size(t.X, t.Y, t.validation_metric, t.task, t.activation, 
            t.min_neurons, t.max_neurons, t.regularized, t.folds, false, t.iterations, 
            t.approximator_neurons)
    end

    if t.regularized
        t.μ₀ = RegularizedExtremeLearner(x₀, y₀, t.num_neurons, t.activation) 
        t.μ₁ = RegularizedExtremeLearner(x₁, y₁, t.num_neurons, t.activation)
    else
        t.μ₀ = ExtremeLearner(x₀, y₀, t.num_neurons, t.activation) 
        t.μ₁ = ExtremeLearner(x₁, y₁, t.num_neurons, t.activation)
    end

    fit!(t.μ₀); fit!(t.μ₁)
    predictionsₜ, predictionsᵪ = predict(t.μ₁, t.X), predict(t.μ₀, t.X)
    t.causal_effect = @fastmath vec(predictionsₜ .- predictionsᵪ)

    return t.causal_effect
end

"""
    estimate_causal_effect!(x)

Estimate the CATE using an X-learner.

For an overview of X-learning see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
-0.025012644892878473
-0.024634294305967294
-0.022144246680543364
⋮ 
-0.021163590874553318
-0.014607310062509895
-0.022449034332142046
```
"""
function estimate_causal_effect!(x::XLearner)
    # Only search for the best number of neurons once and use the same number for inference
    if x.num_neurons === 0
        x.num_neurons = best_size(x.X, x.Y, x.validation_metric, x.task, x.activation, 
            x.min_neurons, x.max_neurons, x.regularized, x.folds, false, x.iterations, 
            x.approximator_neurons)
    end
    
    stage1!(x)
    μχ₀, μχ₁ = stage2!(x)

    x.causal_effect = @fastmath vec(((x.ps.*predict(μχ₀, x.X)) .+ 
        ((1 .- x.ps).*predict(μχ₁, x.X))))

    return x.causal_effect
end

"""
    estimate_causal_effect!(R)

Estimate the CATE using an R-learner.

For an overview of R-learning see:
    Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
    effects." Biometrika 108, no. 2 (2021): 299-319.

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = RLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
 -0.025012644892878473
 -0.024634294305967294
 -0.022144246680543364
 ⋮
 -0.021163590874553318
 -0.014607310062509895 
-0.022449034332142046
```
"""
function estimate_causal_effect!(R::RLearner)
    # Uses the same number of neurons for all phases of estimation
    if R.dml.num_neurons === 0
        R.dml.num_neurons = best_size(R.dml.X, R.dml.Y, R.dml.validation_metric, 
            "regression", R.dml.activation, R.dml.min_neurons, R.dml.max_neurons, 
            R.dml.regularized, R.dml.folds, false, R.dml.iterations, 
            R.dml.approximator_neurons)
    end

    # Just estimate the causal effect using the underlying DML and the weight trick
    R.causal_effect = estimate_effect!(R.dml, true)

    # Makes sure the right quantitiy of interest is printed out if summarize is called
    R.dml.quantity_of_interest = "CATE"

    return R.causal_effect
end

"""
stage1!(x)

Estimate the first stage models for an X-learner.

This method should not be called by the user.

```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> stage1!(m1)
```
"""
function stage1!(x::XLearner)
    if x.regularized
        g = RegularizedExtremeLearner(x.X, x.T, x.num_neurons, x.activation)
        x.μ₀ = RegularizedExtremeLearner(x.X[x.T .== 0,:], x.Y[x.T .== 0], x.num_neurons, 
            x.activation)
        x.μ₁ = RegularizedExtremeLearner(x.X[x.T .== 1,:], x.Y[x.T .== 1], x.num_neurons, 
            x.activation)
    else
        g = ExtremeLearner(x.X, x.T, x.num_neurons, x.activation)
        x.μ₀ = ExtremeLearner(x.X[x.T .== 0,:], x.Y[x.T .== 0], x.num_neurons, x.activation)
        x.μ₁ = ExtremeLearner(x.X[x.T .== 1,:], x.Y[x.T .== 1], x.num_neurons, x.activation)
    end

    # Get propensity scores
    fit!(g)
    x.ps = predict(g, x.X)

    # Fit first stage outcome models
    fit!(x.μ₀); fit!(x.μ₁)
end

"""
stage2!(x)

Estimate the second stage models for an X-learner.

This method should not be called by the user.

```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> stage1!(m1)
julia> stage2!(m1)
100-element Vector{Float64}
 0.6579129842054047
 0.7644471766429705
 0.5462780002052421
 ⋮
 0.8755515354984005
 0.947588000142362
 0.29294343704001025
```
"""
function stage2!(x::XLearner)
    d = ifelse(x.T === 0, predict(x.μ₁, x.X .- x.Y), x.Y .- predict(x.μ₀, x.X))

    if x.regularized
        μχ₀ = RegularizedExtremeLearner(x.X[x.T .== 0,:], d[x.T .== 0], x.num_neurons, 
            x.activation)
        μχ₁ = RegularizedExtremeLearner(x.X[x.T .== 1,:], d[x.T .== 1], x.num_neurons, 
            x.activation)
    else
        μχ₀ = ExtremeLearner(x.X[x.T .== 0,:], d[x.T .== 0], x.num_neurons, x.activation)
        μχ₁ = ExtremeLearner(x.X[x.T .== 1,:], d[x.T .== 1], x.num_neurons, x.activation) 
    end 

    fit!(μχ₀); fit!(μχ₁)

    return μχ₀, μχ₁
end
