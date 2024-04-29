"""Abstract type for metalearners"""
abstract type Metalearner end

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
```
"""
mutable struct SLearner <: Metalearner
    g::GComputation
    causal_effect::Array{Float64}

    function SLearner(X, T, Y; task="regression", regularized=true, activation=relu, 
                      validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
                      iterations=round(size(X, 1)/10), 
                      approximator_neurons=round(size(X, 1)/10))

        new(GComputation(X, T, Y, task=task, quantity_of_interest="ATE", 
                         regularized=regularized, activation=activation, temporal=false, 
                         validation_metric=validation_metric, min_neurons=min_neurons, 
                         max_neurons=max_neurons, folds=folds, iterations=iterations, 
                         approximator_neurons=approximator_neurons))
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
```
"""
mutable struct TLearner <: Metalearner
    X::Array{Float64}
    T::Array{Float64}
    Y::Array{Float64}
    task::String
    regularized::Bool
    activation::Function
    validation_metric::Function
    min_neurons::Int64
    max_neurons::Int64
    folds::Int64
    iterations::Int64
    approximator_neurons::Int64
    quantity_of_interest::String
    temporal::Bool
    num_neurons::Int64
    causal_effect::Array{Float64}
    μ₀::ExtremeLearningMachine
    μ₁::ExtremeLearningMachine

    function TLearner(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
                      task="regression", regularized=false, activation=relu, 
                      validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
                      iterations=round(size(X, 1)/10), 
                      approximator_neurons=round(size(X, 1)/10))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        new(Float64.(X), Float64.(T), Float64.(Y), task, regularized, activation,  
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, "CATE", false, 0)
    end
end

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
```
"""
mutable struct XLearner <: Metalearner
    X::Array{Float64}
    T::Array{Float64}
    Y::Array{Float64}
    task::String
    regularized::Bool
    activation::Function
    validation_metric::Function
    min_neurons::Int64
    max_neurons::Int64
    folds::Int64
    iterations::Int64
    approximator_neurons::Int64
    quantity_of_interest::String
    temporal::Bool
    num_neurons::Int64
    μ₀::ExtremeLearningMachine
    μ₁::ExtremeLearningMachine
    ps::Array{Float64}
    causal_effect::Array{Float64}

    function XLearner(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
                      task="regression", regularized=false, activation=relu, 
                      validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
                      iterations=round(size(X, 1)/10), 
                      approximator_neurons=round(size(X, 1)/10))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        new(Float64.(X), Float64.(T), Float64.(Y), task, regularized, activation,  
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, "CATE", false, 0)
    end
end

function XLearner(X, T, Y; task="regression", regularized=false, activation=relu, 
                  validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
                  iterations=round(size(X, 1)/10), 
                  approximator_neurons=round(size(X, 1)/10))

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    XLearner(X, T, Y; task=task, regularized=regularized, activation=activation, 
             validation_metric=validation_metric, min_neurons=min_neurons, 
             max_neurons=max_neurons, folds=folds, iterations=iterations, 
             approximator_neurons=approximator_neurons)
end

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
- `y_cat::Bool=false`: whether the outcome is categorical.
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
```
"""
mutable struct RLearner <: Metalearner
    dml::DoubleMachineLearning
    causal_effect::Array{Float64}

    function RLearner(X, T, Y; activation=relu, validation_metric=mse, min_neurons=1, 
        max_neurons=100, folds=5, iterations=round(size(X, 1)/10), 
        approximator_neurons=round(size(X, 1)/10))

        new(DoubleMachineLearning(X, T, Y; regularized=true, activation=activation, 
                                  validation_metric=validation_metric, 
                                  min_neurons=min_neurons, max_neurons=max_neurons, 
                                  folds=folds, iterations=iterations, 
                                  approximator_neurons=approximator_neurons))
    end
end

"""
    DoublyRobustLearner(X, T, Y; <keyword arguments>)

Initialize a doubly robust CATE estimator.

For an explanation of doubly robust cate estimation see:
    Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
    effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
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
julia> m1 = DoublyRobustLearner(X, T, Y)
julia> m2 = DoublyRobustLearnerLearner(X, T, Y; t_cat=true)
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = DoublyRobustLearner(x_df, t_df, y_df)
```
"""
mutable struct DoublyRobustLearner <: Metalearner
    X::Array{Float64}
    T::Array{Float64}
    Y::Array{Float64}
    regularized::Bool
    activation::Function
    validation_metric::Function
    min_neurons::Int64
    max_neurons::Int64
    folds::Int64
    iterations::Int64
    approximator_neurons::Int64
    quantity_of_interest::String
    temporal::Bool
    num_neurons::Int64
    causal_effect::Vector{Float64}

    function DoublyRobustLearner(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
                                 regularized=true, activation=relu, validation_metric=mse, 
                                 min_neurons=1, max_neurons=100, folds=5, 
                                 iterations=round(size(X, 1)/10), 
                                 approximator_neurons=round(size(X, 1)/10))

        new(Float64.(X), Float64.(T), Float64.(Y), regularized, activation, 
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, "CATE", false, 0, zeros(length(Y)))
    end
end

function DoublyRobustLearner(X, T, Y; regularized=true, activation=relu, 
                             validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
                             iterations=round(size(X, 1)/10), 
                             approximator_neurons=round(size(X, 1)/10))

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    DoublyRobustLearner(X, T, Y; regularized=regularized, activation=activation, 
                        validation_metric=validation_metric, min_neurons=min_neurons, 
                        max_neurons=max_neurons, folds=folds, iterations=iterations, 
                        approximator_neurons=approximator_neurons)
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
    y_type = var_type(s.g.Y)
    Xₜ, Xᵤ= hcat(s.g.X, ones(size(s.g.T, 1))), hcat(s.g.X, zeros(size(s.g.T, 1)))

    # Clipping binary predictions to be ∈ [0, 1]
    if s.g.task === "classification"
        yₜ = clip_if_binary(predict(s.g.learner, Xₜ), y_type)
        yᵤ = clip_if_binary(predict(s.g.learner, s.Xᵤ), y_type)
    else
        yₜ, yᵤ = predict(s.g.learner, Xₜ), predict(s.g.learner, Xᵤ)
    end

    s.causal_effect = yₜ - yᵤ

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
    type = var_type(t.Y)

    # Only search for the best number of neurons once and use the same number for inference
    if t.num_neurons === 0
        t.num_neurons = best_size(t.X, t.Y, t.validation_metric, t.task, t.activation, 
                                  t.min_neurons, t.max_neurons, t.regularized, t.folds, 
                                  false, t.iterations, t.approximator_neurons)
    end

    if t.regularized
        t.μ₀ = RegularizedExtremeLearner(x₀, y₀, t.num_neurons, t.activation) 
        t.μ₁ = RegularizedExtremeLearner(x₁, y₁, t.num_neurons, t.activation)
    else
        t.μ₀ = ExtremeLearner(x₀, y₀, t.num_neurons, t.activation) 
        t.μ₁ = ExtremeLearner(x₁, y₁, t.num_neurons, t.activation)
    end

    fit!(t.μ₀); fit!(t.μ₁)
    predictionsₜ = clip_if_binary(predict(t.μ₁, t.X), type)
    predictionsᵪ = clip_if_binary(predict(t.μ₀, t.X), type)
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
                                  x.min_neurons, x.max_neurons, x.regularized, x.folds, 
                                  false, x.iterations, x.approximator_neurons)
    end
    
    type = var_type(x.Y)
    stage1!(x)
    μχ₀, μχ₁ = stage2!(x)

    x.causal_effect = @fastmath vec(((x.ps.*clip_if_binary(predict(μχ₀, x.X), type)) .+ 
        ((1 .- x.ps).*clip_if_binary(predict(μχ₁, x.X), type))))

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
                                      "regression", R.dml.activation, R.dml.min_neurons, 
                                      R.dml.max_neurons, R.dml.regularized, R.dml.folds, 
                                      false, R.dml.iterations, R.dml.approximator_neurons)
    end

    # Just estimate the causal effect using the underlying DML and the weight trick
    R.causal_effect = estimate_effect!(R.dml, true)

    # Makes sure the right quantitiy of interest is printed out if summarize is called
    R.dml.quantity_of_interest = "CATE"

    return R.causal_effect
end

"""
    estimate_causal_effect!(DRE)

Estimate the CATE using a doubly robust learner.

For details on how this method estimates the CATE see:
    Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
    effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoublyRobustLearner(X, T, Y)
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
function estimate_causal_effect!(DRE::DoublyRobustLearner)
    X_T, Y = generate_folds(reduce(hcat, (DRE.X, DRE.T)), DRE.Y, 3)
    X, T = [fl[:, 1:size(DRE.X, 2)] for fl in X_T], [fl[:, size(DRE.X, 2)+1] for fl in X_T]

    # Uses the same number of neurons for all phases of estimation
    if DRE.num_neurons === 0
        DRE.num_neurons = best_size(DRE.X, DRE.Y, DRE.validation_metric, "regression", 
                                    DRE.activation, DRE.min_neurons, DRE.max_neurons, 
                                    DRE.regularized, DRE.folds, false, DRE.iterations, 
                                    DRE.approximator_neurons)
    end

    # Rotating folds for cross fitting
    for i in 1:3
        DRE.causal_effect .+= estimate_effect!(DRE, X, T, Y)
        X, T, Y = [X[3], X[1], X[2]], [T[3], T[1], T[2]], [Y[3], Y[1], Y[2]]
    end

    DRE.causal_effect ./= 3

    return DRE.causal_effect
end

"""
    estimate_effect!(DRE, X, T, Y)

Estimate the CATE for a single cross fitting iteration via doubly robust estimation.

This method should not be called directly.

...
# Arguments
- `DRE::DoublyRobustLearner`: the DoubleMachineLearning struct to estimate the effect for.
- `X`: a vector of three covariate folds.
- `T`: a vector of three treatment folds.
- `Y`: a vector of three outcome folds.
...

Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoublyRobustLearner(X, T, Y)
julia> estimate_effect!(m1)
 -0.025012644892878473
 -0.024634294305967294
 -0.022144246680543364
 ⋮
 -0.021163590874553318
 -0.014607310062509895 
 -0.022449034332142046
```
"""
function estimate_effect!(DRE::DoublyRobustLearner, X, T, Y)
    π_args = X[1], T[1], DRE.num_neurons, σ
    μ_arg = X[2], Y[2], DRE.num_neurons, DRE.activation
    y_type = var_type(DRE.Y)

    # Propensity scores
    π_e = DRE.regularized ? RegularizedExtremeLearner(π_args...) : ExtremeLearner(π_args...)
    fit!(π_e)
    π̂ = clip_if_binary(predict(π_e, DRE.X), Binary())

    # Outcome predictions
    μ₀_e = DRE.regularized ? RegularizedExtremeLearner(μ_arg...) : ExtremeLearner(μ_arg...)
    μ₁_e = DRE.regularized ? RegularizedExtremeLearner(μ_arg...) : ExtremeLearner(μ_arg...)
    fit!(μ₀_e); fit!(μ₁_e)
    μ₀̂  = clip_if_binary(predict(μ₀_e, DRE.X), y_type)
    μ₁̂  = clip_if_binary(predict(μ₁_e, DRE.X), y_type)

    # Pseudo outcomes
    ϕ̂  = ((DRE.T.-π̂) ./ (π̂ .*(1 .-π̂))).*(DRE.Y .-DRE.T.*μ₁̂  .-(1 .-DRE.T).*μ₀̂) .+ μ₁̂  .-μ₀̂

    # Final model
    τ_arg = X[3], ϕ̂[length(Y[1])+length(Y[2])+1:end], DRE.num_neurons, DRE.activation
    τ_est = DRE.regularized ? RegularizedExtremeLearner(τ_arg...) : ExtremeLearner(τ_arg...)
    fit!(τ_est)

    return clip_if_binary(predict(τ_est, DRE.X), var_type(DRE.Y))
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
    x.ps = clip_if_binary(predict(g, x.X), Binary())

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
    m₁ = clip_if_binary(predict(x.μ₁, x.X .- x.Y), var_type(x.Y))
    m₀ = clip_if_binary(predict(x.μ₀, x.X), var_type(x.Y))
    d = ifelse(x.T === 0, m₁, x.Y .- m₀)

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
