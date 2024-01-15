"""Abstract type for GComputation and DoubleMachineLearning"""
abstract type  CausalEstimator end

"""Container for the results of an interrupted time series analysis"""
mutable struct InterruptedTimeSeries
    """Covariates for the pre-event period"""
    X₀::Array{Float64}
    """Outcomes for the pre-event period"""
    Y₀::Array{Float64}
    """Covariates for the post-event period"""
    X₁::Array{Float64}
    """Outcomes for the post-event period"""
    Y₁::Array{Float64}
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
    """Whether to a cumulative moving average as an autoregressive term"""
    autoregression::Bool
    """Number of neurons in the ELM used for estimating the abnormal returns"""
    num_neurons::Int64
    """
    The difference betwen the predicted counterfactual and the observed data at each 
    interval during the post-treatment period

    """
    Δ::Array{Float64}

    function InterruptedTimeSeries(X₀::Array{<:Real}, Y₀::Array{<:Real}, X₁::Array{<:Real}, 
        Y₁::Array{<:Real}; regularized=true, activation=relu, validation_metric=mse, 
        min_neurons=1, max_neurons=100, folds=5, iterations=round(size(X₀, 1)/10), 
        approximator_neurons=round(size(X₀, 1)/10), autoregression=true)

        # Add autoregressive term
        X₀ = ifelse(autoregression == true, reduce(hcat, (X₀, moving_average(Y₀))), X₀)
        X₁ = ifelse(autoregression == true, reduce(hcat, (X₁, moving_average(Y₁))), X₁)

        new(X₀, Float64.(Y₀), Float64.(X₁), Float64.(Y₁), regularized, activation, 
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, autoregression, 0)
    end
end

"""
    InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; <keyword arguments>)

Initialize an interrupted time series estimator. 

For a simple linear regression-based tutorial on interrupted time series analysis see:
    Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
    regression for the evaluation of public health interventions: a tutorial." International 
    journal of epidemiology 46, no. 1 (2017): 348-355.

...
# Arguments
- `X₀::Any`: an array or DataFrame of covariates from the pre-treatment period.
- `Y₁::Any`: an array or DataFrame of outcomes from the pre-treatment period.
- `X₁::Any`: an array or DataFrame of covariates from the post-treatment period.
- `Y₀::Any`: an array or DataFrame of outcomes from the post-treatment period.
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
```jldoctest
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> m2 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; regularized=false)
julia> x₀_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100))
julia> y₀_df = DataFrame(y=rand(100))
julia> x₁_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100)) 
julia> y₁_df = DataFrame(y=rand(100))
julia> m3 = InterruptedTimeSeries(x₀_df, y₀_df, x₁_df, y₁_df)
julia> estimate_causal_effect!(m3)
```
"""
function InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; regularized=true, activation=relu, 
    validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
    iterations=round(size(X₀, 1)/10), approximator_neurons=round(size(X₀, 1)/10), 
    autoregression=true)

    # Convert to arrays
    X₀, X₁, Y₀, Y₁ = Matrix{Float64}(X₀), Matrix{Float64}(X₁), Y₀[:, 1], Y₁[:, 1]

    InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; regularized=regularized, 
        activation=activation, validation_metric=validation_metric, 
        min_neurons=min_neurons, max_neurons=max_neurons, folds=folds, 
        iterations=iterations, approximator_neurons=approximator_neurons, 
        autoregression=autoregression)
end

"""Container for the results of G-Computation"""
mutable struct GComputation <: CausalEstimator
    """Covariates"""
    X::Array{Float64}
    """Treatment statuses"""
    T::Array{Float64}
    """Outomes variable"""
    Y::Array{Float64}
    """Either regression or classification"""
    task::String
    """Either ATE, ITT, or ATT"""
    quantity_of_interest::String
    """Whether to use L2 regularization"""
    regularized::Bool
    """Activation function to apply to the outputs from each neuron"""
    activation::Function
    """Whether the data is of a temporal nature eg. (time series, panel data)"""
    temporal::Bool
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
    """Number of neurons in the ELM used for estimating the causal effect"""
    num_neurons::Int64
    """The effect of exposure or treatment"""
    causal_effect::Float64
    """The model used to predict the outcomes"""
    learner::ExtremeLearningMachine

    function GComputation(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
        task="regression", quantity_of_interest="ATE", regularized=true, activation=relu, 
        temporal=true, validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
        iterations=round(size(X, 1)/10), approximator_neurons=round(size(X, 1)/10))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        elseif quantity_of_interest ∉ ("ATE", "ITT", "ATT")
            throw(ArgumentError("quantity_of_interest must be ATE, ITT, or ATT"))
        end

        new(Float64.(X), Float64.(T), Float64.(Y), task, quantity_of_interest, regularized, 
            activation, temporal, validation_metric, min_neurons, max_neurons, folds, 
            iterations, approximator_neurons, 0, NaN)
    end
end

"""
    GComputation(X, T, Y; <keyword arguments>)

Initialize a G-Computation estimator.

For a good overview of G-Computation see:
    Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
    Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
    Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
    estimator for causal inference with different covariates sets: a comparative simulation 
    study." Scientific reports 10, no. 1 (2020): 9219.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
- `task::String`: either regression or classification.
- `quantity_of_interest::String`: ATE for average treatment effect or CTE for cummulative 
    treatment effect.
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
```jldoctest
julia> X, T, Y =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = GComputation(X, T, Y)
julia> m2 = GComputation(X, T, Y; task="regression")
julia> m3 = GComputation(X, T, Y; task="regression", quantity_of_interest="ATE)
julia> m4 = GComputation(X, T, Y; task="regression", quantity_of_interest="ATE, 
julia> regularized=true)
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100)) 
julia> m5 = GComputation(x_df, t_df, y_df)
julia> estimate_causal_effect!(m5)
```
"""
function GComputation(X, T, Y; task="regression", quantity_of_interest="ATE", 
    regularized=true, activation=relu, temporal=true, validation_metric=mse, min_neurons=1, 
    max_neurons=100, folds=5, iterations=round(size(X, 1)/10), 
    approximator_neurons=round(size(X, 1)/10))

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    GComputation(X, T, Y; task=task, quantity_of_interest=quantity_of_interest, 
        regularized=regularized, activation=activation, temporal=temporal, 
        validation_metric=validation_metric, min_neurons=min_neurons, 
        max_neurons=max_neurons, folds=folds, iterations=iterations, 
        approximator_neurons=approximator_neurons)
end

"""Container for the results of a double machine learning estimator"""
mutable struct DoubleMachineLearning <: CausalEstimator
    """Covariates"""
    X::Array{Float64}
    """Treatment statuses"""
    T::Array{Float64}
    """Outomes variable"""
    Y::Array{Float64}
    """True if the treatment variable is categorical and nonbinary"""
    t_cat::Bool
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
    """Number of folds to use in cross validation and cross fitting"""
    folds::Int64
    """Number of iterations to perform cross validation"""
    iterations::Int64
    """Number of neurons in the hidden layer of the approximator ELM for cross validation"""
    approximator_neurons::Int64
    """This will alsways be ATE unless using DML with the weight trick for R-learning"""
    quantity_of_interest::String
    """This will always be false and is just exists to be accessed by summzrize methods"""
    temporal::Bool
    """Number of neurons in the ELM used for estimating the causal effect"""
    num_neurons::Int64
    """The effect of exposure or treatment"""
    causal_effect::Float64

    function DoubleMachineLearning(X::Array{<:Real}, T::Array{<:Real}, Y::Array{<:Real}; 
        t_cat=false, regularized=true, activation=relu, validation_metric=mse, 
        min_neurons=1, max_neurons=100, folds=5, iterations=round(size(X, 1)/10), 
        approximator_neurons=round(size(X, 1)/10))

        new(Float64.(X), Float64.(T), Float64.(Y), t_cat, regularized, activation, 
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, "ATE", false, 0, NaN)
    end
end

"""
    DoubleMachineLearning(X, T, Y; <keyword arguments>)

Initialize a double machine learning estimator with cross fitting.

For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.

...
# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.
- `t_cat::Bool=false`: whether the treatment is categorical.
- `task::String`: either regression or classification.
- `quantity_of_interest::String`: ATE for average treatment effect or CTE for cummulative 
    treatment effect.
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
```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> m2 = DoubleMachineLearning(X, T, Y; task="regression")
julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m3 = DoubleMachineLearning(x_df, t_df, y_df)
julia> estimate_causal_effect!(m3)
```
"""
function DoubleMachineLearning(X, T, Y; t_cat=false, regularized=true, activation=relu, 
    validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
    iterations=round(size(X, 1)/10), approximator_neurons=round(size(X, 1)/10))

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    DoubleMachineLearning(X, T, Y; t_cat=t_cat, regularized=regularized, 
        activation=activation, validation_metric=validation_metric, min_neurons=min_neurons, 
        max_neurons=max_neurons, folds=folds, iterations=iterations, 
        approximator_neurons=approximator_neurons)
end

"""
    estimate_causal_effect!(its)

Estimate the effect of an event relative to a predicted counterfactual.

Examples
```jldoctest
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
 0.25714308
```
"""
function estimate_causal_effect!(its::InterruptedTimeSeries)
    # We will not find the best number of neurons after we have already estimated the causal
    # effect and are getting p-values, confidence intervals, or standard errors. We will use
    # the same number that was found when calling this method.
    if its.num_neurons === 0
        its.num_neurons = best_size(its.X₀, its.Y₀, its.validation_metric, "regression", 
            its.activation, its.min_neurons, its.max_neurons, its.regularized, its.folds, 
            true, its.iterations, its.approximator_neurons)
    end

    if its.regularized
        learner = RegularizedExtremeLearner(its.X₀, its.Y₀, its.num_neurons, its.activation)
    else
        learner = ExtremeLearner(its.X₀, its.Y₀, its.num_neurons, its.activation)
    end

    fit!(learner)
    its.Δ = predict_counterfactual!(learner, its.X₁) - its.Y₁

    return its.Δ
end

"""
    estimate_causal_effect!(g)

Estimate a causal effect of interest using G-Computation.

If treatents are administered at multiple time periods, the effect will be estimated as the 
average difference between the outcome of being treated in all periods and being treated in 
no periods.For example, given that individuals 1, 2, ..., i ∈ I recieved either a treatment 
or a placebo in p different periods, the model would estimate the average treatment effect 
as E[Yᵢ|T₁=1, T₂=1, ... Tₚ=1, Xₚ] - E[Yᵢ|T₁=0, T₂=0, ... Tₚ=0, Xₚ].

Examples
```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = GComputation(X, T, Y)
julia> estimate_causal_effect!(m1)
 0.31067439
```
"""
function estimate_causal_effect!(g::GComputation)
    covariates = hcat(g.X, g.T)

    if g.quantity_of_interest ∈ ("ITT", "ATE")
        Xₜ, Xᵤ= hcat(g.X, ones(size(g.T, 1))), hcat(g.X, zeros(size(g.T, 1)))
    else
        Xₜ, Xᵤ = hcat(g.X, g.T), hcat(g.X, zeros(size(g.Y, 1)))
    end

    if g.num_neurons === 0  # Don't search for the best number of neurons multiple times
        g.num_neurons = best_size(Array(covariates), g.Y, g.validation_metric, g.task, 
            g.activation, g.min_neurons, g.max_neurons, g.regularized, g.folds, g.temporal, 
            g.iterations, g.approximator_neurons)
    end

    if g.regularized
        g.learner = RegularizedExtremeLearner(covariates, g.Y, g.num_neurons, g.activation)
    else
        g.learner = ExtremeLearner(covariates, g.Y, g.num_neurons, g.activation)
    end

    fit!(g.learner)
    g.causal_effect = mean(vec(predict(g.learner, Xₜ) - predict(g.learner, Xᵤ)))
    return g.causal_effect
end

"""
    estimate_causal_effect!(DML)

Estimate a causal effect of interest using double machine learning.

Unlike other estimators, this method does not support time series or panel data. This method 
also does not work as well with smaller datasets because it estimates separate outcome 
models for the treatment and control groups.

Examples
```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> estimate_causal_effect!(m1)
 0.31067439
```
"""
function estimate_causal_effect!(DML::DoubleMachineLearning)
    # Uses the same number of neurons for all phases of estimation
    if DML.num_neurons === 0
        DML.num_neurons = best_size(DML.X, DML.Y, DML.validation_metric, "regression", 
            DML.activation, DML.min_neurons, DML.max_neurons, DML.regularized, DML.folds, 
            false, DML.iterations, DML.approximator_neurons)
    end

    estimate_effect!(DML, false)
    DML.causal_effect /= DML.folds

    return DML.causal_effect
end

"""
    estimate_effect!(DML, cate=false)

Estimate a treatment effect using double machine learning.

This method should not be called directly.

...
# Arguments
- `DML::DoubleMachineLearning`: the DoubleMachineLearning struct to estimate the effect for.
- `cate::Bool=fales`: whether to estimate the cate.
...

Examples
```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> estimate_effect!(m1)
 0.31067439
```
"""
function estimate_effect!(DML::DoubleMachineLearning, cate=false)
    X_T, Y = generate_folds(reduce(hcat, (DML.X, DML.T)), DML.Y, DML.folds)
    X, T = [fl[:, 1:size(DML.X, 2)] for fl in X_T], [fl[:, size(DML.X, 2)+1] for fl in X_T]
    predictors = cate ? Vector{RegularizedExtremeLearner}(undef, DML.folds) : Nothing
    DML.causal_effect = 0

    # Cross fitting by training on the main folds and predicting residuals on the auxillary
    for fld in 1:DML.folds
        X_train, X_test = reduce(vcat, X[1:end .!== fld]), X[fld]
        Y_train, Y_test = reduce(vcat, Y[1:end .!== fld]), Y[fld]
        T_train, T_test = reduce(vcat, T[1:end .!== fld]), T[fld]

        Ỹ, T̃ = predict_residuals(DML, X_train, X_test, Y_train, Y_test, T_train, T_test)
        DML.causal_effect += (reduce(hcat, (T̃, ones(length(T̃))))\Ỹ)[1]

        if cate  # Using the weight trick to get the non-parametric CATE for an R-learner
            X[fld], Y[fld] = (T̃.^2) .* X_test, (T̃.^2) .* (Ỹ./T̃)
            mod = RegularizedExtremeLearner(X[fld], Y[fld], DML.num_neurons, DML.activation)
            fit!(mod)
            predictors[fld] = mod
        end
    end
    final_predictions = cate ? [predict(m, reduce(vcat, X)) for m in predictors] : Nothing
    cate && return vec(mapslices(mean, reduce(hcat, final_predictions), dims=2))
end

"""
    predict_residuals(DML, x_train, x_test, y_train, y_test, t_train, t_test)

Predict treatment and outcome residuals for doubl machine learning.

This method should not be called directly.

Examples
```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> x_train, x_test = X[1:80, :], X[81:end, :]
julia> y_train, y_test = Y[1:80], Y[81:end]
julia> t_train, t_test = T[1:80], T[81:100]
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> predict_residuals(m1, x_train, x_test, y_train, y_test, t_train, t_test)
100-element Vector{Float64}
 0.6944714802199426
 0.6102318624294397
 0.9563033347529682
 ⋮
 0.14016601301278353, 
 0.2217194742841072
 0.199372555924635
```
"""
function predict_residuals(DML::DoubleMachineLearning, x_train, x_test, y_train, y_test, 
    t_train, t_test)
    # One hot encode categorical variables for multiple treatments
    t_train = DML.t_cat && var_type(DML.T) == Count() ? one_hot_encode(t_train) : t_train
    activation = var_type(DML.T) == Binary() ? σ : DML.activation
    
    if DML.regularized
        y = RegularizedExtremeLearner(x_train, y_train, DML.num_neurons, DML.activation)
        t = RegularizedExtremeLearner(x_train, t_train, DML.num_neurons, activation)
    else
        y = ExtremeLearner(x_train, y_train, DML.num_neurons, DML.activation)
        t = ExtremeLearner(x_train, t_train, DML.num_neurons, activation)
    end

    fit!(y); fit!(t)

    if DML.t_cat  && var_type(DML.T) == Count() # Multiclass residual = 1-Pr(Majority Class)
        T̃ = 1 .- vec(mapslices(maximum, softmax((predict(t, x_test))), dims=2))
    else
        T̃ = (t_test-predict(t, x_test))
    end

    return (y_test-predict(y, x_test)), T̃
end

"""
    moving_average(x)

Calculates a cumulative moving average.

Examples
```jldoctest
julia> moving_average([1, 2, 3])
3-element Vector{Float64}
 1.0
 1.5
 2.0
```
"""
function moving_average(x)
    result = similar(x)
    for i = 1:length(x)
        result[i] = mean(x[1:i])
    end
    return result
end
