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
    """Either \"regression\" or \"classification\""""
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
    """Whether to a cumulative moving average as an autoregressive term"""
    autoregression::Bool
    """Number of neurons in the ELM used for estimating the abnormal returns"""
    num_neurons::Int64
    """Extreme Learning Machine used to estimate the abnormal returns"""
    learner::ExtremeLearningMachine
    """Weights learned during training"""
    β::Array{Float64}
    """
    Counterfactual predicted for the post-treatment period using wieghts learned in the 
    pre-treatment period
    """
    Ŷ::Array{Float64}
    """
    The difference betwen the predicted counterfactual and the observed data at each 
    interval during the post-treatment period

    """
    Δ::Array{Float64}

"""
    InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; task, regularized, activation, validation_metric, 
        min_neurons, max_neurons, folds, iterations, approximator_neurons)

Initialize an interrupted time series estimator. 

For a simple linear regression-based tutorial on interrupted time series analysis see:
    Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
    regression for the evaluation of public health interventions: a tutorial." International 
    journal of epidemiology 46, no. 1 (2017): 348-355.

Note that X₀, Y₀, X₁, and Y₁ must all be floating point numbers and ordered temporally from 
the least to the most recent observation.

Examples
```julia-repl
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> m2 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; task="regression")
julia> m3 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; task="regression", regularized=true)
julia> m4 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; task="regression", regularized=true, 
           activation=relu)
```
"""
    function InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; task="regression", regularized=true, 
        activation=relu, validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
        iterations=Int(round(size(X₀, 1)/10)), 
        approximator_neurons=Int(round(size(X₀, 1)/10)), autoregression=true)

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        # Add autoregressive term
        X₀ = ifelse(autoregression == true, reduce(hcat, (X₀, moving_average(Y₀))), X₀)
        X₁ = ifelse(autoregression == true, reduce(hcat, (X₁, moving_average(Y₁))), X₁)

        new(X₀, Float64.(Y₀), X₁, Float64.(Y₁), task, regularized, activation, 
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, autoregression, 0)
    end
end

"""Container for the results of G-Computation"""
mutable struct GComputation <: CausalEstimator
    """Covariates"""
    X::Array{Float64}
    """Outomes variable"""
    Y::Array{Float64}
    """Treatment statuses"""
    T::Array{Float64}
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
    """Number of neurons in the ELM used for estimating the abnormal returns"""
    num_neurons::Int64
    """Whether the causal effect has been estimated"""
    fit::Bool
    """Extreme Learning Machine used to estimate the causal effect"""
    learner::ExtremeLearningMachine
    """Weights learned during training"""
    β::Array{Float64}
    """The effect of exposure or treatment"""
    causal_effect::Float64

"""
GComputation(X, Y, T, task, quantity_of_interest, regularized, activation, temporal, 
    validation_metric, min_neurons, max_neurons, folds, iterations, approximator_neurons)

Initialize a G-Computation estimator.

For a good overview of G-Computation see:
    Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
    Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
    Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
    estimator for causal inference with different covariates sets: a comparative simulation 
    study." Scientific reports 10, no. 1 (2020): 9219.

Note that X, Y, and T must all be floating point numbers.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = GComputation(X, Y, T)
julia> m2 = GComputation(X, Y, T; task="regression")
julia> m3 = GComputation(X, Y, T; task="regression", quantity_of_interest="ATE)
julia> m4 = GComputation(X, Y, T; task="regression", quantity_of_interest="ATE, 
julia> regularized=true)
```
"""
    function GComputation(X, Y, T; task="regression", quantity_of_interest="ATE", 
        regularized=true,activation=relu, temporal=true, validation_metric=mse, 
        min_neurons=1, max_neurons=100, folds=5, iterations=Int(round(size(X, 1)/10)), 
        approximator_neurons=Int(round(size(X, 1)/10)))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        elseif quantity_of_interest ∉ ("ATE", "ITT", "ATT")
            throw(ArgumentError("quantity_of_interest must be ATE, ITT, or ATT"))
        end

        new(X, Y, T, task, quantity_of_interest, regularized, activation, temporal, 
            validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, 0, false)
    end
end

"""Container for the results of a double machine learning estimator"""
mutable struct DoubleMachineLearning <: CausalEstimator
    """Covariates"""
    X::Array{Float64}
    """Outomes variable"""
    Y::Array{Float64}
    """Treatment statuses"""
    T::Array{Float64}
    """Either regression or classification"""
    task::String
    """Either ATE, ITT, or ATT"""
    quantity_of_interest::String
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
    """Number of neurons in the ELM used for estimating the causal effect"""
    num_neurons::Int64
    """Residuals from the outcome predictions"""
    Ỹ::Array{Float64}
    """Residuals from the treatment predictions"""
    T̃::Array{Float64}
    """The estimated coefficients for the partially linear regression"""
    coefficients::Vector{Float64}
    """The effect of exposure or treatment"""
    causal_effect::Float64
    """Whether the causal effect has been estimated"""
    fit::Bool

"""
DoubleMachineLearning(X, Y, T, task, quantity_of_interest, regularized, activation, 
    validation_metric, min_neurons, max_neurons, folds, iterations, approximator_neurons)

Initialize a double machine learning estimator with cross fitting.

For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.

Note that X, Y, and T must all contain floating point numbers.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = DoubleMachineLearning(X, Xₚ, Y, T)
julia> m2 = DoubleMachineLearning(X, Xₚ, Y, T; task="regression")
julia> m3 = DoubleMachineLearning(X, Xₚ, Y, T; task="regression", quantity_of_interest="ATE)
```
"""
    function DoubleMachineLearning(X, Y, T; task="regression", 
        quantity_of_interest="ATE", regularized=true, activation=relu, 
        validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
        iterations=Int(round(size(X, 1)/10)), 
        approximator_neurons=Int(round(size(X, 1)/10)))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        elseif quantity_of_interest ∉ ("ATE", "ITE", "ATT")
            throw(ArgumentError("quantity_of_interest must be ATE, ITE, or ATT"))
        end

        new(Float64.(X), Float64.(Y), Float64.(T), task, quantity_of_interest, regularized, 
            activation, validation_metric, min_neurons, max_neurons, folds, iterations, 
            approximator_neurons, 0)
    end
end

"""
    estimate_causal_effect!(its)

Estimate the effect of an event relative to a predicted counterfactual.

Examples
```julia-repl
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
        its.num_neurons = best_size(its.X₀, its.Y₀, its.validation_metric, its.task, 
            its.activation, its.min_neurons, its.max_neurons, its.regularized, its.folds, 
            true, its.iterations, its.approximator_neurons)
    end

    if its.regularized
        its.learner = RegularizedExtremeLearner(its.X₀, its.Y₀, its.num_neurons, 
            its.activation)
    else
        its.learner = ExtremeLearner(its.X₀, its.Y₀, its.num_neurons, its.activation)
    end

    its.β, its.Ŷ = fit!(its.learner), predict_counterfactual!(its.learner, its.X₁)
    its.Δ = its.Ŷ - its.Y₁

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
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m2 = GComputation(X, Y, T)
julia> estimate_causal_effect!(m2)
0.31067439
```
"""
function estimate_causal_effect!(g::GComputation)
    full_covariates = hcat(g.X, g.T)

    if g.quantity_of_interest ∈ ("ITT", "ATE")
        Xₜ, Xᵤ= hcat(g.X, ones(size(g.T, 1))), hcat(g.X, zeros(size(g.T, 1)))
    else
        Xₜ, Xᵤ = hcat(g.X, g.T), hcat(g.X, zeros(size(g.Y, 1)))
    end

    # This makes sure we don't search for the best number of neurons after we have already 
    # found it
    if g.num_neurons === 0
        g.num_neurons = best_size(Array(full_covariates), g.Y, g.validation_metric, g.task, 
            g.activation, g.min_neurons, g.max_neurons, g.regularized, g.folds, g.temporal, 
            g.iterations, g.approximator_neurons)
    end

    if g.regularized
        g.learner = RegularizedExtremeLearner(full_covariates, g.Y, g.num_neurons, 
            g.activation)
    else
        g.learner = ExtremeLearner(full_covariates, g.Y, g.num_neurons, g.activation)
    end

    g.β = fit!(g.learner)
    g.fit = true
    g.causal_effect = sum(predict(g.learner, Xₜ) - predict(g.learner, Xᵤ))/size(Xₜ, 1)

    return g.causal_effect
end

"""
    estimate_causal_effect!(DML)

Estimate a causal effect of interest using double machine learning.

Unlike other estimators, this method does not support time series or panel data. This method 
also does not work as well with smaller datasets because it estimates separate outcome 
models for the treatment and control groups.

Examples
```julia-repl
julia> X, Xₚ, Y, T =  rand(100, 5), rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m3 = DoubleMachineLearning(X, Xₚ, Y, T)
julia> estimate_causal_effect!(m3)
0.31067439
```
"""
function estimate_causal_effect!(DML::DoubleMachineLearning)
    # Uses the same number of neurons for all phases of estimation
    if DML.num_neurons === 0
        DML.num_neurons = best_size(DML.X, DML.Y, DML.validation_metric, DML.task, 
            DML.activation, DML.min_neurons, DML.max_neurons, DML.regularized, DML.folds, 
            false, DML.iterations, DML.approximator_neurons)
    end

    predict_residuals!(DML)

    # Estimate the final model with a column of ones for the intercept
    DML.coefficients = reduce(hcat, (DML.T̃, ones(length(DML.T̃))))\DML.Ỹ
    DML.causal_effect = DML.coefficients[1]
    DML.fit = true

    return DML.causal_effect
end

function predict_residuals!(DML::DoubleMachineLearning)
    X_T, Y = generate_folds(reduce(hcat, (DML.X, DML.T)), DML.Y, DML.folds)
    X = [fold[:, 1:size(DML.X, 2)] for fold in X_T]
    T = [fold[:, size(DML.X, 2)+1] for fold in X_T]
    DML.Ỹ = Vector{Float64}(undef, length(DML.Y))
    DML.T̃ = Vector{Float64}(undef, length(DML.T))
    idx = 1  # Keeps track of what indices to start appending residuals

    for fold in 1:DML.folds
        X_train, X_test = reduce(vcat, X[1:end .!== fold]), X[fold]
        Y_train , Y_test= reduce(vcat, Y[1:end .!== fold]), Y[fold]
        T_train, T_test = reduce(vcat, T[1:end .!== fold]), T[fold]

        if DML.regularized
            y = RegularizedExtremeLearner(X_train, Y_train, DML.num_neurons, DML.activation)
            t = RegularizedExtremeLearner(X_train, T_train, DML.num_neurons, DML.activation)
        else
            y = ExtremeLearner(X_train, Y_train, DML.num_neurons, DML.activation)
            t = ExtremeLearner(X_train, T_train, DML.num_neurons, DML.activation)
        end

        fit!(y); fit!(t)
        DML.Ỹ[idx:(idx+length(Y_test))-1] = (predict(y, X_test)-Y_test)
        DML.T̃[idx:(idx+length(T_test))-1] = (predict(t, X_test)-T_test)
        idx += lastindex(Y_test)
    end
end

"""
    moving_average(g)

Calculates a cumulative moving average.

Examples
```julia-repl
julia> moving_average([1, 2, 3])
3-element Vector{Float64}
1.0
1.5
2.0
```
"""
function moving_average(g::Vector{Float64})
    result = similar(g)
    for i = 1:length(g)
        result[i] = mean(g[1:i])
    end
    return result
end
