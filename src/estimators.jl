"""
Estimate causal effects with interrupted time series analysis, G-computation, and double 
machine learning using Extreme Learning machines.
   """
module Estimators

using ..ActivationFunctions: relu
using CausalELM: mean
using ..Metrics: mse
using ..CrossValidation: bestsize, shuffledata
using ..Models: ExtremeLearningMachine, ExtremeLearner, RegularizedExtremeLearner, fit!, 
    predictcounterfactual!, placebotest, predict

import CausalELM: estimate_causal_effect!, variable_type, Discrete, Continuous

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
        X₀ = ifelse(autoregression == true, reduce(hcat, (X₀, movingaverage(Y₀))), X₀)
        X₁ = ifelse(autoregression == true, reduce(hcat, (X₁, movingaverage(Y₁))), X₁)

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
    """The risk ratio for the estimated model"""
    risk_ratio::Real

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

        # If not panel or temporal data, randomly shuffle the indices for generating folds
        if !temporal
            X, Y, T = shuffledata(Float64.(X), Float64.(Y), Float64.(T))
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
    """Propensity score covariates"""
    Xₚ::Array{Float64}
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
    """Number of neurons in the ELM used for estimating the abnormal returns"""
    num_neurons::Int64
    """Propensity scores"""
    ps::Array{Float64}
    """Predicted outcomes for the control group"""
    μ₀::Array{Float64}
    """Predicted outcomes for the treatment group"""
    μ₁::Array{Float64}
    """The effect of exposure or treatment"""
    causal_effect::Float64
    """The risk ratio for discrete outcomes"""
    risk_ratio::Real
    """Whether the causal effect has been estimated"""
    fit::Bool

    """
DoubleMachineLearning(X, Xₚ, Y, T, task, quantity_of_interest, regularized, activation, 
    validation_metric, min_neurons, max_neurons, folds, iterations, approximator_neurons)

Initialize a double machine learning estimator with cross fitting.

For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.

Note that X, Xₚ, Y, and T must all contain floating point numbers.

Examples
```julia-repl
julia> X, Xₚ, Y, T =  rand(100, 5), rand(100, 4), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = DoubleMachineLearning(X, Xₚ, Y, T)
julia> m2 = DoubleMachineLearning(X, Xₚ, Y, T; task="regression")
julia> m3 = DoubleMachineLearning(X, Xₚ, Y, T; task="regression", quantity_of_interest="ATE)
```
"""
    function DoubleMachineLearning(X, Xₚ, Y, T; task="regression", 
        quantity_of_interest="ATE", regularized=true, activation=relu, 
        validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
        iterations=Int(round(size(X, 1)/10)), 
        approximator_neurons=Int(round(size(X, 1)/10)))

        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        elseif quantity_of_interest ∉ ("ATE", "ITE", "ATT")
            throw(ArgumentError("quantity_of_interest must be ATE, ITE, or ATT"))
        end

        if size(X, 1) !== size(Xₚ, 1)
            throw(ArgumentError("outcome and treatment covariates must have the same number 
                of observations"))
        end

        # Shuffles the data for cross validation
        X, Y, T = shuffledata(Float64.(X), Float64.(Y), Float64.(T))

        new(Float64.(X), Float64.(Xₚ), Float64.(Y), Float64.(T), task, quantity_of_interest, 
            regularized, activation, validation_metric, min_neurons, max_neurons, folds, 
            iterations, approximator_neurons, 0)
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
        its.num_neurons = bestsize(its.X₀, its.Y₀, its.validation_metric, its.task, 
            its.activation, its.min_neurons, its.max_neurons, its.regularized, its.folds, 
            its.iterations, its.approximator_neurons)
    end

    if its.regularized
        its.learner = RegularizedExtremeLearner(its.X₀, its.Y₀, its.num_neurons, 
            its.activation)
    else
        its.learner = ExtremeLearner(its.X₀, its.Y₀, its.num_neurons, its.activation)
    end

    its.β, its.Ŷ = fit!(its.learner), predictcounterfactual!(its.learner, its.X₁)
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

    # We will not find the best number of neurons after we have already estimated the causal
    # effect and are getting p-values, confidence intervals, or standard errors. We will use
    # the same number that was found when calling this method.
    if g.num_neurons === 0
        g.num_neurons = bestsize(Array(full_covariates), g.Y, g.validation_metric, g.task, 
            g.activation, g.min_neurons, g.max_neurons, g.regularized, g.folds,  
            g.iterations, g.approximator_neurons)
    end

    if g.regularized
        g.learner = RegularizedExtremeLearner(full_covariates, g.Y, g.num_neurons, 
            g.activation)
    else
        g.learner = ExtremeLearner(full_covariates, g.Y, g.num_neurons, g.activation)
    end

    g.β = fit!(g.learner)
    g.causal_effect = sum(predict(g.learner, Xₜ) - predict(g.learner, Xᵤ))/size(Xₜ, 1)

    # We use the risk ratio to calculate e-values in the exchangeability methods
    if variable_type(vec(g.Y)) == Continuous
        d = mean(vec(g.Y))/sqrt(var(vec(g.Y)))
        g.risk_ratio = exp(0.91 * d)
    else
        g.risk_ratio = mean(vec(predict(g.learner, Xₜ))/mean(vec(predict(g.learner, Xᵤ))))
    end

    g.fit = true
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
    propensity_scores = Array{Array{Float64, 1}}(undef, DML.folds)
    treatment_predictions = Array{Array{Float64, 1}}(undef, DML.folds)
    control_predictions = Array{Array{Float64, 1}}(undef, DML.folds)
    fold_level_effects = Array{Float64}(undef, DML.folds)
    X, Xₚ, Y, T = crossfittingsets(DML)

    # Uses the same number of neurons for all phases of estimation
    if DML.num_neurons === 0
        DML.num_neurons = bestsize(DML.X, DML.Y, DML.validation_metric, DML.task, 
            DML.activation, DML.min_neurons, DML.max_neurons, DML.regularized, DML.folds, 
            DML.iterations, DML.approximator_neurons)
    end

    for fold in 1:DML.folds 

        # All the data from the folds used for training
        X_train = reduce(vcat, X[1:end .!== fold])
        Xₚ_train = reduce(vcat, Xₚ[1:end .!== fold])
        Y_train = reduce(vcat, Y[1:end .!== fold])
        T_train = reduce(vcat, T[1:end .!== fold])

        # 0 and 1 subscripts denote treated and untreated units
        x₀_train, x₁_train = X_train[T_train .== 0, :], X_train[T_train .== 1, :]
        y₀_train, y₁_train = Y_train[T_train .== 0], Y_train[T_train .== 1]
        X_test, Xₚ_test, T_test, Y_test = X[fold], Xₚ[fold], T[fold], Y[fold]

        # Train on K-1 folds
        ps_model, μ₀_model = firststage!(DML, x₀_train, Xₚ_train, T_train, y₀_train)

        # Predict on fold K
        ps_pred = predictpropensityscore(ps_model, Xₚ_test)
        treatment_model = ate!(DML, x₁_train, y₁_train)
        treatment_pred = predicttreatmentoutcomes(treatment_model, X_test)
        control_pred = predictcontroloutcomes(μ₀_model, X_test)
        propensity_scores[fold], control_predictions[fold] = ps_pred, control_pred
        treatment_predictions[fold] = treatment_pred

        if DML.quantity_of_interest ∈ ("ATE", "ITE")
            E₁ = @fastmath mean(vec(T_test.*(Y_test.-treatment_pred)
                /(ps_pred.+treatment_pred)))
            E₀ = @fastmath mean(vec(((1 .-T_test).*(Y_test.-control_pred))/((1 .-ps_pred)) 
                .+control_pred))
            fold_level_effects[fold] = E₁ - E₀
    
        else DML.quantity_of_interest === "ATT"
            num = @fastmath ((1 .- T_test).*(Y_test .- control_pred))
            fold_level_effects[fold] = @fastmath mean(vec(num/((1 .- ps_pred) .+ control_pred)))
        end
    end
    DML.ps, DML.μ₀ = reduce(vcat, propensity_scores), reduce(vcat, control_predictions)

    # No outcome model for the treatment prediction if estimating ATT
    if DML.quantity_of_interest ∈ ("ATE", "ITE")
        DML.μ₁ = reduce(vcat, treatment_predictions)
    end

    # We use the risk ratio to calculate e-values in the exchangeability methods
    if variable_type(vec(DML.Y)) == Continuous
        d = mean(vec(DML.Y))/sqrt(var(vec(DML.Y)))
        DML.risk_ratio = exp(0.91 * d)
    else
        DML.risk_ratio = mean(mean.(treatment_predictions))/mean(mean.(control_predictions))
    end

    DML.causal_effect = mean(fold_level_effects)
    DML.fit = true
    return DML.causal_effect
end

"""
    firststage!(DML, x₀, xₚ, T, y₀)

Estimate the average treatment for the treated for a double machine learning estimator with 
cross fitting.

Examples
```julia-repl
julia> X, Xₚ Y, T =  rand(100, 5), rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = DoubleMachineLearning(X, Xₚ, Y, T)
julia> x₀, y₀ = m1.X[m1.T .== 0], m1.Y[m1.T .== 0]
julia> firststage!(m1, x₀, xₚ, T, y₀)
(Regularized Extreme Learning Machine with 10 hidden neurons, 
Regularized Extreme Learning Machine with 10 hidden neurons)
```
"""
function firststage!(DML::DoubleMachineLearning, x₀::Array{Float64}, xₚ::Array{Float64}, 
    T::Array{Float64}, y₀::Array{Float64})
    # Propensity score and separate outcome models
    if DML.regularized
        ps_model = RegularizedExtremeLearner(xₚ, T, DML.num_neurons, DML.activation)
        μ₀_model = RegularizedExtremeLearner(x₀, y₀, DML.num_neurons, DML.activation)
    else
        ps_model = ExtremeLearner(xₚ, T, DML.num_neurons, DML.activation)
        μ₀_model = ExtremeLearner(x₀, y₀, DML.num_neurons, DML.activation)
    end

    fit!(ps_model); fit!(μ₀_model)

    return ps_model, μ₀_model
end

"""
    ate!(DML, x₁, y₁)

Estimate the average treatment effect for a double machine learning estimator with cross 
fitting.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = DoubleMachineLearning(X, Y, T)
julia> x₁, y₁ = m1.X[m1.T .== 1], m1.Y[m1.T .== 1]
julia> ate!(m1, x₁, y₁)
Regularized Extreme Learning Machine with 10 hidden neurons
```
"""
function ate!(DML::DoubleMachineLearning, x₁::Array{Float64}, y₁::Array{Float64})
    if DML.regularized
        μ₁_model = RegularizedExtremeLearner(x₁, y₁, DML.num_neurons, DML.activation)
    else
        μ₁_model = ExtremeLearner(x₁, y₁, DML.num_neurons, DML.activation)
    end

    fit!(μ₁_model)

    return μ₁_model
end

"""
    predictpropensityscore(ps_model, x_pred)

Predict the propensity score for an out of sample fold for the double machine learning 
estimator.

Examples
```julia-repl
julia> X, Xₚ, Y, T =  rand(100, 5), rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> x_pred = rand(20, 5)
julia> m1 = DoubleMachineLearning(X, Xₚ, Y, T)
julia> ps_model, _ = firststage!(m1, x₀, y₀)
julia> predictpropensityscore(ps_model, x_pred)
```
"""
function predictpropensityscore(ps_model::ExtremeLearningMachine, x_pred::Array{Float64})
    return predict(ps_model, x_pred)
end

"""
    predictcontroloutcomes(control_model, x_pred)

Predict the counterfactual control outcomes for an out of sample fold.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> x_pred = rand(20, 5)
julia> m1 = DoubleMachineLearning(X, Y, T)
julia> control_model, _ = firststage!(m1, x₀, y₀)
julia> predictcontroloutcomes(control_model, x_pred)
```
"""
function predictcontroloutcomes(control_model::ExtremeLearningMachine, 
    x_pred::Array{Float64})
    return predict(control_model, x_pred)
end

"""
    predicttreatmentoutcomes(treatment_model, x_pred)

Predict the counterfactual treatment outcomes for an out of sample fold.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> x_pred = rand(20, 5)
julia> m1 = DoubleMachineLearning(X, Y, T)
julia> treatment_model, _ = firststage!(m1, x₀, y₀)
julia> predicttreatmentoutcomes(treatment_model, x_pred)
```
"""
function predicttreatmentoutcomes(treatment_model::ExtremeLearningMachine, 
    x_pred::Array{Float64})
    return predict(treatment_model, x_pred)
end

"""
    crossfitting_sets(DML)

Creates folds for cross fitting a double machine learning estimator.

Examples
```julia-repl
julia> xfolds, y_folds = crossfiting_sets(DML)
([[0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0], 
[0.0 0.0; 0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0; … ; 
0.0 0.0; 0.0 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 
0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
```
"""
function crossfittingsets(DML::DoubleMachineLearning)
    msg = """the number of folds must be less than the number of 
             observations and greater than or equal to iteration"""
    n = length(DML.Y)
    
    if DML.folds >= n throw(ArgumentError(msg)) end

    # Vectors of arrays for each fold in the covariates, propensity score covariates, 
    # outcome, and treatment
    x_set = Array{Array{Float64, 2}}(undef, DML.folds)
    xₚ_set = Array{Array{Float64, 2}}(undef, DML.folds)
    y_set = Array{Array{Float64, 1}}(undef, DML.folds)
    t_set = Array{Array{Float64, 1}}(undef, DML.folds)

    # Indices to start and stop
    stops = round.(Int, range(start=1, stop=n, length=DML.folds+1))

    # Indices to use for making folds
    indices = [s:e-(e < n)*1 for (s, e) in zip(stops[1:end-1], stops[2:end])]

    for (i, idx) in pairs(indices)
        x_set[i], xₚ_set[i] = DML.X[idx, :], DML.Xₚ[idx, :]
        y_set[i], t_set[i] = DML.Y[idx], DML.T[idx]
    end

    return x_set, xₚ_set, y_set, t_set
end

"""
    movingaverage(g)

Calculates a cumulative moving average.

Examples
```julia-repl
julia> movingaverage([1, 2, 3])
3-element Vector{Float64}
1.0
1.5
2.0
```
"""
function movingaverage(g::Vector{Float64})
    result = similar(g)
    for i = 1:length(g)
        result[i] = mean(g[1:i])
    end
    return result
end

end
