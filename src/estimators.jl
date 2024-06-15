"""Abstract type for GComputation and DoubleMachineLearning"""
abstract type CausalEstimator end

"""
    InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; kwargs...)

Initialize an interrupted time seriFes estimator. 

# Arguments
- `X₀::Any`: an array or DataFrame of covariates from the pre-treatment period.
- `Y₁::Any`: an array or DataFrame of outcomes from the pre-treatment period.
- `X₁::Any`: an array or DataFrame of covariates from the post-treatment period.
- `Y₁::Any`: an array or DataFrame of outcomes from the post-treatment period.
- `regularized::Function=true`: whether to use L2 regularization

# Keywords
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of cross validation folds to find the best number of neurons.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.

# Notes
If regularized is set to true then the ridge penalty will be estimated using generalized 
cross validation where the maximum number of iterations is 2 * folds for the successive 
halving procedure. However, if the penalty in on iteration is approximately the same as in 
the previous penalty, then the procedure will stop early.

# References
For a simple linear regression-based tutorial on interrupted time series analysis see:
    Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
    regression for the evaluation of public health interventions: a tutorial." International 
    journal of epidemiology 46, no. 1 (2017): 348-355.

For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
m2 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; regularized=false)
x₀_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100))
y₀_df = DataFrame(y=rand(100))
x₁_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100)) 
y₁_df = DataFrame(y=rand(100))
m3 = InterruptedTimeSeries(x₀_df, y₀_df, x₁_df, y₁_df)
```
"""
mutable struct InterruptedTimeSeries
    X₀::Array{Float64}
    Y₀::Array{Float64}
    X₁::Array{Float64}
    Y₁::Array{Float64}
    @model_config individual_effect

    function InterruptedTimeSeries(
        X₀::Array{<:Real},
        Y₀::Array{<:Real},
        X₁::Array{<:Real},
        Y₁::Array{<:Real};
        regularized=true,
        activation=relu,
        validation_metric=mse,
        min_neurons=1,
        max_neurons=100,
        folds=5,
        iterations=round(size(X₀, 1) / 10),
        approximator_neurons=round(size(X₀, 1) / 10),
        autoregression=true,
    )

        # Add autoregressive term
        X₀ = ifelse(autoregression == true, reduce(hcat, (X₀, moving_average(Y₀))), X₀)
        X₁ = ifelse(autoregression == true, reduce(hcat, (X₁, moving_average(Y₁))), X₁)

        return new(
            X₀,
            Float64.(Y₀),
            Float64.(X₁),
            Float64.(Y₁),
            "difference",
            true,
            "regression",
            regularized,
            activation,
            validation_metric,
            min_neurons,
            max_neurons,
            folds,
            iterations,
            approximator_neurons,
            0,
            fill(NaN, size(Y₁, 1)),
        )
    end
end

function InterruptedTimeSeries(
    X₀,
    Y₀,
    X₁,
    Y₁;
    regularized=true,
    activation=relu,
    validation_metric=mse,
    min_neurons=1,
    max_neurons=100,
    folds=5,
    iterations=round(size(X₀, 1) / 10),
    approximator_neurons=round(size(X₀, 1) / 10),
    autoregression=true,
)

    # Convert to arrays
    X₀, X₁, Y₀, Y₁ = Matrix{Float64}(X₀), Matrix{Float64}(X₁), Y₀[:, 1], Y₁[:, 1]

    return InterruptedTimeSeries(
        X₀,
        Y₀,
        X₁,
        Y₁;
        regularized=regularized,
        activation=activation,
        validation_metric=validation_metric,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        folds=folds,
        iterations=iterations,
        approximator_neurons=approximator_neurons,
        autoregression=autoregression,
    )
end

"""
    GComputation(X, T, Y; kwargs...)

Initialize a G-Computation estimator.

# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `task::String`: either regression or classification.
- `quantity_of_interest::String`: ATE for average treatment effect or CTE for cummulative 
    treatment effect.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of cross validation folds to find the best number of neurons.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.

# Notes
If regularized is set to true then the ridge penalty will be estimated using generalized 
cross validation where the maximum number of iterations is 2 * folds for the successive 
halving procedure. However, if the penalty in on iteration is approximately the same as in 
the previous penalty, then the procedure will stop early.

# References
For a good overview of G-Computation see:
    Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
    Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
    Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
    estimator for causal inference with different covariates sets: a comparative simulation 
    study." Scientific reports 10, no. 1 (2020): 9219.


For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
m1 = GComputation(X, T, Y)
m2 = GComputation(X, T, Y; task="regression")
m3 = GComputation(X, T, Y; task="regression", quantity_of_interest="ATE)
m4 = GComputation(X, T, Y; task="regression", quantity_of_interest="ATE", regularized=true)

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100)) 
m5 = GComputation(x_df, t_df, y_df)
```
"""
mutable struct GComputation <: CausalEstimator
    @standard_input_data
    @model_config average_effect
    learner::ExtremeLearningMachine

    function GComputation(
        X::Array{<:Real},
        T::Array{<:Real},
        Y::Array{<:Real};
        task="regression",
        quantity_of_interest="ATE",
        regularized=true,
        activation=relu,
        temporal=true,
        validation_metric=mse,
        min_neurons=1,
        max_neurons=100,
        folds=5,
        iterations=round(size(X, 1) / 10),
        approximator_neurons=round(size(X, 1) / 10),
    )
        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        elseif quantity_of_interest ∉ ("ATE", "ITT", "ATT")
            throw(ArgumentError("quantity_of_interest must be ATE, ITT, or ATT"))
        end

        return new(
            Float64.(X),
            Float64.(T),
            Float64.(Y),
            quantity_of_interest,
            temporal,
            task,
            regularized,
            activation,
            validation_metric,
            min_neurons,
            max_neurons,
            folds,
            iterations,
            approximator_neurons,
            0,
            NaN,
        )
    end
end

function GComputation(
    X,
    T,
    Y;
    task="regression",
    quantity_of_interest="ATE",
    regularized=true,
    activation=relu,
    temporal=true,
    validation_metric=mse,
    min_neurons=1,
    max_neurons=100,
    folds=5,
    iterations=round(size(X, 1) / 10),
    approximator_neurons=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    return GComputation(
        X,
        T,
        Y;
        task=task,
        quantity_of_interest=quantity_of_interest,
        regularized=regularized,
        activation=activation,
        temporal=temporal,
        validation_metric=validation_metric,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        folds=folds,
        iterations=iterations,
        approximator_neurons=approximator_neurons,
    )
end

"""
    DoubleMachineLearning(X, T, Y; kwargs...)

Initialize a double machine learning estimator with cross fitting.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `W::Any`: an array or dataframe of all possible confounders.
- `task::String`: either regression or classification.
- `quantity_of_interest::String`: ATE for average treatment effect or CTE for cummulative 
    treatment effect.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `folds::Int`: the number of cross validation folds to find the best number of neurons.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Int`: the number of nuerons in the validation loss approximator 
    network.

# Notes
If regularized is set to true then the ridge penalty will be estimated using generalized 
cross validation where the maximum number of iterations is 2 * folds for the successive 
halving procedure. However, if the penalty in on iteration is approximately the same as in 
the previous penalty, then the procedure will stop early.

Unlike other estimators, this method does not support time series or panel data. This method 
also does not work as well with smaller datasets because it estimates separate outcome 
models for the treatment and control groups.

# References
For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2016): C1-C68.


For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = DoubleMachineLearning(X, T, Y)
m2 = DoubleMachineLearning(X, T, Y; task="regression")

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
m3 = DoubleMachineLearning(x_df, t_df, y_df)
```
"""
mutable struct DoubleMachineLearning <: CausalEstimator
    @double_learner_input_data
    @model_config average_effect

    function DoubleMachineLearning(
        X::Array{<:Real},
        T::Array{<:Real},
        Y::Array{<:Real};
        W=X,
        task="regression",
        regularized=true,
        activation=relu,
        validation_metric=mse,
        min_neurons=1,
        max_neurons=100,
        folds=5,
        iterations=round(size(X, 1) / 10),
        approximator_neurons=round(size(X, 1) / 10),
    )
        return new(
            Float64.(X),
            Float64.(T),
            Float64.(Y),
            Float64.(W),
            "ATE",
            false,
            task,
            regularized,
            activation,
            validation_metric,
            min_neurons,
            max_neurons,
            folds,
            iterations,
            approximator_neurons,
            0,
            NaN,
        )
    end
end

function DoubleMachineLearning(
    X,
    T,
    Y;
    W=X,
    regularized=true,
    activation=relu,
    validation_metric=mse,
    min_neurons=1,
    max_neurons=100,
    folds=5,
    iterations=round(size(X, 1) / 10),
    approximator_neurons=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y, W = Matrix{Float64}(X), T[:, 1], Y[:, 1], Matrix{Float64}(W)

    return DoubleMachineLearning(
        X,
        T,
        Y;
        W=W,
        regularized=regularized,
        activation=activation,
        validation_metric=validation_metric,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        folds=folds,
        iterations=iterations,
        approximator_neurons=approximator_neurons,
    )
end

"""
    estimate_causal_effect!(its)

Estimate the effect of an event relative to a predicted counterfactual.

# Examples
```julia
X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(its::InterruptedTimeSeries)
    # We will not find the best number of neurons after we have already estimated the causal
    # effect and are getting p-values, confidence intervals, or standard errors. We will use
    # the same number that was found when calling this method.
    if its.num_neurons === 0
        its.num_neurons = best_size(
            its.X₀,
            its.Y₀,
            its.validation_metric,
            "regression",
            its.activation,
            its.min_neurons,
            its.max_neurons,
            its.regularized,
            its.folds,
            true,
            its.iterations,
            its.approximator_neurons,
        )
    end

    if its.regularized
        learner = RegularizedExtremeLearner(its.X₀, its.Y₀, its.num_neurons, its.activation)
    else
        learner = ExtremeLearner(its.X₀, its.Y₀, its.num_neurons, its.activation)
    end

    fit!(learner)
    its.causal_effect = predict_counterfactual!(learner, its.X₁) - its.Y₁

    return its.causal_effect
end

function estimate_causal_effect!(g::GComputation)
    g.causal_effect = mean(g_formula!(g))
    return g.causal_effect
end

"""
    estimate_causal_effect!(g)

Estimate a causal effect of interest using G-Computation.

# Notes
If treatents are administered at multiple time periods, the effect will be estimated as the 
average difference between the outcome of being treated in all periods and being treated in 
no periods. For example, given that ividuals 1, 2, ..., i ∈ I recieved either a treatment 
or a placebo in p different periods, the model would estimate the average treatment effect 
as E[Yᵢ|T₁=1, T₂=1, ... Tₚ=1, Xₚ] - E[Yᵢ|T₁=0, T₂=0, ... Tₚ=0, Xₚ].

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = GComputation(X, T, Y)
g_formula!(m1)
```
"""
function g_formula!(g)
    covariates, y = hcat(g.X, g.T), var_type(g.Y)

    if g.quantity_of_interest ∈ ("ITT", "ATE")
        Xₜ, Xᵤ = hcat(g.X, ones(size(g.T, 1))), hcat(g.X, zeros(size(g.T, 1)))
    else
        Xₜ, Xᵤ = hcat(g.X, g.T), hcat(g.X, zeros(size(g.Y, 1)))
    end

    if g.num_neurons === 0  # Don't search for the best number of neurons multiple times
        g.num_neurons = best_size(
            Array(covariates),
            g.Y,
            g.validation_metric,
            g.task,
            g.activation,
            g.min_neurons,
            g.max_neurons,
            g.regularized,
            g.folds,
            g.temporal,
            g.iterations,
            g.approximator_neurons,
        )
    end

    if g.regularized
        g.learner = RegularizedExtremeLearner(covariates, g.Y, g.num_neurons, g.activation)
    else
        g.learner = ExtremeLearner(covariates, g.Y, g.num_neurons, g.activation)
    end

    fit!(g.learner)
    yₜ = clip_if_binary(predict(g.learner, Xₜ), y)
    yᵤ = clip_if_binary(predict(g.learner, Xᵤ), y)
    return vec(yₜ) - vec(yᵤ)
end

"""
    estimate_causal_effect!(DML)

Estimate a causal effect of interest using double machine learning.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = DoubleMachineLearning(X, T, Y)
estimate_causal_effect!(m1)

W = rand(100, 6)
m2 = DoubleMachineLearning(X, T, Y, W=W)
estimate_causal_effect!(m2)
```
"""
function estimate_causal_effect!(DML::DoubleMachineLearning)
    # Uses the same number of neurons for all phases of estimation
    if DML.num_neurons === 0
        task = var_type(DML.Y) == Binary() ? "classification" : "regression"
        DML.num_neurons = best_size(
            DML.X,
            DML.Y,
            DML.validation_metric,
            task,
            DML.activation,
            DML.min_neurons,
            DML.max_neurons,
            DML.regularized,
            DML.folds,
            false,
            DML.iterations,
            DML.approximator_neurons,
        )
    end

    causal_loss!(DML)
    DML.causal_effect /= DML.folds

    return DML.causal_effect
end

"""
    causal_loss!(D, [,cate])

Estimate a treatment effect using double machine learning.

# Notes
This method should not be called directly.

# Arguments
- `DML::DoubleMachineLearning`: the DoubleMachineLearning struct to estimate the effect for.
- `cate::Bool=false`: whether to estimate the cate.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = DoubleMachineLearning(X, T, Y)
causal_loss!(m1)
```
"""
function causal_loss!(DML)
    X, T, W, Y = make_folds(DML)
    DML.causal_effect = 0

    # Cross fitting by training on the main folds and predicting residuals on the auxillary
    for fld in 1:(DML.folds)
        X_train, X_test = reduce(vcat, X[1:end .!== fld]), X[fld]
        Y_train, Y_test = reduce(vcat, Y[1:end .!== fld]), Y[fld]
        T_train, T_test = reduce(vcat, T[1:end .!== fld]), T[fld]
        W_train, W_test = reduce(vcat, W[1:end .!== fld]), W[fld]

        Ỹ, T̃ = predict_residuals(
            DML, X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test
        )
        DML.causal_effect += (vec(sum(T̃ .* X_test; dims=2)) \ Ỹ)[1]
    end
end

"""
    predict_residuals(D, x_train, x_test, y_train, y_test, t_train, t_test)

Predict treatment and outcome residuals for doubl machine learning.

# Notes
This method should not be called directly.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
x_train, x_test = X[1:80, :], X[81:end, :]
y_train, y_test = Y[1:80], Y[81:end]
t_train, t_test = T[1:80], T[81:100]
m1 = DoubleMachineLearning(X, T, Y)
predict_residuals(m1, x_train, x_test, y_train, y_test, t_train, t_test)
```
"""
function predict_residuals(
    D,
    x_train,
    x_test,
    y_train,
    y_test,
    t_train,
    t_test,
    w_train,
    w_test,
)
    V = x_train != w_train && x_test != w_test ? reduce(hcat, (x_train, w_train)) : x_train
    V_test = V == x_train ? x_test : reduce(hcat, (x_test, w_test))

    if D.regularized
        y = RegularizedExtremeLearner(V, y_train, D.num_neurons, D.activation)
        t = RegularizedExtremeLearner(V, t_train, D.num_neurons, D.activation)
    else
        y = ExtremeLearner(V, y_train, D.num_neurons, D.activation)
        t = ExtremeLearner(V, t_train, D.num_neurons, D.activation)
    end

    fit!(y)
    fit!(t)
    y_pred = clip_if_binary(predict(y, V_test), var_type(D.Y))
    t_pred = clip_if_binary(predict(t, V_test), var_type(D.T))
    ỹ, t̃ = y_test - y_pred, t_test - t_pred

    return ỹ, t̃
end

"""
    make_folds(D)

Make folds for cross fitting for a double machine learning estimator.

# Notes
This method should not be called directly.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = DoubleMachineLearning(X, T, Y)
make_folds(m1)
```
"""
function make_folds(D)
    X_T_W, Y = generate_folds(reduce(hcat, (D.X, D.T, D.W)), D.Y, D.folds)
    X = [fl[:, 1:size(D.X, 2)] for fl in X_T_W]
    T = [fl[:, size(D.X, 2) + 1] for fl in X_T_W]
    W = [fl[:, (size(D.X, 2) + 2):end] for fl in X_T_W]

    return X, T, W, Y
end

"""
    moving_average(x)

Calculates a cumulative moving average.

# Examples
```julia
moving_average([1, 2, 3])
```
"""
function moving_average(x)
    result = similar(x)
    for i in 1:length(x)
        result[i] = mean(x[1:i])
    end
    return result
end
