"""Abstract type for GComputation and DoubleMachineLearning"""
abstract type CausalEstimator end

"""
    InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; kwargs...)

Initialize an interrupted time series estimator. 

# Arguments
- `X₀::Any`: an array or DataFrame of covariates from the pre-treatment period.
- `Y₁::Any`: an array or DataFrame of outcomes from the pre-treatment period.
- `X₁::Any`: an array or DataFrame of covariates from the post-treatment period.
- `Y₁::Any`: an array or DataFrame of outcomes from the post-treatment period.

# Keywords
- `activation::Function=relu`: activation function to use.
- `sample_size::Integer=size(X₀, 1)`: number of bootstrapped samples for the extreme 
    learner.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(sqrt(size(X₀, 2))))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce computational complexity and overfitting, the model used to estimate the 
counterfactual is a bagged ensemble extreme learning machines. To further reduce the 
computational complexity you can reduce sample_size, num_machines, or num_neurons.

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
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> m2 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; regularized=false)
julia> x₀_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100))
julia> y₀_df = DataFrame(y=rand(100))
julia> x₁_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100)) 
julia> y₁_df = DataFrame(y=rand(100))
julia> m3 = InterruptedTimeSeries(x₀_df, y₀_df, x₁_df, y₁_df)
```
"""
mutable struct InterruptedTimeSeries
    X₀::Array{Float64}
    Y₀::Array{Float64}
    X₁::Array{Float64}
    Y₁::Array{Float64}
    @model_config individual_effect
end

function InterruptedTimeSeries(
    X₀,
    Y₀,
    X₁,
    Y₁;
    activation::Function=relu,
    sample_size::Integer=size(X₀, 1),
    num_machines::Integer=100,
    num_feats::Integer=Int(round(sqrt(size(X₀, 2)))),
    num_neurons::Integer=round(Int, log10(size(X₀, 1)) * size(X₀, 2)),
    autoregression::Bool=true,
)
    # Convert to arrays
    X₀, X₁, Y₀, Y₁ = Matrix{Float64}(X₀), Matrix{Float64}(X₁), Y₀[:, 1], Y₁[:, 1]

    # Add autoregressive term
    X₀ = ifelse(autoregression == true, reduce(hcat, (X₀, moving_average(Y₀))), X₀)
    X₁ = ifelse(autoregression == true, reduce(hcat, (X₁, moving_average(Y₁))), X₁)

    task = var_type(Y₀) isa Binary ? "classification" : "regression"

    return InterruptedTimeSeries(
        X₀,
        float(Y₀),
        X₁,
        float(Y₁),
        "difference",
        true,
        task,
        activation,
        sample_size,
        num_machines,
        num_feats,
        num_neurons,
        fill(NaN, size(Y₁, 1)),
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
- `quantity_of_interest::String`: ATE for average treatment effect or ATT for average 
    treatment effect on the treated.
- `activation::Function=relu`: activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for the extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(sqrt(size(X, 2))))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce computational complexity and overfitting, the model used to estimate the 
counterfactual is a bagged ensemble extreme learning machines. To further reduce the 
computational complexity you can reduce sample_size, num_machines, or num_neurons.

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
julia> X, T, Y =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = GComputation(X, T, Y)
julia> m2 = GComputation(X, T, Y; task="regression")
julia> m3 = GComputation(X, T, Y; task="regression", quantity_of_interest="ATE)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100)) 
julia> m5 = GComputation(x_df, t_df, y_df)
```
"""
mutable struct GComputation <: CausalEstimator
    @standard_input_data
    @model_config average_effect
    ensemble::ELMEnsemble

    function GComputation(
        X,
        T,
        Y;
        quantity_of_interest::String="ATE",
        activation::Function=relu,
        sample_size::Integer=size(X, 1),
        num_machines::Integer=100,
        num_feats::Integer=Int(round(sqrt(size(X, 2)))),
        num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
        temporal::Bool=true,
    )
        if quantity_of_interest ∉ ("ATE", "ITT", "ATT")
            throw(ArgumentError("quantity_of_interest must be ATE, ITT, or ATT"))
        end

        # Convert to arrays
        X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

        task = var_type(Y) isa Binary ? "classification" : "regression"

        return new(
            X,
            float(T),
            float(Y),
            quantity_of_interest,
            temporal,
            task,
            activation,
            sample_size,
            num_machines,
            num_feats,
            num_neurons,
            NaN,
        )
    end
end

"""
    DoubleMachineLearning(X, T, Y; kwargs...)

Initialize a double machine learning estimator with cross fitting.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `W::Any`: array or dataframe of all possible confounders.
- `activation::Function=relu`: activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for teh extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(sqrt(size(X, 2))))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.
- `folds::Integer`: number of folds to use for cross fitting.

# Notes
To reduce computational complexity and overfitting, the model used to estimate the 
counterfactual is a bagged ensemble extreme learning machines. To further reduce the 
computational complexity you can reduce sample_size, num_machines, or num_neurons.

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
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m2 = DoubleMachineLearning(x_df, t_df, y_df)
```
"""
mutable struct DoubleMachineLearning <: CausalEstimator
    @double_learner_input_data
    @model_config average_effect
    folds::Integer
end

function DoubleMachineLearning(
    X,
    T,
    Y;
    W=X,
    activation::Function=relu,
    sample_size::Integer=size(X, 1),
    num_machines::Integer=100,
    num_feats::Integer=Int(round(sqrt(size(X, 2)))),
    num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
    folds::Integer=5,
)
    # Convert to arrays
    X, T, Y, W = Matrix{Float64}(X), T[:, 1], Y[:, 1], Matrix{Float64}(W)

    task = var_type(Y) isa Binary ? "classification" : "regression"

    return DoubleMachineLearning(
        X,
        float(T),
        float(Y),
        W,
        "ATE",
        false,
        task,
        activation,
        sample_size, 
        num_machines, 
        num_feats,
        num_neurons,
        NaN,
        folds,
    )
end

"""
    estimate_causal_effect!(its)

Estimate the effect of an event relative to a predicted counterfactual.

# Examples
```julia
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(its::InterruptedTimeSeries)
    learner = ELMEnsemble(
        its.X₀, 
        its.Y₀, 
        its.sample_size, 
        its.num_machines,
        its.num_feats, 
        its.num_neurons, 
        its.activation
    )

    fit!(learner)
    its.causal_effect = predict_mean(learner, its.X₁) - its.Y₁

    return its.causal_effect
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
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = GComputation(X, T, Y)
julia> estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(g::GComputation)
    g.causal_effect = mean(g_formula!(g))
    return g.causal_effect
end

"""
    g_formula!(g)

Compute the G-formula for G-computation and S-learning.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = GComputation(X, T, Y)
julia> g_formula!(m1)

julia> m2 = SLearner(X, T, Y)
julia> g_formula!(m2)
```
"""
function g_formula!(g)  # Keeping this separate enables it to be reused for S-Learning
    covariates, y = hcat(g.X, g.T), g.Y

    if g.quantity_of_interest ∈ ("ITT", "ATE", "CATE")
        Xₜ = hcat(covariates[:, 1:(end - 1)], ones(size(covariates, 1)))
        Xᵤ = hcat(covariates[:, 1:(end - 1)], zeros(size(covariates, 1)))
    else
        Xₜ = hcat(covariates[g.T .== 1, 1:(end - 1)], ones(size(g.T[g.T .== 1], 1)))
        Xᵤ = hcat(covariates[g.T .== 1, 1:(end - 1)], zeros(size(g.T[g.T .== 1], 1)))
    end

    g.ensemble = ELMEnsemble(
        covariates, 
        y, 
        g.sample_size, 
        g.num_machines, 
        g.num_feats,
        g.num_neurons, 
        g.activation
    )

    fit!(g.ensemble)
    
    yₜ, yᵤ = predict_mean(g.ensemble, Xₜ), predict_mean(g.ensemble, Xᵤ)

    return vec(yₜ) - vec(yᵤ)
end

"""
    estimate_causal_effect!(DML)

Estimate a causal effect of interest using double machine learning.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> estimate_causal_effect!(m1)

julia> W = rand(100, 6)
julia> m2 = DoubleMachineLearning(X, T, Y, W=W)
julia> estimate_causal_effect!(m2)
```
"""
function estimate_causal_effect!(DML::DoubleMachineLearning)
    causal_loss!(DML)
    DML.causal_effect /= DML.folds

    return DML.causal_effect
end

"""
    causal_loss!(DML, [,cate])

Minimize the causal loss function for double machine learning.

# Notes
This method should not be called directly.

# Arguments
- `DML::DoubleMachineLearning`: the DoubleMachineLearning struct to estimate the effect for.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> causal_loss!(m1)
```
"""
function causal_loss!(DML::DoubleMachineLearning)
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

Predict treatment and outcome residuals for double machine learning or R-learning.

# Notes
This method should not be called directly.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> x_train, x_test = X[1:80, :], X[81:end, :]
julia> y_train, y_test = Y[1:80], Y[81:end]
julia> t_train, t_test = T[1:80], T[81:100]
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> predict_residuals(m1, x_train, x_test, y_train, y_test, t_train, t_test)
```
"""
function predict_residuals(
    D, 
    x_train::Array{Float64}, 
    x_test::Array{Float64}, 
    y_train::Vector{Float64}, 
    y_test::Vector{Float64}, 
    t_train::Vector{Float64}, 
    t_test::Vector{Float64}, 
    w_train::Array{Float64}, 
    w_test::Array{Float64},
)
    V = x_train != w_train && x_test != w_test ? reduce(hcat, (x_train, w_train)) : x_train
    V_test = V == x_train ? x_test : reduce(hcat, (x_test, w_test))

    y = ELMEnsemble(
        V, y_train, D.sample_size, D.num_machines, D.num_feats, D.num_neurons, D.activation
    )
    t = ELMEnsemble(
        V, t_train, D.sample_size, D.num_machines, D.num_feats, D.num_neurons, D.activation
    )

    fit!(y)
    fit!(t)

    y_pred, t_pred = predict_mean(y, V_test), predict_mean(t, V_test)
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
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> make_folds(m1)
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
    generate_folds(X, Y, folds)

Create folds for cross validation.

# Examples
```jldoctest
julia> xfolds, y_folds = CausalELM.generate_folds(zeros(4, 2), zeros(4), 2)
([[0.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 0.0]], [[0.0], [0.0, 0.0, 0.0]])
```
"""
function generate_folds(X, Y, folds)
    msg = """the number of folds must be less than the number of observations"""
    n = length(Y)

    if folds >= n
        throw(ArgumentError(msg))
    end

    fold_setx = Array{Array{Float64,2}}(undef, folds)
    fold_sety = Array{Array{Float64,1}}(undef, folds)

    # Indices to start and stop for each fold
    stops = round.(Int, range(; start=1, stop=n, length=folds + 1))

    # Indices to use for making folds
    indices = [s:(e - (e < n) * 1) for (s, e) in zip(stops[1:(end - 1)], stops[2:end])]

    for (i, idx) in enumerate(indices)
        fold_setx[i], fold_sety[i] = X[idx, :], Y[idx]
    end

    return fold_setx, fold_sety
end

"""
    moving_average(x)

Calculates a cumulative moving average.

# Examples
```julia
julia> moving_average([1, 2, 3])
```
"""
function moving_average(x)
    result = similar(x)
    for i in 1:length(x)
        result[i] = mean(x[1:i])
    end
    return result
end
