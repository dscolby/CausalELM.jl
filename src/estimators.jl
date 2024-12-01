"""Abstract type for GComputation and DoubleMachineLearning"""
abstract type CausalEstimator end

"""
    InterruptedTimeSeries(X₀, Y₀, X₁, Y₁; kwargs...)

Initialize an interrupted time series estimator. 

# Arguments
- `X₀::Any`: array or DataFrame of covariates from the pre-treatment period.
- `Y₁::Any`: array or DataFrame of outcomes from the pre-treatment period.
- `X₁::Any`: array or DataFrame of covariates from the post-treatment period.
- `Y₁::Any`: array or DataFrame of outcomes from the post-treatment period.

# Keywords
- `activation::Function=swish`: activation function to use.
- `sample_size::Integer=size(X₀, 1)`: number of bootstrapped samples for the extreme 
    learner.
- `num_machines::Integer=50`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X₀, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For a simple linear regression-based tutorial on interrupted time series analysis see:
    Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
    regression for the evaluation of public health interventions: a tutorial." International 
    journal of epidemiology 46, no. 1 (2017): 348-355.

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
    marginal_effect::Float64
    @model_config individual_effect
end

function InterruptedTimeSeries(
    X₀,
    Y₀,
    X₁,
    Y₁;
    activation::Function=swish,
    sample_size::Integer=size(X₀, 1),
    num_machines::Integer=50,
    num_feats::Integer=Int(round(0.75 * size(X₀, 2))),
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
        NaN,
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
- `X::Any`: array or DataFrame of covariates.
- `T::Any`: vector or DataFrame of treatment statuses.
- `Y::Any`: array or DataFrame of outcomes.

# Keywords
- `quantity_of_interest::String`: ATE for average treatment effect or ATT for average 
    treatment effect on the treated.
- `activation::Function=swish`: activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for the extreme 
    learners.
- `num_machines::Integer=50`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For a good overview of G-Computation see:
    Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
    Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
    Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
    estimator for causal inference with different covariates sets: a comparative simulation 
    study." Scientific reports 10, no. 1 (2020): 9219.

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
    marginal_effect::Float64
    ensemble::ELMEnsemble

    function GComputation(
        X,
        T,
        Y;
        quantity_of_interest::String="ATE",
        activation::Function=swish,
        sample_size::Integer=size(X, 1),
        num_machines::Integer=50,
        num_feats::Integer=Int(round(0.75 * size(X, 2))),
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
            NaN,
        )
    end
end

"""
    DoubleMachineLearning(X, T, Y; kwargs...)

Initialize a double machine learning estimator with cross fitting.

# Arguments
- `X::Any`: array or DataFrame of covariates of interest.
- `T::Any`: vector or DataFrame of treatment statuses.
- `Y::Any`: array or DataFrame of outcomes.

# Keywords
- `activation::Function=swish`: activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for teh extreme 
    learners.
- `num_machines::Integer=50`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75, * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.
- `folds::Integer`: number of folds to use for cross fitting.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2016): C1-C68.

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
    @standard_input_data
    @model_config average_effect
    marginal_effect::Float64
    folds::Integer
end

function DoubleMachineLearning(
    X,
    T,
    Y;
    activation::Function=swish,
    sample_size::Integer=size(X, 1),
    num_machines::Integer=50,
    num_feats::Integer=Int(round(0.75 * size(X, 2))),
    num_neurons::Integer=round(Int, log10(size(X, 1)) * num_feats),
    folds::Integer=5,
)
    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    # Shuffle data with random indices
    indices = shuffle(1:length(Y))
    X, T, Y = X[indices, :], T[indices], Y[indices]

    task = var_type(Y) isa Binary ? "classification" : "regression"

    return DoubleMachineLearning(
        X,
        float(T),
        float(Y),
        "ATE",
        false,
        task,
        activation,
        sample_size, 
        num_machines, 
        num_feats,
        num_neurons,
        NaN,
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
@inline function estimate_causal_effect!(its::InterruptedTimeSeries)
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
    its.causal_effect = predict(learner, its.X₁) .- its.Y₁
    its.marginal_effect = mean(its.causal_effect)

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
@inline function estimate_causal_effect!(g::GComputation)
    causal_effect, marginal_effect = g_formula!(g)
    g.causal_effect, g.marginal_effect = mean(causal_effect), mean(marginal_effect)

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
@inline function g_formula!(g)  # Keeping this separate for reuse with S-Learning
    vars, y = hcat(g.X, g.T), g.Y
    x₁, x₀ = hcat(g.X, ones(size(g.X, 1))), hcat(g.X, zeros(size(g.X, 1)))

    if g.quantity_of_interest ∈ ("ITT", "ATE", "CATE")
        Xₜ = hcat(vars[:, 1:(end - 1)], ones(size(vars, 1)))
        Xᵤ = hcat(vars[:, 1:(end - 1)], zeros(size(vars, 1)))
    else
        Xₜ = hcat(vars[g.T .== 1, 1:(end - 1)], ones(size(g.T[g.T .== 1], 1)))
        Xᵤ = hcat(vars[g.T .== 1, 1:(end - 1)], zeros(size(g.T[g.T .== 1], 1)))
    end

    g.ensemble = ELMEnsemble(
        vars, y, g.sample_size, g.num_machines, g.num_feats, g.num_neurons, g.activation
    )

    fit!(g.ensemble)
    yₜ, yᵤ = predict(g.ensemble, Xₜ), predict(g.ensemble, Xᵤ)

    return vec(yₜ) - vec(yᵤ), predict(g.ensemble, x₁) - predict(g.ensemble, x₀)
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
@inline function estimate_causal_effect!(DML::DoubleMachineLearning)
    X, T, Y = generate_folds(DML.X, DML.T, DML.Y, DML.folds)
    DML.causal_effect, DML.marginal_effect, Δ = 0, 0, 1.5e-8mean(DML.T)

    # Cross fitting by training on the main folds and predicting residuals on the auxillary
    for fold in 1:(DML.folds)
        X_train, X_test = reduce(vcat, X[1:end .!== fold]), X[fold]
        Y_train, Y_test = reduce(vcat, Y[1:end .!== fold]), Y[fold]
        T_train, T_test = reduce(vcat, T[1:end .!== fold]), T[fold]

        Ỹ, T̃, T̃₊, T̃₋ = predict_residuals(
            DML, X_train, X_test, Y_train, Y_test, T_train, T_test, Δ
        )

        DML.causal_effect += T̃\Ỹ
        DML.marginal_effect += (T̃₊\Ỹ - T̃₋\Ỹ) / 2Δ
    end
    
    DML.causal_effect /= DML.folds
    DML.marginal_effect /= DML.folds

    return DML.causal_effect
end

"""
    predict_residuals(D, x_train, x_test, y_train, y_test, t_train, t_test, Δ)

Predict treatment, outcome, and marginal effect residuals for double machine learning or 
R-learning.

# Notes
This method should not be called directly.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> x_train, x_test = X[1:80, :], X[81:end, :]
julia> y_train, y_test = Y[1:80], Y[81:end]
julia> t_train, t_test = T[1:80], T[81:100]
julia> m1 = DoubleMachineLearning(X, T, Y)
julia> predict_residuals(m1, x_tr, x_te, y_tr, y_te, t_tr, t_te, zeros(100), 1e-5)
```
"""
@inline function predict_residuals(
    D, 
    xₜᵣ::Array{Float64}, 
    xₜₑ::Array{Float64}, 
    yₜᵣ::Vector{Float64}, 
    yₜₑ::Vector{Float64}, 
    tₜᵣ::Vector{Float64}, 
    tₜₑ::Vector{Float64},
    Δ::Float64
)
    args = D.sample_size, D.num_machines, D.num_feats, D.num_neurons, D.activation
    y = ELMEnsemble(xₜᵣ, yₜᵣ, args...)
    t = ELMEnsemble(xₜᵣ, tₜᵣ, args...)

    fit!(y); fit!(t)
    yₚᵣ, tₚᵣ = predict(y, xₜₑ), predict(t, xₜₑ)
    tₜₑ₊ = var_type(tₜₑ) isa Binary ? ones(size(tₜₑ)) : tₜₑ .+ Δ
    tₜₑ₋ = var_type(tₜₑ) isa Binary ? ones(size(tₜₑ)) : tₜₑ .- Δ

    return yₜₑ - yₚᵣ, tₜₑ - tₚᵣ, tₜₑ₊ - tₚᵣ, tₜₑ₋ - tₚᵣ
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
