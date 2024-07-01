using LinearAlgebra: Diagonal

"""Abstract type for metalearners"""
abstract type Metalearner end

"""
    SLearner(X, T, Y; kwargs...)

Initialize a S-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `activation::Function=relu`: the activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for eth extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For an overview of S-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of 
    the national academy of sciences 116, no. 10 (2019): 4156-4165.

# Examples
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
    @standard_input_data
    @model_config individual_effect
    ensemble::ELMEnsemble

    function SLearner(
        X,
        T,
        Y;
        activation::Function=relu,
        sample_size::Integer=size(X, 1),
        num_machines::Integer=100,
        num_feats::Integer=Int(round(0.75 * size(X, 2))),
        num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
    )

        # Convert to arrays
        X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

        task = var_type(Y) isa Binary ? "classification" : "regression"

        return new(
            X,
            Float64.(T),
            Float64.(Y),
            "CATE",
            false,
            task,
            activation,
            sample_size,
            num_machines,
            num_feats,
            num_neurons,
            fill(NaN, size(T, 1)),
        )
    end
end

"""
    TLearner(X, T, Y; kwargs...)

Initialize a T-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `activation::Function=relu`: the activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for eth extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For an overview of T-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of 
    the national academy of sciences 116, no. 10 (2019): 4156-4165.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = TLearner(X, T, Y)
julia> m2 = TLearner(X, T, Y; regularized=false)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m3 = TLearner(x_df, t_df, y_df)
```
"""
mutable struct TLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    μ₀::ELMEnsemble
    μ₁::ELMEnsemble

    function TLearner(
        X,
        T,
        Y;
        activation::Function=relu,
        sample_size::Integer=size(X, 1),
        num_machines::Integer=100,
        num_feats::Integer=Int(round(0.75 * size(X, 2))),
        num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
    )
        # Convert to arrays
        X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

        task = var_type(Y) isa Binary ? "classification" : "regression"

        return new(
            X,
            Float64.(T),
            Float64.(Y),
            "CATE",
            false,
            task,
            activation,
            sample_size,
            num_machines,
            num_feats,
            num_neurons,
            fill(NaN, size(T, 1)),
        )
    end
end

"""
    XLearner(X, T, Y; kwargs...)

Initialize an X-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `activation::Function=relu`: the activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for eth extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For an overview of X-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> m2 = XLearner(X, T, Y; regularized=false)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m3 = XLearner(x_df, t_df, y_df)
```
"""
mutable struct XLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    μ₀::ELMEnsemble
    μ₁::ELMEnsemble
    ps::Array{Float64}

    function XLearner(
        X,
        T,
        Y;
        activation::Function=relu,
        sample_size::Integer=size(X, 1),
        num_machines::Integer=100,
        num_feats::Integer=Int(round(0.75 * size(X, 2))),
        num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
    )
        # Convert to arrays
        X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

        task = var_type(Y) isa Binary ? "classification" : "regression"

        return new(
            X,
            Float64.(T),
            Float64.(Y),
            "CATE",
            false,
            task,
            activation,
            sample_size,
            num_machines,
            num_feats,
            num_neurons,
            fill(NaN, size(T, 1)),
        )
    end
end

"""
    RLearner(X, T, Y; kwargs...)

Initialize an R-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `activation::Function=relu`: the activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for eth extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

## References
For an explanation of R-Learner estimation see:
    Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
    effects." Biometrika 108, no. 2 (2021): 299-319.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = RLearner(X, T, Y)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m2 = RLearner(x_df, t_df, y_df)
```
"""
mutable struct RLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    folds::Integer
end

function RLearner(
    X,
    T,
    Y;
    activation::Function=relu,
    sample_size::Integer=size(X, 1),
    num_machines::Integer=100,
    num_feats::Integer=Int(round(0.75 * size(X, 2))),
    num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
    folds::Integer=5,
)

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    task = var_type(Y) isa Binary ? "classification" : "regression"

    return RLearner(
        X,
        Float64.(T),
        Float64.(Y),
        "CATE",
        false,
        task,
        activation,
        sample_size,
        num_machines,
        num_feats,
        num_neurons,
        fill(NaN, size(T, 1)),
        folds,
    )
end

"""
    DoublyRobustLearner(X, T, Y; kwargs...)

Initialize a doubly robust CATE estimator.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `activation::Function=relu`: the activation function to use.
- `sample_size::Integer=size(X, 1)`: number of bootstrapped samples for eth extreme 
    learners.
- `num_machines::Integer=100`: number of extreme learning machines for the ensemble.
- `num_feats::Integer=Int(round(0.75 * size(X, 2)))`: number of features to bootstrap for 
    each learner in the ensemble.
- `num_neurons::Integer`: number of neurons to use in the extreme learning machines.

# Notes
To reduce the computational complexity you can reduce sample_size, num_machines, or 
num_neurons.

# References
For an explanation of doubly robust cate estimation see:
    Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
    effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoublyRobustLearner(X, T, Y)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m2 = DoublyRobustLearner(x_df, t_df, y_df)

julia> w = rand(100, 6)
julia> m3 = DoublyRobustLearner(X, T, Y, W=w)
```
"""
mutable struct DoublyRobustLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    folds::Integer
end

function DoublyRobustLearner(
    X,
    T,
    Y;
    activation::Function=relu,
    sample_size::Integer=size(X, 1),
    num_machines::Integer=100,
    num_feats::Integer=Int(round(0.75 * size(X, 2))),
    num_neurons::Integer=round(Int, log10(size(X, 1)) * size(X, 2)),
    folds::Integer=5,
)
    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    task = var_type(Y) isa Binary ? "classification" : "regression"

    return DoublyRobustLearner(
        X,
        Float64.(T),
        Float64.(Y),
        "CATE",
        false,
        task,
        activation,
        sample_size,
        num_machines,
        num_feats,
        num_neurons,
        fill(NaN, size(T, 1)),
        folds,
    )
end

"""
    estimate_causal_effect!(s)

Estimate the CATE using an S-learner.

# References
For an overview of S-learning see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m4 = SLearner(X, T, Y)
julia> estimate_causal_effect!(m4)
```
"""
function estimate_causal_effect!(s::SLearner)
    s.causal_effect = g_formula!(s)
    return s.causal_effect
end

"""
    estimate_causal_effect!(t)

Estimate the CATE using an T-learner.

# References
For an overview of T-learning see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m5 = TLearner(X, T, Y)
julia> estimate_causal_effect!(m5)
```
"""
function estimate_causal_effect!(t::TLearner)
    x₀, x₁, y₀, y₁ = t.X[t.T .== 0, :], t.X[t.T .== 1, :], t.Y[t.T .== 0], t.Y[t.T .== 1]

    t.μ₀ = ELMEnsemble(
        x₀, y₀, t.sample_size, t.num_machines, t.num_feats, t.num_neurons, t.activation
    )

    t.μ₁ = ELMEnsemble(
        x₁, y₁, t.sample_size, t.num_machines, t.num_feats, t.num_neurons, t.activation
    )

    fit!(t.μ₀)
    fit!(t.μ₁)
    predictionsₜ, predictionsᵪ = predict(t.μ₁, t.X), predict(t.μ₀, t.X)
    t.causal_effect = @fastmath vec(predictionsₜ - predictionsᵪ)

    return t.causal_effect
end

"""
    estimate_causal_effect!(x)

Estimate the CATE using an X-learner.

# References
For an overview of X-learning see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(x::XLearner)
    stage1!(x)
    μχ₀, μχ₁ = stage2!(x)

    x.causal_effect = @fastmath vec((
        (x.ps .* predict(μχ₀, x.X)) .+ ((1 .- x.ps) .* predict(μχ₁, x.X))
    ))

    return x.causal_effect
end

"""
    estimate_causal_effect!(R)

Estimate the CATE using an R-learner.

# References
For an overview of R-learning see:
    Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
    effects." Biometrika 108, no. 2 (2021): 299-319.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = RLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(R::RLearner)
    X, T̃, Ỹ = generate_folds(R.X, R.T, R.Y, R.folds)
    R.X, R.T, R.Y = reduce(vcat, X), reduce(vcat, T̃), reduce(vcat, Ỹ)

    # Get residuals from out-of-fold predictions
    for f in 1:(R.folds)
        X_train, X_test = reduce(vcat, X[1:end .!== f]), X[f]
        Y_train, Y_test = reduce(vcat, Ỹ[1:end .!== f]), Ỹ[f]
        T_train, T_test = reduce(vcat, T̃[1:end .!== f]), T̃[f]
        Ỹ[f], T̃[f] = predict_residuals(R, X_train, X_test, Y_train, Y_test, T_train, T_test)
    end

    # Using target transformation and the weight trick to minimize the causal loss
    T̃², target = reduce(vcat, T̃).^2, reduce(vcat, T̃) ./ reduce(vcat, Ỹ)
    W⁻⁵ᵉ⁻¹ = Diagonal(sqrt.(T̃²))
    Xʷ, Yʷ = W⁻⁵ᵉ⁻¹ * R.X, W⁻⁵ᵉ⁻¹ * target

    # Fit a weighted residual-on-residual model
    final_model = ELMEnsemble(
        Xʷ, Yʷ, R.sample_size, R.num_machines, R.num_feats, R.num_neurons, R.activation
    )
    fit!(final_model)
    R.causal_effect = predict(final_model, R.X)

    return R.causal_effect
end

"""
    estimate_causal_effect!(DRE)

Estimate the CATE using a doubly robust learner.

# References
For details on how this method estimates the CATE see:
    Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
    effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoublyRobustLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(DRE::DoublyRobustLearner)
    X, T, Y = generate_folds(DRE.X, DRE.T, DRE.Y, DRE.folds)
    causal_effect = zeros(size(DRE.T, 1))

    # Rotating folds for cross fitting
    for i in 1:2
        causal_effect .+= doubly_robust_formula!(DRE, X, T, Y)
        X, T, Y = [X[2], X[1]], [T[2], T[1]], [Y[2], Y[1]]
    end

    causal_effect ./= 2
    DRE.causal_effect = causal_effect

    return DRE.causal_effect
end

"""
    doubly_robust_formula!(DRE, X, T, Y)

Estimate the CATE for a single cross fitting iteration via doubly robust estimation.

# Notes
This method should not be called directly.

# Arguments
- `DRE::DoublyRobustLearner`: the DoubleMachineLearning struct to estimate the effect for.
- `X`: a vector of three covariate folds.
- `T`: a vector of three treatment folds.
- `Y`: a vector of three outcome folds.

# Examples
```julia
julia> X, T, Y, W =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100), rand(6, 100)
julia> m1 = DoublyRobustLearner(X, T, Y)

julia> X, T, W, Y = make_folds(m1)
julia> Z = m1.W == m1.X ? X : [reduce(hcat, (z)) for z in zip(X, W)]
julia> g_formula!(m1, X, T, Y, Z)
```
"""
function doubly_robust_formula!(DRE::DoublyRobustLearner, X, T, Y)
    # Propensity scores
    π_e = ELMEnsemble(
        X[1], 
        T[1], 
        DRE.sample_size, 
        DRE.num_machines, 
        DRE.num_feats, 
        DRE.num_neurons, 
        DRE.activation
    )

    # Outcome models
    μ₀ = ELMEnsemble(
        X[1][T[1] .== 0, :], 
        Y[1][T[1] .== 0], 
        DRE.sample_size, 
        DRE.num_machines, 
        DRE.num_feats,
        DRE.num_neurons, 
        DRE.activation
    )

    μ₁ = ELMEnsemble(
        X[1][T[1] .== 1, :], 
        Y[1][T[1] .== 1], 
        DRE.sample_size, 
        DRE.num_machines, 
        DRE.num_feats,
        DRE.num_neurons, 
        DRE.activation
    )

    fit!.((π_e, μ₀, μ₁))
    π̂ , μ₀̂, μ₁̂  = predict(π_e, X[2]), predict(μ₀, X[2]), predict(μ₁, X[2])

    # Pseudo outcomes
    ϕ̂ =
        ((T[2] .- π̂) ./ (π̂ .* (1 .- π̂))) .*
        (Y[2] .- T[2] .* μ₁̂ .- (1 .- T[2]) .* μ₀̂) .+ μ₁̂ .- μ₀̂

    # Final model
    τ_est = ELMEnsemble(
        X[2], 
        ϕ̂, 
        DRE.sample_size, 
        DRE.num_machines, 
        DRE.num_feats, 
        DRE.num_neurons, 
        DRE.activation
    )
    fit!(τ_est)
    return predict(τ_est, DRE.X)
end

"""
stage1!(x)

Estimate the first stage models for an X-learner.

# Notes
This method should not be called by the user.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> stage1!(m1)
```
"""
function stage1!(x::XLearner)
    g = ELMEnsemble(
        x.X, x.T, x.sample_size, x.num_machines, x.num_feats, x.num_neurons, x.activation
    )

    x.μ₀ = ELMEnsemble(
        x.X[x.T .== 0, :], 
        x.Y[x.T .== 0], 
        x.sample_size, 
        x.num_machines, 
        x.num_feats,
        x.num_neurons, 
        x.activation
    )

    x.μ₁ = ELMEnsemble(
        x.X[x.T .== 1, :], 
        x.Y[x.T .== 1], 
        x.sample_size, 
        x.num_machines, 
        x.num_feats,
        x.num_neurons, 
        x.activation
    )

    # Get propensity scores
    fit!(g)
    x.ps = predict(g, x.X)

    # Fit first stage outcome models
    fit!(x.μ₀)
    return fit!(x.μ₁)
end

"""
stage2!(x)

Estimate the second stage models for an X-learner.

# Notes
This method should not be called by the user.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> stage1!(m1)
julia> stage2!(m1)
```
"""
function stage2!(x::XLearner)
    m₁, m₀ = predict(x.μ₁, x.X .- x.Y), predict(x.μ₀, x.X)
    d = ifelse(x.T === 0, m₁, x.Y .- m₀)
    
    μχ₀ = ELMEnsemble(
        x.X[x.T .== 0, :], 
        d[x.T .== 0], 
        x.sample_size, 
        x.num_machines, 
        x.num_feats,
        x.num_neurons, 
        x.activation
    )

    μχ₁ = ELMEnsemble(
        x.X[x.T .== 1, :], 
        d[x.T .== 1], 
        x.sample_size, 
        x.num_machines, 
        x.num_feats,
        x.num_neurons, 
        x.activation
    )

    fit!(μχ₀)
    fit!(μχ₁)

    return μχ₀, μχ₁
end
