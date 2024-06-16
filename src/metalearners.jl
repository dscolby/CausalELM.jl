"""Abstract type for metalearners"""
abstract type Metalearner end

"""Stores variables, results, and configuration for S-learning"""
mutable struct SLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    learner::ExtremeLearningMachine

    """@doc
        SLearner(X, T, Y; kwargs...)

    Initialize a S-Learner.

    # Arguments
    - `X::Any`: an array or DataFrame of covariates.
    - `T::Any`: an vector or DataFrame of treatment statuses.
    - `Y::Any`: an array or DataFrame of outcomes.

    # Keywords
    - `regularized::Function=true`: whether to use L2 regularization
    - `activation::Function=relu`: the activation function to use.
    - `validation_metric::Function`: the validation metric to calculate during cross validation.
    - `min_neurons::Real`: the minimum number of neurons to consider for the extreme learner.
    - `max_neurons::Real`: the maximum number of neurons to consider for the extreme learner.
    - `folds::Real`: the number of cross validation folds to find the best number of neurons.
    - `iterations::Real`: the number of iterations to perform cross validation between 
        min_neurons and max_neurons.
    - `approximator_neurons::Real`: the number of nuerons in the validation loss approximator 
        network.

    # Notes
    If regularized is set to true then the ridge penalty will be estimated using generalized 
    cross validation where the maximum number of iterations is 2 * folds for the successive 
    halving procedure. However, if the penalty in on iteration is approximately the same as 
    in the previous penalty, then the procedure will stop early.

    # References
    For an overview of S-Learners and other metalearners see:
        Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
        estimating heterogeneous treatment effects using machine learning." Proceedings of 
        the national academy of sciences 116, no. 10 (2019): 4156-4165.

    For details and a derivation of the generalized cross validation estimator see:
        Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
        method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 
        215-223.

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
    function SLearner(
        X,
        T,
        Y;
        regularized::Bool=true,
        activation::Function=relu,
        validation_metric::Function=mse,
        min_neurons::Real=1,
        max_neurons::Real=100,
        folds::Real=5,
        iterations::Real=round(size(X, 1) / 10),
        approximator_neurons::Real=round(size(X, 1) / 10),
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
            regularized,
            activation,
            validation_metric,
            min_neurons,
            max_neurons,
            folds,
            iterations,
            approximator_neurons,
            0,
            fill(NaN, size(T, 1)),
        )
    end
end

"""Stores variables, results, and configuration for T-learning"""
mutable struct TLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    μ₀::ExtremeLearningMachine
    μ₁::ExtremeLearningMachine

    """@doc
        TLearner(X, T, Y; kwargs...)

    Initialize a T-Learner.

    # Arguments
    - `X::Any`: an array or DataFrame of covariates.
    - `T::Any`: an vector or DataFrame of treatment statuses.
    - `Y::Any`: an array or DataFrame of outcomes.

    # Keywords
    - `regularized::Function=true`: whether to use L2 regularization
    - `activation::Function=relu`: the activation function to use.
    - `validation_metric::Function`: the validation metric to calculate during cross 
        validation.
    - `min_neurons::Real`: the minimum number of neurons to consider for the extreme 
        learner.
    - `max_neurons::Real`: the maximum number of neurons to consider for the extreme 
        learner.
    - `folds::Real`: the number of cross validation folds to find the best number of 
        neurons.
    - `iterations::Real`: the number of iterations to perform cross validation between 
        min_neurons and max_neurons.
    - `approximator_neurons::Real`: the number of nuerons in the validation loss approximator 
        network.

    # Notes
    If regularized is set to true then the ridge penalty will be estimated using generalized 
    cross validation where the maximum number of iterations is 2 * folds for the successive 
    halving procedure. However, if the penalty in on iteration is approximately the same as 
    in the previous penalty, then the procedure will stop early.

    # References
    For an overview of T-Learners and other metalearners see:
        Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
        estimating heterogeneous treatment effects using machine learning." Proceedings of 
        the national academy of sciences 116, no. 10 (2019): 4156-4165.

    For details and a derivation of the generalized cross validation estimator see:
        Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
        method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 
        215-223.

    # Examples
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
    function TLearner(
        X,
        T,
        Y;
        regularized::Bool=false,
        activation::Function=relu,
        validation_metric::Function=mse,
        min_neurons::Real=1,
        max_neurons::Real=100,
        folds::Real=5,
        iterations::Real=round(size(X, 1) / 10),
        approximator_neurons::Real=round(size(X, 1) / 10),
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
            regularized,
            activation,
            validation_metric,
            min_neurons,
            max_neurons,
            folds,
            iterations,
            approximator_neurons,
            0,
            fill(NaN, size(T, 1)),
        )
    end
end

"""Stores variables, results, and configuration for X-learning"""
mutable struct XLearner <: Metalearner
    @standard_input_data
    @model_config individual_effect
    μ₀::ExtremeLearningMachine
    μ₁::ExtremeLearningMachine
    ps::Array{Float64}

    """@doc
        XLearner(X, T, Y; kwargs...)

    Initialize an X-Learner.

    # Arguments
    - `X::Any`: an array or DataFrame of covariates.
    - `T::Any`: an vector or DataFrame of treatment statuses.
    - `Y::Any`: an array or DataFrame of outcomes.

    # Keywords
    - `regularized::Function=true`: whether to use L2 regularization
    - `activation::Function=relu`: the activation function to use.
    - `validation_metric::Function`: the validation metric to calculate during cross 
        validation.
    - `min_neurons::Real`: the minimum number of neurons to consider for the extreme 
        learner.
    - `max_neurons::Real`: the maximum number of neurons to consider for the extreme 
        learner.
    - `folds::Real`: the number of cross validation folds to find the best number of 
        neurons.
    - `iterations::Real`: the number of iterations to perform cross validation between 
        min_neurons and max_neurons.
    - `approximator_neurons::Real`: the number of nuerons in the validation loss 
        approximator network.

    # Notes
    If regularized is set to true then the ridge penalty will be estimated using generalized 
    cross validation where the maximum number of iterations is 2 * folds for the successive 
    halving procedure. However, if the penalty in on iteration is approximately the same as 
    in the previous penalty, then the procedure will stop early.

    # References
    For an overview of X-Learners and other metalearners see:
        Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
        estimating heterogeneous treatment effects using machine learning." Proceedings of 
        the national academy of sciences 116, no. 10 (2019): 4156-4165.

    For details and a derivation of the generalized cross validation estimator see:
        Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
        method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 
        215-223.

    # Examples
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
    function XLearner(
        X,
        T,
        Y;
        regularized::Bool=false,
        activation::Function=relu,
        validation_metric::Function=mse,
        min_neurons::Real=1,
        max_neurons::Real=100,
        folds::Real=5,
        iterations::Real=round(size(X, 1) / 10),
        approximator_neurons::Real=round(size(X, 1) / 10),
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
            regularized,
            activation,
            validation_metric,
            min_neurons,
            max_neurons,
            folds,
            iterations,
            approximator_neurons,
            0,
            fill(NaN, size(T, 1)),
        )
    end
end

"""Stores variables, results, and configuration for R-learning"""
mutable struct RLearner <: Metalearner
    @double_learner_input_data
    @model_config individual_effect
end

"""@doc
    RLearner(X, T, Y; kwargs...)

Initialize an R-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `W::Any` : an array of all possible confounders.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Real`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Real`: the maximum number of neurons to consider for the extreme learner.
- `folds::Real`: the number of cross validation folds to find the best number of neurons.
- `iterations::Real`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Real`: the number of nuerons in the validation loss approximator 
    network.

# Notes
If regularized is set to true then the ridge penalty will be estimated using generalized 
cross validation where the maximum number of iterations is 2 * folds for the successive 
halving procedure. However, if the penalty in on iteration is approximately the same as in 
the previous penalty, then the procedure will stop early.

## References
For an explanation of R-Learner estimation see:
    Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
    effects." Biometrika 108, no. 2 (2021): 299-319.
    
For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = RLearner(X, T, Y)
julia> m2 = RLearner(X, T, Y; t_cat=true)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = RLearner(x_df, t_df, y_df)

julia> w = rand(100, 6)
julia> m5 = RLearner(X, T, Y, W=w)
```
"""
function RLearner(
    X,
    T,
    Y;
    W=X,
    activation::Function=relu,
    validation_metric::Function=mse,
    min_neurons::Real=1,
    max_neurons::Real=100,
    folds::Real=5,
    iterations::Real=round(size(X, 1) / 10),
    approximator_neurons::Real=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y, W = Matrix{Float64}(X), T[:, 1], Y[:, 1], Matrix{Float64}(W)

    task = var_type(Y) isa Binary ? "classification" : "regression"

    return RLearner(
        X,
        Float64.(T),
        Float64.(Y),
        W,
        "CATE",
        false,
        task,
        true,
        activation,
        validation_metric,
        min_neurons,
        max_neurons,
        folds,
        iterations,
        approximator_neurons,
        0,
        fill(NaN, size(T, 1)),
    )
end

"""Stores variables, results, and configuration for doubly robust learning"""
mutable struct DoublyRobustLearner <: Metalearner
    @double_learner_input_data
    @model_config individual_effect
end

"""@doc
    DoublyRobustLearner(X, T, Y; kwargs...)

Initialize a doubly robust CATE estimator.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `W::Any`: an array or dataframe of all possible confounders.
- `regularized::Function=true`: whether to use L2 regularization
- `activation::Function=relu`: the activation function to use.
- `validation_metric::Function`: the validation metric to calculate during cross validation.
- `min_neurons::Real`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Real`: the maximum number of neurons to consider for the extreme learner.
- `folds::Real`: the number of cross validation folds to find the best number of neurons.
- `iterations::Real`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `approximator_neurons::Real`: the number of nuerons in the validation loss approximator 
    network.

# Notes
If regularized is set to true then the ridge penalty will be estimated using generalized 
cross validation where the maximum number of iterations is 2 * folds for the successive 
halving procedure. However, if the penalty in on iteration is approximately the same as in 
the previous penalty, then the procedure will stop early.

# References
For an explanation of doubly robust cate estimation see:
    Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
    effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = DoublyRobustLearner(X, T, Y)
julia> m2 = DoublyRobustLearnerLearner(X, T, Y; t_cat=true)

julia> x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
julia> t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
julia> m4 = DoublyRobustLearner(x_df, t_df, y_df)

julia> w = rand(100, 6)
julia> m5 = DoublyRobustLearner(X, T, Y, W=w)
```
"""
function DoublyRobustLearner(
    X,
    T,
    Y;
    W=X,
    regularized::Bool=true,
    activation::Function=relu,
    validation_metric::Function=mse,
    min_neurons::Real=1,
    max_neurons::Real=100,
    iterations::Real=round(size(X, 1) / 10),
    approximator_neurons::Real=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y, W = Matrix{Float64}(X), T[:, 1], Y[:, 1], Matrix{Float64}(W)

    task = var_type(Y) isa Binary ? "classification" : "regression"

    return DoublyRobustLearner(
        X,
        Float64.(T),
        Float64.(Y),
        W,
        "CATE",
        false,
        task,
        regularized,
        activation,
        validation_metric,
        min_neurons,
        max_neurons,
        2,
        iterations,
        approximator_neurons,
        0,
        fill(NaN, size(T, 1)),
    )
end

"""
    estimate_causal_effect!(s)

Estimate the CATE using an S-learner.

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
    type = var_type(t.Y)

    # Only search for the best number of neurons once and use the same number for inference
    t.num_neurons = t.num_neurons === 0 ? best_size(t) : t.num_neurons

    if t.regularized
        t.μ₀ = RegularizedExtremeLearner(x₀, y₀, t.num_neurons, t.activation)
        t.μ₁ = RegularizedExtremeLearner(x₁, y₁, t.num_neurons, t.activation)
    else
        t.μ₀ = ExtremeLearner(x₀, y₀, t.num_neurons, t.activation)
        t.μ₁ = ExtremeLearner(x₁, y₁, t.num_neurons, t.activation)
    end

    fit!(t.μ₀)
    fit!(t.μ₁)
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

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(x::XLearner)
    # Only search for the best number of neurons once and use the same number for inference
    x.num_neurons = x.num_neurons === 0 ? best_size(x) : x.num_neurons

    type = var_type(x.Y)
    stage1!(x)
    μχ₀, μχ₁ = stage2!(x)

    x.causal_effect = @fastmath vec((
        (x.ps .* clip_if_binary(predict(μχ₀, x.X), type)) .+
        ((1 .- x.ps) .* clip_if_binary(predict(μχ₁, x.X), type))
    ))

    return x.causal_effect
end

"""
    estimate_causal_effect!(R)

Estimate the CATE using an R-learner.

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
    # Uses the same number of neurons for all phases of estimation
    R.num_neurons = R.num_neurons === 0 ? best_size(R) : R.num_neurons

    # Just estimate the causal effect using the underlying DML and the weight trick
    R.causal_effect = causal_loss(R)

    return R.causal_effect
end

function causal_loss(R::RLearner)
    X, T, W, Y = make_folds(R)
    predictors = Vector{RegularizedExtremeLearner}(undef, R.folds)

    # Cross fitting by training on the main folds and predicting residuals on the auxillary
    for fld in 1:(R.folds)
        X_train, X_test = reduce(vcat, X[1:end .!== fld]), X[fld]
        Y_train, Y_test = reduce(vcat, Y[1:end .!== fld]), Y[fld]
        T_train, T_test = reduce(vcat, T[1:end .!== fld]), T[fld]
        W_train, W_test = reduce(vcat, W[1:end .!== fld]), W[fld]

        Ỹ, T̃ = predict_residuals(
            R, X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test
        )

        # Using the weight trick to get the non-parametric CATE for an R-learner
        X[fld], Y[fld] = (T̃ .^ 2) .* X_test, (T̃ .^ 2) .* (Ỹ ./ T̃)
        mod = RegularizedExtremeLearner(X[fld], Y[fld], R.num_neurons, R.activation)
        fit!(mod)
        predictors[fld] = mod
    end
    final_predictions = [predict(m, reduce(vcat, X)) for m in predictors]
    return vec(mapslices(mean, reduce(hcat, final_predictions); dims=2))
end

"""
    estimate_causal_effect!(DRE)

Estimate the CATE using a doubly robust learner.

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
    X, T, W, Y = make_folds(DRE)
    Z = DRE.W == DRE.X ? X : [reduce(hcat, (z)) for z in zip(X, W)]
    causal_effect = zeros(size(DRE.T, 1))

    # Uses the same number of neurons for all phases of estimation
    DRE.num_neurons = DRE.num_neurons === 0 ? best_size(DRE) : DRE.num_neurons

    # Rotating folds for cross fitting
    for i in 1:(DRE.folds)
        causal_effect .+= doubly_robust_formula!(DRE, X, T, Y, Z)
        X, T, Y, Z = [X[2], X[1]], [T[2], T[1]], [Y[2], Y[1]], [Z[2], Z[1]]
    end

    causal_effect ./= 2
    DRE.causal_effect = causal_effect

    return DRE.causal_effect
end

"""
    g_formula!(DRE, X, T, Y, Z)

Estimate the CATE for a single cross fitting iteration via doubly robust estimation.

This method should not be called directly.

# Arguments
- `DRE::DoublyRobustLearner`: the DoubleMachineLearning struct to estimate the effect for.
- `X`: a vector of three covariate folds.
- `T`: a vector of three treatment folds.
- `Y`: a vector of three outcome folds.
- `Z` : a vector of three confounder folds and covariate folds.

# Examples
```julia
julia> X, T, Y, W =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100), rand(6, 100)
julia> m1 = DoublyRobustLearner(X, T, Y, W=W)

julia> X, T, W, Y = make_folds(m1)
julia> Z = m1.W == m1.X ? X : [reduce(hcat, (z)) for z in zip(X, W)]
julia> g_formula!(m1, X, T, Y, Z)
```
"""
function doubly_robust_formula!(DRE::DoublyRobustLearner, X, T, Y, Z)
    π_arg, P = (Z[1], T[1], DRE.num_neurons, σ), var_type(DRE.Y)
    μ₀_arg = Z[1][T[1] .== 0, :], Y[1][T[1] .== 0], DRE.num_neurons, DRE.activation
    μ₁_arg = Z[1][T[1] .== 1, :], Y[1][T[1] .== 1], DRE.num_neurons, DRE.activation

    # Propensity scores
    π_e = DRE.regularized ? RegularizedExtremeLearner(π_arg...) : ExtremeLearner(π_arg...)

    # Outcome predictions
    μ₀ = DRE.regularized ? RegularizedExtremeLearner(μ₀_arg...) : ExtremeLearner(μ₀_arg...)
    μ₁ = DRE.regularized ? RegularizedExtremeLearner(μ₁_arg...) : ExtremeLearner(μ₁_arg...)

    fit!.((π_e, μ₀, μ₁))
    π̂ = clip_if_binary(predict(π_e, Z[2]), Binary())
    μ₀̂, μ₁̂ = clip_if_binary(predict(μ₀, Z[2]), P), clip_if_binary(predict(μ₁, Z[2]), P)

    # Pseudo outcomes
    ϕ̂ =
        ((T[2] .- π̂) ./ (π̂ .* (1 .- π̂))) .*
        (Y[2] .- T[2] .* μ₁̂ .- (1 .- T[2]) .* μ₀̂) .+ μ₁̂ .- μ₀̂

    # Final model
    τ_arg = X[2], ϕ̂, DRE.num_neurons, DRE.activation
    τ_est = DRE.regularized ? RegularizedExtremeLearner(τ_arg...) : ExtremeLearner(τ_arg...)
    fit!(τ_est)

    return clip_if_binary(predict(τ_est, DRE.X), P)
end

"""
stage1!(x)

Estimate the first stage models for an X-learner.

This method should not be called by the user.

# Examples
```julia
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = XLearner(X, T, Y)
julia> stage1!(m1)
```
"""
function stage1!(x::XLearner)
    if x.regularized
        g = RegularizedExtremeLearner(x.X, x.T, x.num_neurons, x.activation)
        x.μ₀ = RegularizedExtremeLearner(
            x.X[x.T .== 0, :], x.Y[x.T .== 0], x.num_neurons, x.activation
        )
        x.μ₁ = RegularizedExtremeLearner(
            x.X[x.T .== 1, :], x.Y[x.T .== 1], x.num_neurons, x.activation
        )
    else
        g = ExtremeLearner(x.X, x.T, x.num_neurons, x.activation)
        x.μ₀ = ExtremeLearner(
            x.X[x.T .== 0, :], x.Y[x.T .== 0], x.num_neurons, x.activation
        )
        x.μ₁ = ExtremeLearner(
            x.X[x.T .== 1, :], x.Y[x.T .== 1], x.num_neurons, x.activation
        )
    end

    # Get propensity scores
    fit!(g)
    x.ps = clip_if_binary(predict(g, x.X), Binary())

    # Fit first stage outcome models
    fit!(x.μ₀)
    return fit!(x.μ₁)
end

"""
stage2!(x)

Estimate the second stage models for an X-learner.

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
    m₁ = clip_if_binary(predict(x.μ₁, x.X .- x.Y), var_type(x.Y))
    m₀ = clip_if_binary(predict(x.μ₀, x.X), var_type(x.Y))
    d = ifelse(x.T === 0, m₁, x.Y .- m₀)

    if x.regularized
        μχ₀ = RegularizedExtremeLearner(
            x.X[x.T .== 0, :], d[x.T .== 0], x.num_neurons, x.activation
        )
        μχ₁ = RegularizedExtremeLearner(
            x.X[x.T .== 1, :], d[x.T .== 1], x.num_neurons, x.activation
        )
    else
        μχ₀ = ExtremeLearner(x.X[x.T .== 0, :], d[x.T .== 0], x.num_neurons, x.activation)
        μχ₁ = ExtremeLearner(x.X[x.T .== 1, :], d[x.T .== 1], x.num_neurons, x.activation)
    end

    fit!(μχ₀)
    fit!(μχ₁)

    return μχ₀, μχ₁
end
