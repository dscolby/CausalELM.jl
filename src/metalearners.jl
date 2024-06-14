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
- `task::String`: either regression or classification.
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
For an overview of S-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = SLearner(X, T, Y)
m2 = SLearner(X, T, Y; task="regression")
m3 = SLearner(X, T, Y; task="regression", regularized=true)

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
m4 = SLearner(x_df, t_df, y_df)
```
"""
mutable struct SLearner <: Metalearner
    g::GComputation
    causal_effect::Array{Float64}

    function SLearner(
        X,
        T,
        Y;
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
            GComputation(
                X,
                T,
                Y;
                task=task,
                quantity_of_interest="ATE",
                regularized=regularized,
                activation=activation,
                temporal=false,
                validation_metric=validation_metric,
                min_neurons=min_neurons,
                max_neurons=max_neurons,
                folds=folds,
                iterations=iterations,
                approximator_neurons=approximator_neurons,
            ),
            fill(NaN, size(T, 1))
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
- `task::String`: either regression or classification.
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
For an overview of T-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = TLearner(X, T, Y)
m2 = TLearner(X, T, Y; task="regression")
m3 = TLearner(X, T, Y; task="regression", regularized=true)

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
m4 = TLearner(x_df, t_df, y_df)
```
"""
mutable struct TLearner <: Metalearner
    @standard_input_data
    @model_config "individual_effect"
    μ₀::ExtremeLearningMachine
    μ₁::ExtremeLearningMachine

    function TLearner(
        X::Array{<:Real},
        T::Array{<:Real},
        Y::Array{<:Real};
        task="regression",
        regularized=false,
        activation=relu,
        validation_metric=mse,
        min_neurons=1,
        max_neurons=100,
        folds=5,
        iterations=round(size(X, 1) / 10),
        approximator_neurons=round(size(X, 1) / 10),
    )
        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        return new(
            Float64.(X),
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
            fill(NaN, size(T, 1))
        )
    end
end

function TLearner(
    X,
    T,
    Y;
    task="regression",
    regularized=false,
    activation=relu,
    validation_metric=mse,
    min_neurons=1,
    max_neurons=100,
    folds=5,
    iterations=round(size(X, 1) / 10),
    approximator_neurons=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    return TLearner(
        X,
        T,
        Y;
        task=task,
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
    XLearner(X, T, Y; kwargs...)

Initialize an X-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `task::String`: either regression or classification.
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
For an overview of X-Learners and other metalearners see:
    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = XLearner(X, T, Y)
m2 = XLearner(X, T, Y; task="regression")
m3 = XLearner(X, T, Y; task="regression", regularized=true)

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
m4 = XLearner(x_df, t_df, y_df)
```
"""
mutable struct XLearner <: Metalearner
    @standard_input_data
    @model_config "individual_effect"
    μ₀::ExtremeLearningMachine
    μ₁::ExtremeLearningMachine
    ps::Array{Float64}

    function XLearner(
        X::Array{<:Real},
        T::Array{<:Real},
        Y::Array{<:Real};
        task="regression",
        regularized=false,
        activation=relu,
        validation_metric=mse,
        min_neurons=1,
        max_neurons=100,
        folds=5,
        iterations=round(size(X, 1) / 10),
        approximator_neurons=round(size(X, 1) / 10),
    )
        if task ∉ ("regression", "classification")
            throw(ArgumentError("task must be either regression or classification"))
        end

        return new(
            Float64.(X),
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
            fill(NaN, size(T, 1))
        )
    end
end

function XLearner(
    X,
    T,
    Y;
    task="regression",
    regularized=false,
    activation=relu,
    validation_metric=mse,
    min_neurons=1,
    max_neurons=100,
    folds=5,
    iterations=round(size(X, 1) / 10),
    approximator_neurons=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y = Matrix{Float64}(X), T[:, 1], Y[:, 1]

    return XLearner(
        X,
        T,
        Y;
        task=task,
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
    RLearner(X, T, Y; kwargs...)

Initialize an R-Learner.

# Arguments
- `X::Any`: an array or DataFrame of covariates of interest.
- `T::Any`: an vector or DataFrame of treatment statuses.
- `Y::Any`: an array or DataFrame of outcomes.

# Keywords
- `W::Any` : an array of all possible confounders.
- `task::String`: either regression or classification.
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

## References
For an explanation of R-Learner estimation see:
    Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
    effects." Biometrika 108, no. 2 (2021): 299-319.
    
For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = RLearner(X, T, Y)
m2 = RLearner(X, T, Y; t_cat=true)

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
m4 = RLearner(x_df, t_df, y_df)

w = rand(100, 6)
m5 = RLearner(X, T, Y, W=w)
```
"""
mutable struct RLearner <: Metalearner
    dml::DoubleMachineLearning
    causal_effect::Array{Float64}

    function RLearner(
        X,
        T,
        Y;
        W=X,
        activation=relu,
        validation_metric=mse,
        min_neurons=1,
        max_neurons=100,
        folds=5,
        iterations=round(size(X, 1) / 10),
        approximator_neurons=round(size(X, 1) / 10),
    )
        return new(
            DoubleMachineLearning(
                X,
                T,
                Y;
                W=W,
                regularized=true,
                activation=activation,
                validation_metric=validation_metric,
                min_neurons=min_neurons,
                max_neurons=max_neurons,
                folds=folds,
                iterations=iterations,
                approximator_neurons=approximator_neurons,
            ),
            fill(NaN, size(T, 1))
        )
    end
end

"""
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
For an explanation of doubly robust cate estimation see:
    Kennedy, Edward H. "Towards optimal doubly robust estimation of heterogeneous causal 
    effects." Electronic Journal of Statistics 17, no. 2 (2023): 3008-3049.

For details and a derivation of the generalized cross validation estimator see:
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = DoublyRobustLearner(X, T, Y)
m2 = DoublyRobustLearnerLearner(X, T, Y; t_cat=true)

x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
m4 = DoublyRobustLearner(x_df, t_df, y_df)

w = rand(100, 6)
m5 = DoublyRobustLearner(X, T, Y, W=w)
```
"""
mutable struct DoublyRobustLearner <: Metalearner
    @double_learner_input_data
    @model_config "individual_effect"

    function DoublyRobustLearner(
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
        iterations=round(size(X, 1) / 10),
        approximator_neurons=round(size(X, 1) / 10),
    )
        return new(
            Float64.(X),
            Float64.(T),
            Float64.(Y),
            Float64.(W),
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
end

function DoublyRobustLearner(
    X,
    T,
    Y;
    W=X,
    task="regression",
    regularized=true,
    activation=relu,
    validation_metric=mse,
    min_neurons=1,
    max_neurons=100,
    iterations=round(size(X, 1) / 10),
    approximator_neurons=round(size(X, 1) / 10),
)

    # Convert to arrays
    X, T, Y, W = Matrix{Float64}(X), T[:, 1], Y[:, 1], Matrix{Float64}(W)

    return DoublyRobustLearner(
        X,
        T,
        Y;
        W=W,
        task=task,
        regularized=regularized,
        activation=activation,
        validation_metric=validation_metric,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        iterations=iterations,
        approximator_neurons=approximator_neurons,
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
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m4 = SLearner(X, T, Y)
estimate_causal_effect!(m4)
```
"""
function estimate_causal_effect!(s::SLearner)
    estimate_causal_effect!(s.g)
    y_type = var_type(s.g.Y)
    Xₜ, Xᵤ = hcat(s.g.X, ones(size(s.g.T, 1))), hcat(s.g.X, zeros(size(s.g.T, 1)))

    # Clipping binary predictions to be ∈ [0, 1]
    if s.g.task === "classification"
        yₜ = clip_if_binary(predict(s.g.learner, Xₜ), y_type)
        yᵤ = clip_if_binary(predict(s.g.learner, s.Xᵤ), y_type)
    else
        yₜ, yᵤ = predict(s.g.learner, Xₜ), predict(s.g.learner, Xᵤ)
    end

    s.causal_effect = yₜ .- yᵤ

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
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m5 = TLearner(X, T, Y)
estimate_causal_effect!(m5)
```
"""
function estimate_causal_effect!(t::TLearner)
    x₀, x₁, y₀, y₁ = t.X[t.T .== 0, :], t.X[t.T .== 1, :], t.Y[t.T .== 0], t.Y[t.T .== 1]
    type = var_type(t.Y)

    # Only search for the best number of neurons once and use the same number for inference
    if t.num_neurons === 0
        t.num_neurons = best_size(
            t.X,
            t.Y,
            t.validation_metric,
            t.task,
            t.activation,
            t.min_neurons,
            t.max_neurons,
            t.regularized,
            t.folds,
            false,
            t.iterations,
            t.approximator_neurons,
        )
    end

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
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = XLearner(X, T, Y)
estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(x::XLearner)
    # Only search for the best number of neurons once and use the same number for inference
    if x.num_neurons === 0
        x.num_neurons = best_size(
            x.X,
            x.Y,
            x.validation_metric,
            x.task,
            x.activation,
            x.min_neurons,
            x.max_neurons,
            x.regularized,
            x.folds,
            false,
            x.iterations,
            x.approximator_neurons,
        )
    end

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
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = RLearner(X, T, Y)
estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(R::RLearner)
    # Uses the same number of neurons for all phases of estimation
    if R.dml.num_neurons === 0
        R.dml.num_neurons = best_size(
            R.dml.X,
            R.dml.Y,
            R.dml.validation_metric,
            "regression",
            R.dml.activation,
            R.dml.min_neurons,
            R.dml.max_neurons,
            R.dml.regularized,
            R.dml.folds,
            false,
            R.dml.iterations,
            R.dml.approximator_neurons,
        )
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

# Examples
```julia
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = DoublyRobustLearner(X, T, Y)
estimate_causal_effect!(m1)
```
"""
function estimate_causal_effect!(DRE::DoublyRobustLearner)
    X, T, W, Y = make_folds(DRE)
    Z = DRE.W == DRE.X ? X : [reduce(hcat, (z)) for z in zip(X, W)]
    task = var_type(DRE.Y) == Binary() ? "classification" : "regression"
    causal_effect = zeros(size(DRE.T, 1))

    # Uses the same number of neurons for all phases of estimation
    if DRE.num_neurons === 0
        DRE.num_neurons = best_size(
            DRE.X,
            DRE.Y,
            DRE.validation_metric,
            task,
            DRE.activation,
            DRE.min_neurons,
            DRE.max_neurons,
            DRE.regularized,
            DRE.folds,
            false,
            DRE.iterations,
            DRE.approximator_neurons,
        )
    end

    # Rotating folds for cross fitting
    for i in 1:(DRE.folds)
        causal_effect .+= estimate_effect!(DRE, X, T, Y, Z)
        X, T, Y, Z = [X[2], X[1]], [T[2], T[1]], [Y[2], Y[1]], [Z[2], Z[1]]
    end

    causal_effect ./= 2
    DRE.causal_effect = causal_effect

    return DRE.causal_effect
end

"""
    estimate_effect!(DRE, X, T, Y, Z)

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
X, T, Y, W =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100), rand(6, 100)
m1 = DoublyRobustLearner(X, T, Y, W=W)

X, T, W, Y = make_folds(m1)
Z = m1.W == m1.X ? X : [reduce(hcat, (z)) for z in zip(X, W)]
estimate_effect!(m1, X, T, Y, Z)
```
"""
function estimate_effect!(DRE::DoublyRobustLearner, X, T, Y, Z)
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
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = XLearner(X, T, Y)
stage1!(m1)
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
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
m1 = XLearner(X, T, Y)
stage1!(m1)
stage2!(m1)
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
