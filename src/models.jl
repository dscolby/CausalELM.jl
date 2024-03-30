using LinearAlgebra: inv, pinv, I

"""Abstract type that includes vanilla and L2 regularized Extreme Learning Machines"""
abstract type ExtremeLearningMachine end

"""
    ExtremeLearner(X, Y, hidden_neurons, activation)

Construct an ExtremeLearner for fitting and prediction.

While it is possible to use an ExtremeLearner for regression, it is recommended to use 
RegularizedExtremeLearner, which imposes an L2 penalty, to reduce multicollinearity.

For more details see: 
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

See also ['RegularizedExtremeLearner'](@ref).

Examples
```julia
julia> x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
4×2 Matrix{Float64}:
 1.0  1.0
 0.0 1.0
 0.0 0.0
 1.0 0.0
julia> y = [0.0, 1.0, 0.0, 1.0]
 4-element Vector{Int64}:
 0.0
 1.0
 0.0
 1.0
julia> m1 = ExtremeLearner(x, y, 10, σ)
Extreme Learning Machine with 10 hidden neurons
```
"""
mutable struct ExtremeLearner <: ExtremeLearningMachine
    X::Array{Float64}
    Y::Array{Float64}
    training_samples::Int64
    features::Int64
    hidden_neurons::Int64
    activation::Function
    __fit::Bool             
    __estimated::Bool       
    weights::Array{Float64}
    β::Array{Float64}
    H::Array{Float64}
    counterfactual::Array{Float64}

    function ExtremeLearner(X, Y, hidden_neurons, activation)
        new(X, Y, size(X, 1), size(X, 2), hidden_neurons, activation, false, false)
    end
end

"""
    RegularizedExtremeLearner(X, Y, hidden_neurons, activation)

Construct a RegularizedExtremeLearner for fitting and prediction.

For more details see: 
    Li, Guoqiang, and Peifeng Niu. "An enhanced extreme learning machine based on ridge 
    regression for regression." Neural Computing and Applications 22, no. 3 (2013): 
    803-810.

Examples
```julia
julia> x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
4×2 Matrix{Float64}:
 1.0  1.0
 0.0 1.0
 0.0 0.0
 1.0 0.0
julia> y = [0.0, 1.0, 0.0, 1.0]
4-element Vector{Int64}:
 0.0
 1.0
 0.0
 1.0
julia> m1 = RegularizedExtremeLearner(x, y, 10, σ)
Regularized Extreme Learning Machine with 10 hidden neurons
```
"""
mutable struct RegularizedExtremeLearner <: ExtremeLearningMachine
    X::Array{Float64}
    Y::Array{Float64}
    training_samples::Int64
    features::Int64
    hidden_neurons::Int64
    activation::Function
    __fit::Bool             
    __estimated::Bool       
    weights::Array{Float64}
    β::Array{Float64}
    k::Float64
    H::Array{Float64}
    counterfactual::Array{Float64}
    
    function RegularizedExtremeLearner(X, Y, hidden_neurons, activation)
        new(X, Y, size(X, 1), size(X, 2), hidden_neurons, activation, false, false)
    end
end

"""
    fit!(model)

Make predictions with an ExtremeLearner.

For more details see: 
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

Examples
```julia
julia> m1 = ExtremeLearner(x, y, 10, σ)
 Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit!(m1)
 10-element Vector{Float64}
 -4.403356409043448
 -5.577616954029608
 -2.1732800642523595
 ⋮
 -2.4741301876094655
 40.642730531608635
 -11.058942121275233
```
"""
function fit!(model::ExtremeLearner)
    set_weights_biases(model)

    model.__fit, model.β = true, @fastmath model.H\model.Y

    return model.β
end

"""
    fit!(model)

Fit a Regularized Extreme Learner.

For more details see: 
    Li, Guoqiang, and Peifeng Niu. "An enhanced extreme learning machine based on ridge 
    regression for regression." Neural Computing and Applications 22, no. 3 (2013): 
    803-810.

Examples
```julia
julia> m1 = RegularizedExtremeLearner(x, y, 10, σ)
Regularized Extreme Learning Machine with 10 hidden neurons
julia> f1 = fit!(m1)
10-element Vector{Float64}
 -4.403356409043448
 -5.577616954029608
 -2.1732800642523595
 ⋮
 -2.4741301876094655
 40.642730531608635
 -11.058942121275233
```
"""
function fit!(model::RegularizedExtremeLearner)
    set_weights_biases(model)
    β0 = @fastmath pinv(model.H) * model.Y

    Id, k = Matrix(I, size(model.H, 2), size(model.H, 2)), ridge_constant(model)   # L2

    model.β = @fastmath (Id-k^2*inv(transpose(model.H)*model.H + k*Id)^2)*β0

    model.__fit = true  # Enables running predict

    return model.β
end

"""
    predict(model, X)

Use an ExtremeLearningMachine to make predictions.

For more details see: 
    Huang G-B, Zhu Q-Y, Siew C. Extreme learning machine: theory and applications. 
    Neurocomputing. 2006;70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126

Examples
```julia
julia> m1 = ExtremeLearner(x, y, 10, σ)
Extreme Learning Machine with 10 hidden neurons
julia> f1 = fit(m1, sigmoid)
10-element Vector{Float64}
 -4.403356409043448
 -5.577616954029608
 -2.1732800642523595
 ⋮
 -2.4741301876094655
 40.642730531608635
 -11.058942121275233
julia> predict(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
4-element Vector{Float64}
 9.811656638113011e-16
 0.9999999999999962
 -9.020553785284482e-17
 0.9999999999999978
```
"""
function predict(model::ExtremeLearningMachine, X) 
    if !model.__fit
        throw(ErrorException("run fit! before calling predict"))
    end

    return @fastmath model.activation(X * model.weights) * model.β
end

"""
    predictcounterfactual(model, X)

Use an ExtremeLearningMachine to predict the counterfactual.

This should be run with the observed covariates. To use synthtic data for what-if 
    scenarios use predict.

See also [`predict`](@ref).

Examples
```julia
julia> m1 = ExtremeLearner(x, y, 10, σ)
 Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit(m1, sigmoid)
 10-element Vector{Float64}
 -4.403356409043448
 -5.577616954029608
 -2.1732800642523595
 ⋮
 -2.4741301876094655
 40.642730531608635
 -11.058942121275233
julia> predict_counterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
4-element Vector{Float64}
 9.811656638113011e-16
 0.9999999999999962
 -9.020553785284482e-17
 0.9999999999999978
```
"""
function predict_counterfactual!(model::ExtremeLearningMachine, X)
    model.counterfactual, model.__estimated = predict(model, X), true
    
    return model.counterfactual
end

"""
    placebo_test(model)

Conduct a placebo test.

This method makes predictions for the post-event or post-treatment period using data 
in the pre-event or pre-treatment period and the post-event or post-treament. If there
is a statistically significant difference between these predictions the study design may
be flawed. Due to the multitude of significance tests for time series data, this function
returns the predictions but does not test for statistical significance.

Examples
```julia
julia> m1 = ExtremeLearner(x, y, 10, σ)
Extreme Learning Machine with 10 hidden neurons
julia> f1 = fit(m1, sigmoid)
10-element Vector{Float64}
 -4.403356409043448
 -5.577616954029608
 -2.1732800642523595
 ⋮
 -2.4741301876094655
 40.642730531608635
 -11.058942121275233
julia> predict_counterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
4-element Vector{Float64}
 9.811656638113011e-16
 0.9999999999999962
 -9.020553785284482e-17
 0.9999999999999978
julia> placebo_test(m1)
 ([9.811656638113011e-16, 0.9999999999999962, -9.020553785284482e-17, 0.9999999999999978],
 [0.5, 0.4, 0.3, 0.2])
```
"""
function placebo_test(model::ExtremeLearningMachine)
    m = "Use predict_counterfactual! to estimate a counterfactual before using placebo_test"
    if !model.__estimated
        throw(ErrorException(m))
    end
    return predict(model, model.X), model.counterfactual
end

"""
    ridge_constant(model)

Calculate the L2 penalty for a regularized extreme learning machine.

For more information see: 
    Li, Guoqiang, and Peifeng Niu. "An enhanced extreme learning machine based on ridge 
    regression for regression." Neural Computing and Applications 22, no. 3 (2013): 
    803-810.

Examples
```julia
julia> m1 = RegularizedExtremeLearner(x, y, 10, σ)
Extreme Learning Machine with 10 hidden neurons
julia> ridge_constant(m1)
 0.26789338524662887
```
"""
function ridge_constant(model::RegularizedExtremeLearner)
    β0, L, N = @fastmath pinv(model.H) * model.Y, size(model.H)[2], model.features
    σ̃ = @fastmath ((transpose(model.Y .- (model.H*β0))*(model.Y .- (model.H*β0)))/(N-L))

    return @fastmath first((L*σ̃)/(transpose(β0)*(transpose(model.H)*model.H)*β0))
end

"""
    set_weights_biases(model)

Calculate the weights and biases for an extreme learning machine or regularized extreme 
learning machine.

For details see;
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

Examples
```julia
julia> m1 = RegularizedExtremeLearner(x, y, 10, σ)
Extreme Learning Machine with 10 hidden neurons
julia> set_weights_biases(m1)
```
"""
function set_weights_biases(model::ExtremeLearningMachine)
    model.weights = rand(model.features, model.hidden_neurons)
    model.weights = reduce(hcat, (model.weights, repeat([rand()], size(model.weights, 1))))

    model.H = @fastmath model.activation((model.X * model.weights))
end

Base.show(io::IO, model::ExtremeLearner) = print(io, 
    "Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons")

Base.show(io::IO, model::RegularizedExtremeLearner) = print(io, 
    "Regularized Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons")
