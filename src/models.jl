"""
Base models to perform extreme learning with and without L2 penalization.

For details on Extreme learning machines see;
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

For details on Extreme learning machines with an L2 penalty see:
    Li, Guoqiang, and Peifeng Niu. "An enhanced extreme learning machine based on ridge 
    regression for regression." Neural Computing and Applications 22, no. 3 (2013): 
    803-810.
"""
module Models

using LinearAlgebra: pinv

"""Abstract type that includes vanilla and L2 regularized Extreme Learning Machines"""
abstract type ExtremeLearningMachine end

"""Struct to hold data for an Extreme Learning machine"""
mutable struct ExtremeLearner <: ExtremeLearningMachine
    """Training features"""
    X::Array{Float64}
    """Training outcome data, which may be continuous or discrete"""
    Y::Array{Float64}
    """Number of training samples"""
    training_samples::Int64
    """Number of features used in training"""
    features::Int64
    """Number of hidden neurons"""
    hidden_neurons::Int64
    """Activation function to be used"""
    activation::Function
    __fit::Bool             # Whether fit! has been called
    __estimated::Bool       # Whether a counterfactual has been predicted
    """Random weights used in the model"""
    weights::Array{Float64}
    """Random bias used in the model"""
    bias::Array{Float64}
    """Estimated coefficients"""
    β::Array{Float64}
    """Output from hidden neurons"""
    H::Array{Float64}
    """Predicted counterfactual data"""
    counterfactual::Array{Float64}
    __tol::Float64

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
```julia-repl
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
    function ExtremeLearner(X, Y, hidden_neurons, activation)
        new(X, Y, size(X, 1), size(X, 2), hidden_neurons, activation, false, false)
    end
end

"""Struct to hold data for a regularized Extreme Learning Machine"""
mutable struct RegularizedExtremeLearner <: ExtremeLearningMachine
    """Training Features"""
    X::Array{Float64}
    """Training outcome data, which may be continuous or discrete"""
    Y::Array{Float64}
    """Number of training samples"""
    training_samples::Int64
    """Number of features used in training"""
    features::Int64
    """Number of hidden neurons"""
    hidden_neurons::Int64
    """Activation function to be used"""
    activation::Function
    __fit::Bool             # Whether fit! has been called
    __estimated::Bool       # Whether a counterfactual has been estimated
    """Random weights used in the model"""
    weights::Array{Float64}
    """Random bias used in the model"""
    bias::Array{Float64}
    """Estimated coefficients"""
    β::Array{Float64}
    """L2 penalty term"""
    k::Float64
    """Output from hidden nodes"""
    H::Array{Float64}
    """Predicted counterfactual data"""
    counterfactual::Array{Float64}
    __tol::Float64
    
"""
    RegularizedExtremeLearner(X, Y, hidden_neurons, activation)

Construct a RegularizedExtremeLearner for fitting and prediction.

For more details see: 
    Li, Guoqiang, and Peifeng Niu. "An enhanced extreme learning machine based on ridge 
    regression for regression." Neural Computing and Applications 22, no. 3 (2013): 
    803-810.

Examples
```julia-repl
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
```julia-repl
julia> m1 = ExtremeLearner(x, y, 10, σ)
 Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit!(m1)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 ```
 """
function fit!(model::ExtremeLearner)
    setweightsbiases(model)

    model.__tol = sqrt(eps(real(float(one(eltype(model.H))))))  # For numerical stability

    model.__fit, model.β = true, @fastmath pinv(model.H, rtol=model.__tol) * model.Y

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
```julia-repl
julia> m1 = RegularizedExtremeLearner(x, y, 10, σ)
 Regularized Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit!(m1)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 ```
 """
function fit!(model::RegularizedExtremeLearner)
    setweightsbiases(model)
    
    model.__tol = sqrt(eps(real(float(one(eltype(model.H))))))

    I, k = ones(Float64, size(model.H, 2), size(model.H, 2)), ridgeconstant(model)   # L2

    model.β = @fastmath (pinv((transpose(model.H) * model.H) + ((1.0/k) * I), 
        rtol=model.__tol) * (transpose(model.H) * model.Y))

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
```julia-repl
julia> m1 = ExtremeLearner(x, y, 10, σ)
 Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit(m1, sigmoid)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 julia> predict(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
 [9.811656638113011e-16, 0.9999999999999962, -9.020553785284482e-17, 0.9999999999999978]
 ```
 """
function predict(model::ExtremeLearningMachine, X::Array) 
    if !model.__fit
        throw(ErrorException("run fit! before calling predict"))
    end

    return @fastmath model.activation(X * model.weights .+ model.bias) * model.β
end

"""
    predictcounterfactual(model, X)

Use an ExtremeLearningMachine to predict the counterfactual.

This should be run with the observed covariates. To use synthtic data for what-if 
    scenarios use predict.

See also [`predict`](@ref).

Examples
```julia-repl
julia> m1 = ExtremeLearner(x, y, 10, σ)
 Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit(m1, sigmoid)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 julia> predictcounterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
 [9.811656638113011e-16, 0.9999999999999962, -9.020553785284482e-17, 0.9999999999999978]
 ```
 """
function predictcounterfactual!(model::ExtremeLearningMachine, X::Array)
    model.counterfactual, model.__estimated = predict(model, X), true
    
    return model.counterfactual
end

"""
    placebotest(model)

Conduct a placebo test.

This method makes predictions for the post-event or post-treatment period using data 
in the pre-event or pre-treatment period and the post-event or post-treament. If there
is a statistically significant difference between these predictions the study design may
be flawed. Due to the multitude of significance tests for time series data, this function
returns the predictions but does not test for statistical significance.

Examples
```julia-repl
julia> m1 = ExtremeLearner(x, y, 10, σ)
 Extreme Learning Machine with 10 hidden neurons
 julia> f1 = fit(m1, sigmoid)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 julia> predictcounterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
 [9.811656638113011e-16, 0.9999999999999962, -9.020553785284482e-17, 0.9999999999999978]
 julia> placebotest(m1)
 ([9.811656638113011e-16, 0.9999999999999962, -9.020553785284482e-17, 0.9999999999999978],
 [0.5, 0.4, 0.3, 0.2])
 ```
 """
function placebotest(model::ExtremeLearningMachine)
    m = "Use predictcounterfactual to estimate a counterfactual before calling placebotest"
    if !model.__estimated
        throw(ErrorException(m))
    end
    return predict(model, model.X), model.counterfactual
end

function ridgeconstant(model::RegularizedExtremeLearner)
    β0 = @fastmath pinv(model.H) * model.Y
    σ̃  = @fastmath ((transpose(model.Y .- (model.H * β0)) * (model.Y .- (model.H * β0))) / 
        (model.features - size(model.H)[2]))

    return @fastmath first((model.H[2]*σ̃)/(transpose(β0)*transpose(model.H)*model.H*β0))
end

function setweightsbiases(model::ExtremeLearningMachine)
    model.weights = rand(Float64, model.features, model.hidden_neurons)
    model.bias = rand(Float64, 1, model.hidden_neurons)

    model.H = @fastmath model.activation((model.X * model.weights) .+ model.bias)
end

Base.show(io::IO, model::ExtremeLearner) = print(io, 
    "Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons")

Base.show(io::IO, model::RegularizedExtremeLearner) = print(io, 
    "Regularized Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons")

end
