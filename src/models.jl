using LinearAlgebra: pinv, I, norm, tr

"""Abstract type that includes vanilla and L2 regularized Extreme Learning Machines"""
abstract type ExtremeLearningMachine end

"""
    ExtremeLearner(X, Y, hidden_neurons, activation)

Construct an ExtremeLearner for fitting and prediction.

# Notes
While it is possible to use an ExtremeLearner for regression, it is recommended to use 
RegularizedExtremeLearner, which imposes an L2 penalty, to reduce multicollinearity.

# References
For more details see: 
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

See also ['RegularizedExtremeLearner'](@ref).

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = ExtremeLearner(x, y, 10, σ)
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

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = RegularizedExtremeLearner(x, y, 10, σ)
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

# References
For more details see: 
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = ExtremeLearner(x, y, 10, σ)
```
"""
function fit!(model::ExtremeLearner)
    set_weights_biases(model)

    model.__fit = true
    model.β = @fastmath pinv(model.H) * model.Y
    return model.β
end

"""
    fit!(model)

Fit a Regularized Extreme Learner.

# References
For more details see: 
    Li, Guoqiang, and Peifeng Niu. "An enhanced extreme learning machine based on ridge 
    regression for regression." Neural Computing and Applications 22, no. 3 (2013): 
    803-810.

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = RegularizedExtremeLearner(x, y, 10, σ)
f1 = fit!(m1)
```
"""
function fit!(model::RegularizedExtremeLearner)
    set_weights_biases(model)
    k = ridge_constant(model)
    Id = Matrix(I, size(model.H, 2), size(model.H, 2))

    model.β = @fastmath pinv(transpose(model.H)*model.H + k*Id)*transpose(model.H)*model.Y

    model.__fit = true  # Enables running predict

    return model.β
end

"""
    predict(model, X)

Use an ExtremeLearningMachine to make predictions.

# References
For more details see: 
    Huang G-B, Zhu Q-Y, Siew C. Extreme learning machine: theory and applications. 
    Neurocomputing. 2006;70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = ExtremeLearner(x, y, 10, σ)
f1 = fit(m1, sigmoid)
julia> predict(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
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

# Notes
This should be run with the observed covariates. To use synthtic data for what-if scenarios 
use predict.

See also [`predict`](@ref).

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = ExtremeLearner(x, y, 10, σ)
f1 = fit(m1, sigmoid)
predict_counterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
```
"""
function predict_counterfactual!(model::ExtremeLearningMachine, X)
    model.counterfactual, model.__estimated = predict(model, X), true
    
    return model.counterfactual
end

"""
    placebo_test(model)

Conduct a placebo test.

# Notes
This method makes predictions for the post-event or post-treatment period using data 
in the pre-event or pre-treatment period and the post-event or post-treament. If there
is a statistically significant difference between these predictions the study design may
be flawed. Due to the multitude of significance tests for time series data, this function
returns the predictions but does not test for statistical significance.

# Examples
```julia
x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
m1 = ExtremeLearner(x, y, 10, σ)
f1 = fit(m1, sigmoid)
predict_counterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
placebo_test(m1)
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
    ridge_constant(model, [,iterations])

Calculate the L2 penalty for a regularized extreme learning machine using generalized cross 
validation with successive halving.

# Arguments
- `model::RegularizedExtremeLearner`: a regularized extreme learning machine
- `iterations::Int`: the number of iterations to perform for successive halving.

# References
For more information see: 
    Golub, Gene H., Michael Heath, and Grace Wahba. "Generalized cross-validation as a 
    method for choosing a good ridge parameter." Technometrics 21, no. 2 (1979): 215-223.

# Examples
```julia
m1 = RegularizedExtremeLearner(x, y, 10, σ)
ridge_constant(m1)
ridge_constant(m1, iterations=20)
```
"""
function ridge_constant(model::RegularizedExtremeLearner, iterations::Int=10)
    S(λ, X, X̂, n) =  X * pinv(X̂ .+ (n * λ * Matrix(I, n, n))) * transpose(X)
    set_weights_biases(model)
    Ĥ = transpose(model.H) * model.H

    function gcv(H, Y, λ)  # Estimates the generalized cross validation function for given λ
        S̃, n = S(λ, H, Ĥ, size(H, 2)), size(H, 1)
        return ((norm((ones(n) .- S̃) * Y)^2) / n) / ((tr(Matrix(I, n, n) .- S̃) / n)^2)
    end

    k₁, k₂, Λ = 1e-9, 1 - 1e-9, sum((1e-9, 1 - 1e-9)) / 2  # Initial window to search
    for i in 1:iterations
        gcv₁, gcv₂ = @fastmath gcv(model.H, model.Y, k₁), gcv(model.H, model.Y, k₂)

        # Divide the search space in half
        if gcv₁ < gcv₂
            k₂ /= 2
        elseif gcv₁ > gcv₂
            k₁ *= 2
        elseif gcv₁ ≈ gcv₂
            return (k₁ + k₂) / 2  # Early stopping
        end

        Λ = (k₁ + k₂) / 2
    end
    return Λ
end

"""
    set_weights_biases(model)

Calculate the weights and biases for an extreme learning machine or regularized extreme 
learning machine.

# References
For details see;
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

# Examples
```julia
m1 = RegularizedExtremeLearner(x, y, 10, σ)
set_weights_biases(m1)
```
"""
function set_weights_biases(model::ExtremeLearningMachine)
    a, b = -length(model.X), length(model.X)
    model.weights = (b - a) * rand(model.features, model.hidden_neurons) .+ a
    model.weights .+= (b - a) * rand(model.features) .+ a

    model.H = @fastmath model.activation((model.X * model.weights))
end

Base.show(io::IO, model::ExtremeLearner) = print(io, 
    "Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons")

Base.show(io::IO, model::RegularizedExtremeLearner) = print(io, 
    "Regularized Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons")
