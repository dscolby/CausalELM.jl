module Models
using LinearAlgebra: pinv

"""
    Elm(X, Y, hidden_nodes, activation)

Construct an Elm object for fitting and predicting with an Extreme Learning Machine.

For more details see: 
    Huang G-B, Zhu Q-Y, Siew C. Extreme learning machine: theory and applications. 
    Neurocomputing. 2006;70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126

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
 julia> m1 = Elm(x, y, 10, σ)
 Extreme Learning machine with 10 hidden nodes
 """
mutable struct Elm
    X::Array
    Y::Array
    training_samples::Integer
    features::Integer
    hidden_nodes::Integer
    activation::Function
    __fit::Bool             # Whether fit! has been called
    weights::Array
    bias::Array
    β::Array
    function Elm(X, Y, hidden_nodes, activation)
        new(X, Y, size(X)[1], size(X)[2], hidden_nodes, activation, false)
    end
end

"""
    fit!(model, activation)

Make predictions with an Elm object.

For more details see: 
    Huang G-B, Zhu Q-Y, Siew C. Extreme learning machine: theory and applications. 
    Neurocomputing. 2006;70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126

Examples
```julia-repl
julia> m1 = Elm(x, y, 10, σ)
 Extreme Learning machine with 10 hidden nodes
 julia> f1 = fit(m1, sigmoid)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 """
function fit!(model::Elm)
    model.weights = rand(model.features, model.hidden_nodes)
    model.bias = rand(model.training_samples)
    weights_matrix = reduce(hcat, [model.X * model.weights, model.bias])
    
    H = model.activation(weights_matrix)

    model.β = pinv(H) * model.Y

    model.__fit = true  # Enables running predict

    return model.β
end

"""
    predict(model, X)

Train an Extreme Learning machine.

For more details see: 
    Huang G-B, Zhu Q-Y, Siew C. Extreme learning machine: theory and applications. 
    Neurocomputing. 2006;70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126

Examples
```julia-repl
julia> m1 = Elm(x, y, 10, σ)
 Extreme Learning machine with 10 hidden nodes
 julia> f1 = fit(m1, sigmoid)
 [-4.403356409043448, -5.577616954029608, -2.1732800642523595, 0.9669137012255704, 
 -3.6474913410560013, -4.206228346376102, -7.575391282978456, 4.528774205936467, 
 -2.4741301876094655, 40.642730531608635, -11.058942121275233]
 julia> predict(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
 [9.811656638113011e-16, 0.9999999999999962, -9.020553785284482e-17, 0.9999999999999978]
 """
function predict(model::Elm, X::Array) 
    @assert model.__fit "Run fit! before calling predict"

    weights_matrix = reduce(hcat, [X * model.weights, model.bias])

    return model.activation(weights_matrix) * model.β
end

Base.show(io::IO, model::Elm) = print(io, "Extreme Learning machine with ", 
    model.hidden_nodes, " hidden nodes")

end
