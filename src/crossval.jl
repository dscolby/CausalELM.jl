"""
Methods to perform cross validation and find the optimum number of neurons.

To reduce computation time, the number of neurons is optimized by using cross validation
to estimate the validation error on a small subset of the range of possible numbers of 
neurons. Then, an Extreme Learning Machine is trained to predict validation loss from 
the given cross validation sets. Finally, the number of neurons is selected that has the 
smallest predicted loss or the highest classification metric.
"""
module CrossValidation

using Random: shuffle

include("models.jl")
using .Models: ExtremeLearner, RegularizedExtremeLearner, fit!, predict

include("activation.jl")
using .ActivationFunctions: relu

"""
    recode(ŷ)

Round predicted values to their predicted class for classification tasks.

If the smallest predicted label is 0, all labels are shifted up 1; if the smallest 
label is -1, all labels are shifted up 2. Also labels cannot be smaller than -1.

Examples
```julia-repl
julia> recode([-0.7, 0.2, 1.1])
3-element Vector{Float64}
1
2
3
julia> recode([0.1, 0.2, 0.3])
3-element Vector{Float64}
1
1
1
julia> recode([1.1, 1.51, 1.8])
3-element Vector{Float64}
1
2
2
```
"""
@inline function recode(ŷ::Array{Float64})
    rounded = round.(ŷ)
    if minimum(rounded) < 0
        rounded .+= 2
    elseif minimum(rounded) == 0
        rounded .+= 1
    else
    end
    return rounded
end

"""
    traintest(X, Y, folds)

Create a train-test split.

If an iteration is specified, the train test split will be treated as time series/panel
data.

Examples
```julia-repl
julia> xtrain, ytrain, xtest, ytest = traintest(zeros(20, 2), zeros(20), 5)
```
"""
function traintest(X::Array{Float64}, Y::Array{Float64}, folds::Int64)
    n = size(Y, 1)
    @assert folds <= n

    idx, train_size = shuffle(1:n), @fastmath n*(folds-1)/folds

    train_idx, test_idx = view(idx, 1:floor(Int, train_size)), 
        view(idx, (floor(Int, train_size)+1):n)

    xtrain, ytrain, xtest, ytest = X[train_idx, :], Y[train_idx, :], 
        X[test_idx, :], Y[test_idx, :]
        
    return xtrain, ytrain, xtest, ytest
end

"""
    traintest(X, Y, folds, iteration)

Create a rolling train-test split for time series/panel data.

An iteration should not be specified for non-time series/panel data.

Examples
```julia-repl
julia> xtrain, ytrain, xtest, ytest = traintest(zeros(20, 2), zeros(20), 5, 1)
```
"""
function traintest(X::Array{Float64}, Y::Vector{Float64}, folds::Int64, iteration::Integer)
    n = length(Y)
    @assert folds <= n && folds > iteration

    last_idx = floor(Int, (iteration/folds)*n)

    return X[1:last_idx, :], Y[1:last_idx], X[last_idx+1:end, :], Y[last_idx+1:end]
end

"""
    validate(X, Y, nodes, metric, iteration...; activation, regularized, folds)

Calculate a validation metric for a single fold in k-fold cross validation.

Examples
```julia-repl
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> validate(x, y, 5, accuracy, 3)
0.0
```
"""
function validate(X::Array{Float64}, Y::Array{Float64}, nodes::Integer, metric::Function, 
    iteration...; activation::Function=relu,  regularized::Bool=true, folds::Integer=5)

    xtrain, ytrain, xtest, ytest = traintest(X, Y, folds, iteration...)

    if regularized
        network = RegularizedExtremeLearner(xtrain, ytrain, nodes, activation)
    else
        network = ExtremeLearner(xtrain, ytrain, nodes, activation)
    end

    fit!(network)
    predictions = predict(network, xtest)

    return metric(ytest[1, :], predictions[1, :])
end

"""
    crossvalidate(X, Y, neurons, metric, activation, regularized, folds)

Calculate a validation metric for k folds using a single set of hyperparameters.

Examples
```julia-repl
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> crossvalidate(x, y, 5, accuracy)
0.0257841765251021
```
"""
function crossvalidate(X::Array{Float64}, Y::Array{Float64}, neurons::Integer, 
    metric::Function, activation::Function=relu, regularized::Bool=true, folds::Integer=5, 
    temporal::Bool=false)
    mean_metric = 0.0

    folds = ifelse(temporal, folds+1, folds)

    @inbounds for fold in 1:folds
        
        # For time series or panel data
        if temporal && fold < folds
            mean_metric += validate(X, Y, neurons, metric, fold, activation=activation, 
                regularized=regularized, folds=folds)
        else
            mean_metric += validate(X, Y, neurons, metric, activation=activation, 
                regularized=regularized, folds=folds)
        end
    end
    return mean_metric/folds
end

"""
    bestsize(X, Y, metric, task, activation, min_neurons, max_neurons, regularized, folds, temporal, 
        iterations, approximator_neurons)

Compute the best number of neurons for an Extreme Learning Machine.

The procedure tests networks with numbers of neurons in a sequence whose length is given 
by iterations on the interval [min_neurons, max_neurons]. Then, it uses the networks 
sizes and validation errors from the sequence to predict the validation error or metric 
for every network size between min_neurons and max_neurons using the function 
approximation ability of an Extreme Learning Machine. Finally, it returns the network 
size with the best predicted validation error or metric.

Examples
```julia-repl
julia> bestsize(rand(100, 5), rand(100), mse, "regression")
11
```
"""
function bestsize(X::Array{Float64}, Y::Array{Float64}, metric::Function, task::String,
    activation::Function=relu, min_neurons::Integer=1, max_neurons::Integer=100, 
    regularized::Bool=true, folds::Integer=5, temporal::Bool=false, 
    iterations::Integer=Int(round(size(X, 1)/10)), 
    approximator_neurons=Integer=Int(round(size(X, 1)/10)))
    
    act, loops = Vector{Float64}(undef, iterations), 
        round.(Int, collect(range(min_neurons, max_neurons, length=iterations)))
   
    @inbounds for (i, n) in enumerate(loops)
        if temporal
            act[i] = crossvalidate(X, Y, round(Int, n), metric, activation, regularized, 
                folds, true)
        else
            act[i] = crossvalidate(X, Y, round(Int, n), metric, activation, regularized, 
                folds)
        end
    end
    
    # Approximate error function using validation error from cross validation
    approximator = ExtremeLearner(reshape(loops, :, 1), reshape(act, :, 1), 
        approximator_neurons, relu)
        
    fit!(approximator)
    pred_metrics = predict(approximator, Float64[min_neurons:max_neurons;])

    return ifelse(startswith(task, "c"), argmax([pred_metrics]), argmin([pred_metrics]))
end
end
