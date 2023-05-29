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
using ..ActivationFunctions: relu
using ..Models: ExtremeLearner, RegularizedExtremeLearner, fit!, predict

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
    generatefolds(X, Y, folds)

Creates folds for cross validation.

Examples
```julia-repl
julia> xfolds, y_folds = generatefolds(zeros(20, 2), zeros(20), 5)
```
"""
function generatefolds(X::Array{Float64}, Y::Vector{Float64}, folds::Int64)
    msg = """the number of folds must be less than the number of 
             observations and greater than or equal to iteration"""
    n = length(Y)
    
    if folds >= n
        throw(ArgumentError(msg))
    end

    fold_setx = Array{Array{Float64, 2}}(undef, folds)
    fold_sety = Array{Array{Float64, 1}}(undef, folds)

    # Indices to start and stop
    stops = round.(Int, range(start=1, stop=n, length=folds+1))

    # Indices to use for making folds
    indices = [s:e-(e < n)*1 for (s, e) in zip(stops[1:end-1], stops[2:end])]

    for (i, idx) in enumerate(indices)
        fold_setx[i] = X[idx, :]
        fold_sety[i] = Y[idx]
    end

    return fold_setx, fold_sety
end

"""
    validate(xtrain, ytrain, xtest, ytest, nodes, metric; activation, regularized)

Calculate a validation metric for a single fold in k-fold cross validation.

Examples
```julia-repl
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> validate(x, y, 5, accuracy, 3)
0.0
```
"""
function validate(xtrain::Array{Float64}, ytrain::Array{Float64}, xtest::Array{Float64}, 
    ytest::Array{Float64}, nodes::Integer, metric::Function; activation::Function=relu,  
    regularized::Bool=true)

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
    metric::Function, activation::Function=relu, regularized::Bool=true, folds::Integer=5)

    mean_metric = 0.0
    x_folds, y_folds = generatefolds(X, Y, folds)
    
    @inbounds for fold in 1:folds
        training_size = sum([size(x_folds[f], 1) for f in 1:folds if f != fold])

        xtrain = reduce(vcat, [x_folds[f] for f in 1:folds if f != fold])
        ytrain = reduce(vcat, [y_folds[f] for f in 1:folds if f != fold])
        xtest, ytest = x_folds[fold], y_folds[fold]

        mean_metric += validate(xtrain, ytrain, xtest, ytest, neurons, metric, 
            activation=activation, regularized=regularized)
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
                folds)
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
