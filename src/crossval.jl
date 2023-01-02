"""
Methods to perform cross validation and find the optimum number of nodes

To reduce computation time, the number of nodes is optimized by using cross validation
to estimate the validation error on a subset of the range of possible umbers of nodes.
Then, an Extreme Learning Machine is trained to predict validation loss from the given
cross validation sets. Finally, the number of nodes is selected that has the smallest
predicted loss or the highest classification metric.
"""
module CrossValidation

using Random: shuffle
using CausalELM.Models: ExtremeLearner, RegularizedExtremeLearner, fit!, predict
using CausalELM.ActivationFunctions: relu

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
3
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
    idx = shuffle(1:n)
    train_size = @fastmath n*(folds-1)/folds

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
    n = size(Y, 1)
    @assert folds <= n && folds > iteration
    last_idx = floor(Int, (iteration/folds)*n)

    return X[1:last_idx, :], Y[1:last_idx], X[last_idx+1:end, :], Y[last_idx+1:end]
end

function onestep(X::Array{Float64}, Y::Array{Float64}, nodes::Integer, metric::Function, 
    iteration...; activation::Function=relu,  regularized::Bool=true, folds::Integer=5)

    xtrain, ytrain, xtest, ytest = traintest(X, Y, folds, iteration...)

    if regularized
        network = RegularizedExtremeLearner(xtrain, ytrain, nodes, activation)
    else
        network = ExtremeLearner(xtrain, ytrain, nodes, activation)
    end

    fit!(network)
    predictions = predict(network, xtest)

    return xtrain
end

function setnodes(X::Array{Float64}, Y::Array{Float64}, metric::Function, task::String,
    activation::Function=relu, min_nodes::Integer=1, max_nodes::Integer=100, regularized::Bool=true, 
    folds::Integer=5, temporal::Bool=false, iterations::Integer=round(size(X)/10),
    approximator_nodes=size(X, 1)/size(X, 2))
    
    pred_metrics = Vector{Float64}(undef, max_nodes-min_nodes)

    # Cross validation loop    
    
    
    # Approximate error function using validation error from cross validation
    approximator = ExtremeLearner(X, Y, approximator_nodes, relu)
    fit!(approximator)
    pred_metrics = predict(Float64[min_nodes:max_nodes;], avg_per_it)

    if task == "classification"
        return argmax([pred_metrics])
    else
        return argmin([pred_metrics])
    end
end
end