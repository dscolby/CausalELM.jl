"""
Methods to perform cross validation and find the optimum number of neurons.

To reduce computation time, the number of neurons is optimized by using cross validation
to estimate the validation error on a small subset of the range of possible numbers of 
neurons. Then, an Extreme Learning Machine is trained to predict validation loss from 
the given cross validation sets. Finally, the number of neurons is selected that has the 
smallest predicted loss or the highest classification metric.
"""
module CrossValidation

using ..ActivationFunctions: relu
using Random: randperm
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
function generatefolds(X::Array{Float64}, Y::Array{Float64}, folds::Int64)
    msg = """the number of folds must be less than the number of 
             observations and greater than or equal to iteration"""
    n = length(Y)
    
    if folds >= n throw(ArgumentError(msg)) end

    fold_setx = Array{Array{Float64, 2}}(undef, folds)
    fold_sety = Array{Array{Float64, 1}}(undef, folds)

    # Indices to start and stop for each fold
    stops = round.(Int, range(start=1, stop=n, length=folds+1))

    # Indices to use for making folds
    indices = [s:e-(e < n)*1 for (s, e) in zip(stops[1:end-1], stops[2:end])]

    for (i, idx) in enumerate(indices)
        fold_setx[i], fold_sety[i] = X[idx, :], Y[idx]
    end

    return fold_setx, fold_sety
end

"""
    validatefold(xtrain, ytrain, xtest, ytest, nodes, metric; activation, regularized)

Calculate a validation metric for a single fold in k-fold cross validation.

Examples
```julia-repl
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> validatefold(x, y, 5, accuracy, 3)
0.0
```
"""
function validatefold(xtrain::Array{Float64}, ytrain::Array{Float64}, xtest::Array{Float64}, 
    ytest::Array{Float64}, nodes::Integer, metric::Function; activation::Function=relu,  
    regularized::Bool=true)

    if regularized
        network = RegularizedExtremeLearner(xtrain, ytrain, nodes, activation)
    else
        network = ExtremeLearner(xtrain, ytrain, nodes, activation)
    end

    fit!(network)
    predictions = predict(network, xtest)

    return metric(recode(ytest[1, :]), recode(predictions[1, :]))
end

"""
    crossvalidate(X, Y, neurons, metric, activation, regularized, folds, temporal)

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

    if temporal
        indices = reduce(vcat, (collect(1:5:size(X, 1)), size(X, 1)))
        x_folds = [X[i:j, :] for (i, j) in zip(indices, indices[2:end] .- 1)]
        y_folds = [Y[i:j] for (i, j) in zip(indices, indices[2:end] .- 1)]
    else
        x_folds, y_folds = generatefolds(X, Y, folds)
    end
    
    @inbounds for fold in 1:folds
        xtrain = reduce(vcat, [x_folds[f] for f in 1:folds if f != fold])
        ytrain = reduce(vcat, [y_folds[f] for f in 1:folds if f != fold])
        xtest, ytest = x_folds[fold], y_folds[fold]

        mean_metric += validatefold(xtrain, ytrain, xtest, ytest, neurons, metric, 
            activation=activation, regularized=regularized)
    end
    return mean_metric/folds
end

"""
    bestsize(X, Y, metric, task, activation, min_neurons, max_neurons, regularized, folds, 
        temporal, iterations, approximator_neurons)

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
        round.(Int, range(min_neurons, max_neurons, length=iterations))
   
    @inbounds for (i, n) in pairs(loops)
        act[i] = crossvalidate(X, Y, round(Int, n), metric, activation, regularized, folds, 
            temporal)
    end
    
    # Approximates the error function using validation error from cross validation
    approximator = ExtremeLearner(reshape(loops, :, 1), reshape(act, :, 1), 
        approximator_neurons, relu)
        
    fit!(approximator)
    pred_metrics = predict(approximator, Float64[min_neurons:max_neurons;])

    return ifelse(startswith(task, "c"), argmax([pred_metrics]), argmin([pred_metrics]))
end

"""
    shuffledata(X, Y, T)

Shuffles covariates, treatment vector, and outcome vector for cross validation.

Examples
```julia-repl
julia> x, y, t = rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> shuffledata(x, y, t)
([0.6124923085225416 0.2713900065807924 … 0.6094796972512194 0.6067966603192685; 
0.7186612932571539 0.8047878363606299 … 0.9787878554455594 0.885819212905816; … ; 
0.773543733306263 0.10880091279797399 … 0.10525512055751185 0.6303472234021711; 
0.10845217539341823 0.9911071602976902 … 0.014754069216096566 0.5256103389041187], 
[0.4302689295553531, 0.2396683446618325, 0.7954433314513768, 0.7191098533903124, 
0.8168563428651753, 0.7064320936729905, 0.048113106979693065, 0.3102938851371281, 
0.6246380539228858, 0.3510284321966193  …  0.5324022501182528, 0.8354720951777901, 
0.7526652774981095, 0.3639742621882005, 0.21030903031988923, 0.6936212944871928, 
0.3910592143534404, 0.15152013651215301, 0.38891692138831735, 0.08827711410802941], 
Float64[0, 0, 1, 1, 0, 1, 0, 0, 1, 0  …  0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
```
"""
function shuffledata(X::Array{Float64}, Y::Array{Float64}, T::Array{Float64})
        idx = randperm(size(X, 1))
        new_data = mapslices.(x->x[idx], [X, Y, T], dims=1)
        X, Y, T = new_data[1], new_data[2], Float64.(new_data[3])

        return X, vec(Y), vec(T)
end

end
