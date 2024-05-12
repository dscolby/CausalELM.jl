using Random: randperm

"""
    generate_folds(X, Y, folds)

Create folds for cross validation.

# Examples
```julia
xfolds, y_folds = generate_folds(zeros(20, 2), zeros(20), 5)
```
"""
function generate_folds(X, Y, folds)
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
    generate_folds(X, Y, folds)

Create rolling folds for cross validation of time series data.

# Examples
```julia
xfolds, y_folds = generate_temporal_folds(zeros(20, 2), zeros(20), 5, temporal=true)
```
"""
function generate_temporal_folds(X, Y, folds=5)
    msg = """the number of folds must be less than the number of 
             observations and greater than or equal to iteration"""
    n = length(Y)
    
    # Make sure there aren't more folds than observations
    if folds >= n throw(ArgumentError(msg)) end

    # The indices are evely spaced and start at the top to make rolling splits for TS data
    indices = Int.(floor.(collect(range(1, size(X, 1), folds+1))))
    x_folds, y_folds = [X[1:i, :] for i in indices[2:end]], [Y[1:i] for i in indices[2:end]]

    return x_folds, y_folds
end

"""
    validation_loss(xtrain, ytrain, xtest, ytest, nodes, metric; kwargs...)

Calculate a validation metric for a single fold in k-fold cross validation.

# Arguments
- `xtrain::Any`: an array of features to train on.
- `ytrain::Any`: an array of training labels.
- `xtest::Any`: an array of features to test on.
- `ytrain::Any`: an array of testing labels.
- `nodes::Int`: the number of neurons in the extreme learning machine.
- `metric::Function`: the validation metric to calculate.

# Keywords
- `activation::Function=relu`: the activation function to use.
- `regularized::Function=true`: whether to use L2 regularization.

# Examples
```julia
x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
validation_loss(x, y, 5, accuracy, 3)
```
"""
function validation_loss(xtrain, ytrain, xtest, ytest, nodes, metric; activation=relu, 
                         regularized=true)

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
    cross_validate(X, Y, neurons, metric, activation, regularized, folds, temporal)

Calculate a validation metric for k folds using a single set of hyperparameters.

# Arguments
- `X::Array`: an array of features to train on.
- `Y::Vector`: a vector of labels to train on.
- `neurons::Int`: the number of neurons to use in the extreme learning machine.
- `metric::Function`: the validation metric to calculate.
- `activation::Function=relu`: the activation function to use.
- `regularized::Function=true`: whether to use L2 regularization
- `folds::Int`: the number of folds to use for cross validation.
- `temporal::Function=true`: whether the data is of a time series or panel nature.

# Examples
```julia
x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
cross_validate(x, y, 5, accuracy)
```
"""
function cross_validate(X, Y, neurons, metric, activation, regularized, folds, temporal)

    mean_metric = 0.0
    xfs, yfs = temporal ? generate_temporal_folds(X, Y, folds) : generate_folds(X, Y, folds)
    
    @inbounds for fold in 1:folds
        if !temporal
            xtr = reduce(vcat, [xfs[f] for f in 1:folds if f != fold])
            ytr = reduce(vcat, [yfs[f] for f in 1:folds if f != fold])
            xtst, ytst = xfs[fold], yfs[fold]
        # The last fold can't be used to training since it will leave nothing to predict
        elseif temporal && fold < folds
            xtr, ytr = reduce(vcat, xfs[1:fold]), reduce(vcat, yfs[1:fold])
            xtst, ytst = reduce(vcat, xfs[fold+1:end]), reduce(vcat, yfs[fold+1:end])
        else
            continue
        end

        mean_metric += validation_loss(xtr, ytr, xtst, ytst, neurons, metric, 
                                       activation=activation, regularized=regularized)
    end
    return mean_metric/folds
end

"""
    best_size(X, Y, metric, task, activation, min_neurons, max_neurons, regularized, folds, 
              temporal, iterations, elm_size)

Compute the best number of neurons for an Extreme Learning Machine.

# Notes
The procedure tests networks with numbers of neurons in a sequence whose length is given 
by iterations on the interval [min_neurons, max_neurons]. Then, it uses the networks 
sizes and validation errors from the sequence to predict the validation error or metric 
for every network size between min_neurons and max_neurons using the function 
approximation ability of an Extreme Learning Machine. Finally, it returns the network 
size with the best predicted validation error or metric.

# Arguments
- `X::Array`: an array of features to train on.
- `Y::Vector`: a vector of labels to train on.
- `metric::Function`: the validation metric to calculate.
- `task::String`: either regression or classification.
- `activation::Function=relu`: the activation function to use.
- `min_neurons::Int`: the minimum number of neurons to consider for the extreme learner.
- `max_neurons::Int`: the maximum number of neurons to consider for the extreme learner.
- `regularized::Function=true`: whether to use L2 regularization
- `folds::Int`: the number of folds to use for cross validation.
- `temporal::Function=true`: whether the data is of a time series or panel nature.
- `iterations::Int`: the number of iterations to perform cross validation between 
    min_neurons and max_neurons.
- `elm_size::Int`: the number of nuerons in the validation loss approximator network.

# Examples
```julia
best_size(rand(100, 5), rand(100), mse, "regression")
```
"""
function best_size(X, Y, metric, task, activation, min_neurons, max_neurons, regularized, 
                   folds, temporal, iterations, elm_size)
    
    loss = Vector{Float64}(undef, iterations) 
    num_neurons = round.(Int, range(min_neurons, max_neurons, length=iterations))
   
    # Use cross validation to calculate the validation loss for each number of neurons in 
    # the interval of min_neurons to max_neurons spaced evenly in steps of iterations
    @inbounds for (idx, potential_neurons) in pairs(num_neurons)
        loss[idx] = cross_validate(X, Y, round(Int, potential_neurons), metric, activation, 
                                   regularized, folds, temporal)
    end
    
    # Use an extreme learning machine to learn a mapping from number of neurons to 
    # validation error
    mapper = ExtremeLearner(reshape(num_neurons, :, 1), reshape(loss, :, 1), elm_size, relu)
    fit!(mapper)
    pred_metrics = predict(mapper, Float64[min_neurons:max_neurons;])

    return ifelse(startswith(task, "c"), argmax([pred_metrics]), argmin([pred_metrics]))
end

"""
    shuffle_data(X, Y)

Shuffles covariates and outcome vector for cross validation.

# Examples
```julia
x, y, t = rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
shuffle_data(x, y, t)
```
"""
function shuffle_data(X, Y)
        idx = randperm(size(X, 1))
        new_data = mapslices.(x->x[idx], [X, Y], dims=1)
        X, Y = new_data

        return Array(X), vec(Y)
end
