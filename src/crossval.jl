using Random: randperm

"""
    generate_folds(X, Y, folds)

Create folds for cross validation.

# Examples
```jldoctest
julia> xfolds, y_folds = CausalELM.generate_folds(zeros(4, 2), zeros(4), 2)
([[0.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 0.0]], [[0.0], [0.0, 0.0, 0.0]])
```
"""
function generate_folds(X, Y, folds)
    msg = """the number of folds must be less than the number of observations"""
    n = length(Y)

    if folds >= n
        throw(ArgumentError(msg))
    end

    fold_setx = Array{Array{Float64,2}}(undef, folds)
    fold_sety = Array{Array{Float64,1}}(undef, folds)

    # Indices to start and stop for each fold
    stops = round.(Int, range(; start=1, stop=n, length=folds + 1))

    # Indices to use for making folds
    indices = [s:(e - (e < n) * 1) for (s, e) in zip(stops[1:(end - 1)], stops[2:end])]

    for (i, idx) in enumerate(indices)
        fold_setx[i], fold_sety[i] = X[idx, :], Y[idx]
    end

    return fold_setx, fold_sety
end

"""
    generate_temporal_folds(X, Y, folds)

Create rolling folds for cross validation of time series data.

# Examples
```jldoctest
julia> xfolds, yfolds = CausalELM.generate_temporal_folds([1 1; 1 1; 0 0; 0 0], zeros(4), 2)
([[1 1; 1 1], [1 1; 1 1; 0 0; 0 0]], [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
```
"""
function generate_temporal_folds(X, Y, folds=5)
    msg = """the number of folds must be less than the number of 
             observations and greater than or equal to iteration"""
    n = length(Y)

    # Make sure there aren't more folds than observations
    if folds >= n
        throw(ArgumentError(msg))
    end

    # The indices are evely spaced and start at the top to make rolling splits for TS data
    indices = Int.(floor.(collect(range(1, size(X, 1), folds + 1))))
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
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> validation_loss(x, y, 5, accuracy, 3)
0.5402532843396273
```
"""
function validation_loss(
    xtrain, ytrain, xtest, ytest, nodes, metric; activation=relu, regularized=true
)
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
- `X::Array`: array of features to train on.
- `Y::Vector`: vector of labels to train on.
- `neurons::Int`: number of neurons to use in the extreme learning machine.
- `metric::Function`: validation metric to calculate.
- `activation::Function=relu`: activation function to use.
- `regularized::Function=true`: whether to use L2 regularization
- `folds::Int`: number of folds to use for cross validation.
- `temporal::Function=true`: whether the data is of a time series or panel nature.

# Examples
```julia
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> cross_validate(x, y, 5, accuracy)
0.8891028047100136
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
            xtst, ytst = reduce(vcat, xfs[(fold + 1):end]),
            reduce(vcat, yfs[(fold + 1):end])
        else
            continue
        end

        mean_metric += validation_loss(
            xtr,
            ytr,
            xtst,
            ytst,
            neurons,
            metric;
            activation=activation,
            regularized=regularized,
        )
    end
    return mean_metric / folds
end

"""
    best_size(m)

Compute the best number of neurons for an estimator.

# Notes
The procedure tests networks with numbers of neurons in a sequence whose length is given 
by iterations on the interval [min_neurons, max_neurons]. Then, it uses the networks 
sizes and validation errors from the sequence to predict the validation error or metric 
for every network size between min_neurons and max_neurons using the function 
approximation ability of an Extreme Learning Machine. Finally, it returns the network 
size with the best predicted validation error or metric.

# Arguments
- `m::Any`: estimator to find the best number of neurons for.

# Examples
```julia
julia> X, T, Y = rand(100, 5), rand(0:1, 100), rand(100)
julia> m1 = GComputation(X, T, y)
julia> best_size(m1)
8
```
"""
function best_size(m)
    loss = Vector{Float64}(undef, m.iterations)
    num_neurons = round.(Int, range(m.min_neurons, m.max_neurons; length=m.iterations))
    (X, Y) = m isa InterruptedTimeSeries ? (m.X₀, m.Y₀) : (m.X, m.Y)

    # Use cross validation to get testing loss from [min_neurons, max_neurons] by iterations
    @inbounds for (idx, potential_neurons) in pairs(num_neurons)
        loss[idx] = cross_validate(
            X,
            Y,
            round(Int, potential_neurons),
            m.validation_metric,
            m.activation,
            m.regularized,
            m.folds,
            m.temporal,
        )
    end

    # Use an extreme learning machine to learn a function F:num_neurons -> loss
    mapper = ExtremeLearner(
        reshape(num_neurons, :, 1), 
        reshape(loss, :, 1), 
        m.approximator_neurons, 
        relu,
    )
    fit!(mapper)
    pred_metrics = predict(mapper, Float64[m.min_neurons:m.max_neurons;])

    return ifelse(startswith(m.task, "c"), argmax([pred_metrics]), argmin([pred_metrics]))
end

"""
    shuffle_data(X, Y)

Shuffles covariates and outcome vector for cross validation.

# Examples
```julia
julia> shuffle_data([1 1; 2 2; 3 3; 4 4], collect(1:4))
([4 4; 2 2; 1 1; 3 3], [4, 2, 1, 3])
```
"""
function shuffle_data(X, Y)
    idx = randperm(size(X, 1))
    new_data = mapslices.(x -> x[idx], [X, Y], dims=1)
    X, Y = new_data

    return Array(X), vec(Y)
end
