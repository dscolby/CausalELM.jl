using Random: randperm

"""
    generate_folds(X, Y, folds)

Creates folds for cross validation.

Examples
```julia-repl
julia> xfolds, y_folds = generate_folds(zeros(20, 2), zeros(20), 5)
```
"""
function generate_folds(X::Array{Float64}, Y::Array{Float64}, folds::Int64)
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

Creates rolling folds for cross validation of time series data.

Examples
```julia-repl
julia> xfolds, y_folds = generate_temporal_folds(zeros(20, 2), zeros(20), 5, temporal=true)
```
"""
function generate_temporal_folds(X::Array{<:Real}, Y::Array{<:Real}, folds::Int=5)
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
    validation_loss(xtrain, ytrain, xtest, ytest, nodes, metric; activation, regularized)

Calculate a validation metric for a single fold in k-fold cross validation.

Examples
```julia-repl
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> validation_loss(x, y, 5, accuracy, 3)
0.0
```
"""
function validation_loss(xtrain::Array{Float64}, ytrain::Array{Float64}, 
    xtest::Array{Float64}, ytest::Array{Float64}, nodes::Integer, metric::Function; 
    activation::Function=relu, regularized::Bool=true)

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

Examples
```julia-repl
julia> x = rand(100, 5); y = Float64.(rand(100) .> 0.5)
julia> cross_validate(x, y, 5, accuracy)
0.0257841765251021
```
"""
function cross_validate(X::Array{Float64}, Y::Array{Float64}, neurons::Integer, 
    metric::Function, activation::Function=relu, regularized::Bool=true, folds::Integer=5, 
    temporal::Bool=false)

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

The procedure tests networks with numbers of neurons in a sequence whose length is given 
by iterations on the interval [min_neurons, max_neurons]. Then, it uses the networks 
sizes and validation errors from the sequence to predict the validation error or metric 
for every network size between min_neurons and max_neurons using the function 
approximation ability of an Extreme Learning Machine. Finally, it returns the network 
size with the best predicted validation error or metric.

Examples
```julia-repl
julia> best_size(rand(100, 5), rand(100), mse, "regression")
11
```
"""
function best_size(X::Array{Float64}, Y::Array{Float64}, metric::Function, task::String,
    activation::Function=relu, min_neurons::Integer=1, max_neurons::Integer=100, 
    regularized::Bool=true, folds::Integer=5, temporal::Bool=false,
    iterations::Integer=Int(round(size(X, 1)/10)), 
    elm_size=Integer=Int(round(size(X, 1)/10)))
    
    loss, num_neurons = Vector{Float64}(undef, iterations), 
        round.(Int, range(min_neurons, max_neurons, length=iterations))
   
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
    shuffle_data(X, Y, T)

Shuffles covariates, treatment vector, and outcome vector for cross validation.

Examples
```julia-repl
julia> x, y, t = rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> shuffle_data(x, y, t)
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
function shuffle_data(X::Array{Float64}, Y::Array{Float64})
        idx = randperm(size(X, 1))
        new_data = mapslices.(x->x[idx], [X, Y], dims=1)
        X, Y = new_data

        return X, vec(Y)
end
