using Random: shuffle
using CausalELM: mean, var_type, clip_if_binary

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

See also [`CausalELM.RegularizedExtremeLearner`](@ref).

# Examples
```julia
julia> x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
julia> m1 = ExtremeLearner(x, y, 10, σ)
```
"""
mutable struct ExtremeLearner
    X::Array{Float64}
    Y::Array{Float64}
    training_samples::Int64
    features::Int64
    hidden_neurons::Int64
    activation::Function
    __fit::Bool
    weights::Array{Float64}
    β::Array{Float64}
    H::Array{Float64}
    counterfactual::Array{Float64}

    function ExtremeLearner(X, Y, hidden_neurons, activation)
        return new(X, Y, size(X, 1), size(X, 2), hidden_neurons, activation, false)
    end
end

"""
    ELMEnsemble(X, Y, sample_size, num_machines, num_neurons)

Initialize a bagging ensemble of extreme learning machines. 

# Arguments
- `X::Array{Float64}`: array of features for predicting labels.
- `Y::Array{Float64}`: array of labels to predict.
- `sample_size::Integer`: how many data points to use for each extreme learning machine.
- `num_machines::Integer`: how many extreme learning machines to use.
- `num_feats::Integer`: how many features to consider for eac exreme learning machine.
- `num_neurons::Integer`: how many neurons to use for each extreme learning machine.
- `activation::Function`: activation function to use for the extreme learning machines.

# Notes
ELMEnsemble uses the same bagging approach as random forests when the labels are continuous 
but uses the average predicted probability, rather than voting, for classification.

# Examples
```julia
julia> X, Y =  rand(100, 5), rand(100)
julia> m1 = ELMEnsemble(X, Y, 10, 50, 5, 5, CausalELM.relu)
```
"""
mutable struct ELMEnsemble
    X::Array{Float64}
    Y::Array{Float64}
    elms::Array{ExtremeLearner}
    feat_indices::Vector{Vector{Int64}}
end

function ELMEnsemble(
    X::Array{Float64}, 
    Y::Array{Float64}, 
    sample_size::Integer, 
    num_machines::Integer,
    num_feats::Integer, 
    num_neurons::Integer,
    activation::Function
)
    # Sampling from the data with replacement
    indices = [rand(1:length(Y), sample_size) for i ∈ 1:num_machines]
    feat_indices = [shuffle(1:size(X, 2))[1:num_feats] for i ∈ 1:num_machines]
    xs = [X[indices[i], feat_indices[i]] for i ∈ 1:num_machines]
    ys = [Y[indices[i]] for i ∈ 1:num_machines]
    elms = [ExtremeLearner(xs[i], ys[i], num_neurons, activation) for i ∈ eachindex(xs)]

    return ELMEnsemble(X, Y, elms, feat_indices)
end

"""
    fit!(model)

Fit an ExtremeLearner to the data.

# References
For more details see: 
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

# Examples
```julia
julia> x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
julia> m1 = ExtremeLearner(x, y, 10, σ)
```
"""
function fit!(model::ExtremeLearner)
    set_weights_biases(model)

    model.__fit = true
    model.β = model.H\model.Y
    return model.β
end

"""
    fit!(model)

Fit an ensemble of ExtremeLearners to the data. 

# Arguments
- `model::ELMEnsemble`: ensemble of ExtremeLearners to fit.

# Notes
This uses the same bagging approach as random forests when the labels are continuous but 
uses the average predicted probability, rather than voting, for classification.

# Examples
```julia
julia> X, Y =  rand(100, 5), rand(100)
julia> m1 = ELMEnsemble(X, Y, 10, 50, 5, CausalELM.relu)
julia> fit!(m1)
```
"""
function fit!(model::ELMEnsemble)
    Threads.@threads for elm in model.elms
        fit!(elm)
    end
end

"""
    predict(model, X)

Use an ExtremeLearningMachine or ELMEnsemble to make predictions.

# Notes
If using an ensemble to make predictions, this method returns a maxtirs where each row is a
prediction and each column is a model.

# References
For more details see: 
    Huang G-B, Zhu Q-Y, Siew C. Extreme learning machine: theory and applications. 
    Neurocomputing. 2006;70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126

# Examples
```julia
julia> x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
julia> m1 = ExtremeLearner(x, y, 10, σ)
julia> fit!(m1, sigmoid)
julia> predict(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])

julia> m2 = ELMEnsemble(X, Y, 10, 50, 5, CausalELM.relu)
julia> fit!(m2)
julia> predict(m2)
```
"""
function predict(model::ExtremeLearner, X)
    if !model.__fit
        throw(ErrorException("run fit! before calling predict"))
    end

    predictions = model.activation(X * model.weights) * model.β

    return clip_if_binary(predictions, var_type(model.Y))
end

@inline function predict(model::ELMEnsemble, X) 
    predictions = reduce(
        hcat, 
        [predict(model.elms[i], X[:, model.feat_indices[i]]) for i ∈ 1:length(model.elms)]
    )

    return vec(mapslices(mean, predictions, dims=2))
end

"""
    predict_counterfactual!(model, X)

Use an ExtremeLearningMachine to predict the counterfactual.

# Notes
This should be run with the observed covariates. To use synthtic data for what-if scenarios 
use predict.

See also [`predict`](@ref).

# Examples
```julia
julia> x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
julia> m1 = ExtremeLearner(x, y, 10, σ)
julia> f1 = fit(m1, sigmoid)
julia> predict_counterfactual!(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
```
"""
function predict_counterfactual!(model::ExtremeLearner, X)
    model.counterfactual = predict(model, X)

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
julia> x, y = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0], [0.0, 1.0, 0.0, 1.0]
julia> m1 = ExtremeLearner(x, y, 10, σ)
julia> f1 = fit(m1, sigmoid)
julia> predict_counterfactual(m1, [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0])
julia> placebo_test(m1)
```
"""
function placebo_test(model::ExtremeLearner)
    m = "Use predict_counterfactual! to estimate a counterfactual before using placebo_test"
    if !isdefined(model, :counterfactual)
        throw(ErrorException(m))
    end
    return predict(model, model.X), model.counterfactual
end

"""
    set_weights_biases(model)

Calculate the weights and biases for an extreme learning machine.

# Notes
Initialization is done using uniform Xavier initialization.

# References
For details see;
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.

# Examples
```julia
julia> m1 = RegularizedExtremeLearner(x, y, 10, σ)
julia> set_weights_biases(m1)
```
"""
function set_weights_biases(model::ExtremeLearner)
    a, b = -1, 1
    model.weights = @fastmath a .+ ((b - a) .* rand(model.features, model.hidden_neurons))

    return model.H = @fastmath model.activation((model.X * model.weights))
end

function Base.show(io::IO, model::ExtremeLearner)
    return print(
        io, "Extreme Learning Machine with ", model.hidden_neurons, " hidden neurons"
    )
end

function Base.show(io::IO, model::ELMEnsemble)
    return print(
        io, "Extreme Learning Machine Ensemble with ", length(model.elms), " learners"
    )
end
