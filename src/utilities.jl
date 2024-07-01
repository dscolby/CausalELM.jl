"""Abstract type used to dispatch risk_ratio on nonbinary treatments"""
abstract type Nonbinary end

"""Type used to dispatch risk_ratio on binary treatments"""
struct Binary end

"""Type used to dispatch risk_ratio on count treatments"""
struct Count <: Nonbinary end

"""Type used to dispatch risk_ratio on continuous treatments"""
struct Continuous <: Nonbinary end

"""
    var_type(x)

Determine the type of variable held by a vector.

# Examples
```jldoctest
julia> CausalELM.var_type([1, 2, 3, 2, 3, 1, 1, 3, 2])
CausalELM.Count()
```
"""
function var_type(x::Array{<:Real})
    x_set = Set(x)
    
    if x_set == Set([0, 1]) || x_set == Set([0]) || x_set == Set([1])
        return Binary()
    elseif x_set == Set(round.(x_set))
        return Count()
    else
        return Continuous()
    end
end

"""
    mean(x)

Calculate the mean of a vector.

# Examples
```jldoctest
julia> CausalELM.mean([1, 2, 3, 4])
2.5
```
"""
mean(x) = sum(x) / size(x, 1)

"""
    var(x)

Calculate the (sample) mean of a vector.

# Examples
```jldoctest
julia> CausalELM.var([1, 2, 3, 4])
1.6666666666666667
```
"""
var(x) = sum((x .- mean(x)) .^ 2) / (length(x) - 1)

"""
    one_hot_encode(x)

One hot encode a categorical vector for multiclass classification.

# Examples
```jldoctest
julia> CausalELM.one_hot_encode([1, 2, 3, 4, 5])
5×5 Matrix{Float64}:
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0
```
"""
function one_hot_encode(x)
    one_hot = permutedims(float(unique(x) .== reshape(x, (1, size(x, 1))))), (2, 1)
    return one_hot[1]
end

"""
    clip_if_binary(x, var)

Constrain binary values between 1e-7 and 1 - 1e-7, otherwise return the original values.

# Arguments
- `x::Array`: array to clip if it is binary.
- `var`: type of x based on calling var_type.

See also [`var_type`](@ref).

# Examples
```jldoctest
julia> CausalELM.clip_if_binary([1.2, -0.02], CausalELM.Binary())
2-element Vector{Float64}:
 1.0
 0.0

julia> CausalELM.clip_if_binary([1.2, -0.02], CausalELM.Count())
2-element Vector{Float64}:
  1.2
 -0.02
```
"""
clip_if_binary(x::Array{<:Real}, var) = var isa Binary ? clamp.(x, 0.0, 1.0) : x

"""
    model_config(effect_type)

Generate fields common to all CausalEstimator, Metalearner, and InterruptedTimeSeries 
structs.

# Arguments
- `effect_type::String`: "average_effect" or "individual_effect" to define fields for either 
    models that estimate average effects or the CATE.

# Examples
```julia
julia> struct TestStruct CausalELM.@model_config average_effect end
julia> TestStruct("ATE", false, "classification", true, relu, F1, 2, 10, 5, 100, 5, 5, 0.25)
TestStruct("ATE", false, "classification", true, relu, F1, 2, 10, 5, 100, 5, 5, 0.25)
```
"""
macro model_config(effect_type)
    msg = "the effect type must either be average_effect or individual_effect"

    if string(effect_type) == "average_effect"
        field_type = :Float64
    elseif string(effect_type) == "individual_effect"
        field_type = :(Array{Float64})
    else
        throw(ArgumentError(msg))
    end

    fields = quote
        quantity_of_interest::String
        temporal::Bool
        task::String
        activation::Function
        sample_size::Integer
        num_machines::Integer
        num_feats::Integer
        num_neurons::Integer
        causal_effect::$field_type
    end
    return esc(fields)
end

"""
    standard_input_data()

Generate fields common to all CausalEstimators except DoubleMachineLearning and all 
Metalearners except RLearner and DoublyRobustLearner.

# Examples
```julia
julia> struct TestStruct CausalELM.@standard_input_data end
julia> TestStruct([5.2], [0.8], [0.96])
TestStruct([5.2], [0.8], [0.96])
```
"""
macro standard_input_data()
    inputs = quote
        X::Array{Float64}
        T::Array{Float64}
        Y::Array{Float64}
    end
    return esc(inputs)
end

"""
    generate_folds(X, T, Y, folds)

Create folds for cross validation.

# Examples
```jldoctest
julia> xfolds, tfolds, yfolds = CausalELM.generate_folds(zeros(4, 2), zeros(4), ones(4), 2)
([[0.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 0.0]], [[0.0], [0.0, 0.0, 0.0]], [[1.0], [1.0, 1.0, 1.0]])
```
"""
function generate_folds(X, T, Y, folds)
    msg = """the number of folds must be less than the number of observations"""
    n = length(Y)

    if folds >= n
        throw(ArgumentError(msg))
    end

    x_folds = Array{Array{Float64, 2}}(undef, folds)
    t_folds = Array{Array{Float64, 1}}(undef, folds)
    y_folds = Array{Array{Float64, 1}}(undef, folds)

    # Indices to start and stop for each fold
    stops = round.(Int, range(; start=1, stop=n, length=folds + 1))

    # Indices to use for making folds
    indices = [s:(e - (e < n) * 1) for (s, e) in zip(stops[1:(end - 1)], stops[2:end])]

    for (i, idx) in enumerate(indices)
        x_folds[i], t_folds[i], y_folds[i] = X[idx, :], T[idx], Y[idx]
    end

    return x_folds, t_folds, y_folds
end
