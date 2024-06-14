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
 0.9999999
 1.0e-7

julia> CausalELM.clip_if_binary([1.2, -0.02], CausalELM.Count())
2-element Vector{Float64}:
  1.2
 -0.02
```
"""
clip_if_binary(x::Array{<:Real}, var) = var isa Binary ? clamp.(x, 1e-7, 1 - 1e-7) : x

"""
    model_config(effect_type)

Generate fields common to all CausalEstimator, Metalearner, and InterruptedTimeSeries 
structs.

# Arguments
- `effect_type::String`: "average_effect" or "individual_effect" to define fields for either 
    models that estimate average effects or the CATE.

# Examples
```julia
julia> struct TestStruct CausalELM.@model_config "average_effect" end
julia> TestStruct("ATE", false, "classification", true, relu, F1, 2, 10, 5, 100, 5, 5, 0.25)
TestStruct("ATE", false, "classification", true, relu, F1, 2, 10, 5, 100, 5, 5, 0.25)
```
"""
macro model_config(effect_type::String)
    msg = "the effect type must either be average_effect or individual_effect"
    if !(effect_type in ("average_effect", "individual_effect"))
        throw(ArgumentError(msg))
    end

    field_type = if effect_type == "average_effect"
        Float64
    else
        Array{Float64}
    end

    fields = quote
        quantity_of_interest::String
        temporal::Bool
        task::String
        regularized::Bool
        activation::Function
        validation_metric::Function
        min_neurons::Int64
        max_neurons::Int64
        folds::Int64
        iterations::Int64
        approximator_neurons::Int64
        num_neurons::Int64
        causal_effect::$field_type
    end
    return esc(fields)
end

"""
    standard_input_data()

Generate fields common to all CausalEstimators except DoubleMachineLearning and all 
Metalearners excetp RLearner and DoublyRobustLearner.

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
    double_learner_input_data()

Generate fields common to DoubleMachineLearning, RLearner, and DoublyRobustLearner.

# Examples
```julia
julia> struct TestStruct CausalELM.@double_learner_input_data end
julia> TestStruct([5.2], [0.8], [0.96], [0.87 1.8])
TestStruct([5.2], [0.8], [0.96], [0.87 1.8])
```
"""
macro double_learner_input_data()
    inputs = quote
        X::Array{Float64}
        T::Array{Float64}
        Y::Array{Float64}
        W::Array{Float64}
    end
    return esc(inputs)
end
