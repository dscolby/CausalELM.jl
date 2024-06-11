"""
    mean(x)

Calculate the mean of a vector.

# Examples
```jldoctest
julia> CausalELM.mean([1, 2, 3, 4])
2.5
```
"""
mean(x) = sum(x)/size(x, 1)

"""
    var(x)

Calculate the (sample) mean of a vector.

# Examples
```jldoctest
julia> CausalELM.var([1, 2, 3, 4])
1.6666666666666667
```
"""
var(x) = sum((x .- mean(x)).^2)/(length(x)-1)

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
- `x::Array`: the array to clip if it is binary.
- `var`: the type of x based on calling var_type.

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
