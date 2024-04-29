"""
    mean(x)

Calculate the mean of a vector.

Examples
```julia
julia> mean([1, 2, 3, 4])
2.5
```
"""
mean(x) = sum(x)/size(x, 1)

"""
    var(x)

Calculate the (sample) mean of a vector.

Examples
```julia
julia> var([1, 2, 3, 4])
1.6666666666666667
```
"""
var(x) = sum((x .- mean(x)).^2)/(length(x)-1)


"""
    consecutive(x)

Subtract consecutive elements in a vector.

This function is only used to create a rolling average for interrupted time series analysis.

Examples
```julia
julia> consecutive([1, 2, 3, 4, 5])
4-element Vector{Int64}:
 1
 1
 1
 1
```
"""
consecutive(v) = [-(v[i+1], v[i]) for i = 1:length(v)-1]

"""
    one_hot_encode(x)

One hot encode a categorical vector for multiclass classification.

Examples
```julia
julia> one_hot_encode([1, 2, 3, 4, 5])
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

Examples
```julia
julia> clip_if_binary([1.2, -0.02], Binary())
2-element Vector{Float64}:
 0.9999999
 1.0e-7
julia> clip_if_binary([1.2, -0.02], Count())
 2-element Vector{Float64}:
 1.2
 -0.02
```
"""
clip_if_binary(x::Array{<:Real}, var) = var isa Binary ? clamp.(x, 1e-7, 1 - 1e-7) : x
