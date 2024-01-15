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
