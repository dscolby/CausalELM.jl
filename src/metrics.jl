"""
Metrics to evaluate the performance of an Extreme learning machine for regression
and classification tasks.
"""
module Metrics

"""
    mse(y, ŷ)

Calculate the mean squared error

See also [`mae`](@ref).

Examples
```julia-repl
julia> mse([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
0
julia> mse([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
4
```
"""
function mse(y::Vector{<:Real}, ŷ::Vector{<:Real}) 
    if length(y) !== length(ŷ)
        throw(DimensionMismatch("y and ̂y must be the same length"))
    end

    return @fastmath sum((y - ŷ).^2) / length(y)
end

"""
    mae(y, ŷ)

Calculate the mean absolute error

See also [`mse`](@ref).

Examples
```julia-repl
julia> mae([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
2
julia> mae([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
1
```
"""
function mae(y::Vector{Float64}, ŷ::Vector{Float64}) 
    if length(y) !== length(ŷ)
        throw(DimensionMismatch("y and ̂y must be the same length"))
    end

    return @fastmath sum(abs.(y .- ŷ)) / length(y)
end

"""
    accuracy(y, ŷ)

Calculate the accuracy for a classification task

Examples
```julia-repl
julia> accuracy([1, 1, 1, 1], [0, 1, 1, 0])
0.5
julia> accuracy([1, 2, 3, 4], [1, 1, 1, 1])
0.25
```
"""
function accuracy(y::Vector{Float64}, ŷ::Vector{Float64})
    if length(y) !== length(ŷ)
        throw(DimensionMismatch("y and ̂y must be the same length"))
    end

    @fastmath differences = y .- ŷ
    return @fastmath length(differences[differences .== 0]) / length(ŷ)
end

"""
    precision(y, ŷ)

Calculate the precision for a classification task

See also [`recall`](@ref).

Examples
```julia-repl
julia> precision([0, 1, 0, 0], [0, 1, 1, 0])
0.5
julia> precision([0, 1, 0, 0], [0, 1, 0, 0])
1
```
"""
function precision(y::Vector{Int64}, ŷ::Vector{Int64})
    confmat = confusionmatrix(y, ŷ)
    n = length(Set(y))

    # Binary classification
    if size(confmat) == (2, 2)
        tp = confmat[2, 2]
        return @fastmath tp / sum(confmat[2, :])
    
    # Multiclass classification
    else
        classwise_prec = Array{Float64, 1}(undef, n)

        for idx in axes(confmat, 1)
            tp, all_pos = confmat[idx, idx], @fastmath sum(confmat[idx, :])
            single = ifelse(all_pos == 0, 0, @fastmath tp / all_pos)

            classwise_prec[idx] = single
        end
        return @fastmath sum(classwise_prec) / n
    end
end

"""
    recall(y, ŷ)

Calculate the recall for a classification task

See also [`precision`](@ref).

Examples
```julia-repl
julia> recall([1, 2, 1, 3, 0], [2, 2, 2, 3, 1])
0.5
julia> recall([1, 2, 1, 3, 2], [2, 2, 2, 3, 1])
1
```
"""
function recall(y::Vector{Int64}, ŷ::Vector{Int64})
    confmat, n = confusionmatrix(y, ŷ), length(Set(y))

    # Binary classification
    if size(confmat) == (2, 2)
        tp = confmat[2, 2]
        return @fastmath tp / sum(confmat[:, 2])
    
    # Multiclass classification
    else
        classwise_rec = Array{Float64, 1}(undef, n)

        for idx in axes(confmat, 2)
            tp, all_pos = confmat[idx, idx], @fastmath sum(confmat[:, idx])
            single = ifelse(all_pos == 0, 0, @fastmath tp / all_pos)
            classwise_rec[idx] = single
        end
        return @fastmath sum(classwise_rec) / n
    end
end

"""
    F1(y, ŷ)

Calculate the F1 score for a classification task

Examples
```julia-repl
julia> F1([1, 2, 1, 3, 0], [2, 2, 2, 3, 1])
0.4
julia> F1([1, 2, 1, 3, 2], [2, 2, 2, 3, 1])
0.47058823529411764
```
"""
function F1(y::Vector{Int64}, ŷ::Vector{Int64})
    prec, rec = precision(y, ŷ), recall(y, ŷ)
    return @fastmath 2(prec * rec) / (prec + rec)
end

"""
    confusionmatrix(y, ŷ)

Generate a confusion matrix

Examples
```julia-repl
julia> confusionmatrix([1, 1, 1, 1, 0], [1, 1, 1, 1, 0])
2×2 Matrix{Int64}:
 1  0
 0 4
julia> confusionmatrix([1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 0, 2])
3×3 Matrix{Int64}:
 1  0 0
 0 4 0
 0 0 1
```
"""
function confusionmatrix(y::Vector{Int64}, ŷ::Vector{Int64})
    confmat = zeros(Int64, length(Set(y)), length(Set(y)))

    # Recode since Julia is a 1-index language
    if minimum(y) == 0
        @fastmath y .+= 1; @fastmath ŷ .+= 1
    end

    for (predicted, actual) in zip(ŷ, y)
        @inbounds @fastmath confmat[predicted, actual] += 1
    end
    return confmat
end

end
