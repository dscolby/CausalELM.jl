"""
Metrics to evaluate the performance of an Extreme learning machine for regression
and classification tasks.
"""
module Metrics

"""
    mse(y_actual, y_pred)

Calculate the mean squared error

See also ['mae'](@ref).

Examples
```julia-repl
julia> mse([0, 0, 0], [0, 0, 0])
0
julia> mse([-1, -1, -1], [1, 1, 1])
4
```
"""
function mse(y_actual::Vector, y_pred::Vector) 
    @assert length(y_actual) == length(y_pred) "y_actual and y_pred must be the same length"
    return @fastmath sum((y_actual - y_pred).^2) / length(y_actual)
end

"""
    mae(y_actual, y_pred)

Calculate the mean absolute error

See also ['mse'](@ref).

Examples
```julia-repl
julia> mae([-1, -1, -1], [1, 1, 1])
2
julia> mae([1, 1, 1], [2, 2, 2])
1
```
"""
function mae(y_actual::Vector, y_pred::Vector) 
    @assert length(y_actual) == length(y_pred) "y_actual and y_pred must be the same length"
    return @fastmath sum(abs.(y_actual .- y_pred)) / length(y_actual)
end

"""
    accuracy(y_actual, y_pred)

Calculate the accuracy for a classification task

Examples
```julia-repl
julia> accuracy([1, 1, 1, 1], [0, 1, 1, 0])
0.5
julia> accuracy([1, 2, 3, 4], [1, 1, 1, 1])
0.25
```
"""
function accuracy(y_actual::Vector, y_pred::Vector)
    differences = y_actual .- y_pred
    return length(differences[differences .== 0]) / length(y_pred)
end

"""
    precision(y_actual, y_pred)

Calculate the precision for a classification task

See also ['recall'](@ref).

Examples
```julia-repl
julia> precision([0, 1, 0, 0], [0, 1, 1, 0])
0.5
julia> precision([0, 1, 0, 0], [0, 1, 0, 0])
1
```
"""
function precision(y_actual::Vector, y_pred::Vector)
    confmat = confusionmatrix(y_actual, y_pred)
    n = length(Set(y_actual))

    # Binary classification
    if size(confmat) == (2, 2)
        tp = confmat[2, 2]
        return tp / sum(confmat[2, :])
    
    # Multiclass classification
    else
        classwise_prec = Array{Float64, 1}(undef, n)

        for idx in axes(confmat, 1)
            tp = confmat[idx, idx]
            sum(confmat[idx, :]) == 0 ? single = 0 : single = tp / sum(confmat[idx, :])
            classwise_prec[idx] = single
        end
        return sum(classwise_prec) / n
    end
end

"""
    recall(y_actual, y_pred)

Calculate the recall for a classification task

See also ['precision'](@ref).

Examples
```julia-repl
julia> recall([1, 2, 1, 3, 0], [2, 2, 2, 3, 1])
0.5
julia> recall([1, 2, 1, 3, 2], [2, 2, 2, 3, 1])
1
```
"""
function recall(y_actual::Vector, y_pred::Vector)
    confmat = confusionmatrix(y_actual, y_pred)
    n = length(Set(y_actual))

    # Binary classification
    if size(confmat) == (2, 2)
        tp = confmat[2, 2]
        return tp / sum(confmat[:, 2])
    
    # Multiclass classification
    else
        classwise_rec = Array{Float64, 1}(undef, n)

        for idx in axes(confmat, 2)
            tp = confmat[idx, idx]
            sum(confmat[:, idx]) == 0 ? single = 0 : single = tp / sum(confmat[:, idx])
            classwise_rec[idx] = single
        end
        return sum(classwise_rec) / n
    end
end

"""
    F1(y_actual, y_pred)

Calculate the F1 score for a classification task

Examples
```julia-repl
julia> F1([1, 2, 1, 3, 0], [2, 2, 2, 3, 1])
0.4
julia> F1([1, 2, 1, 3, 2], [2, 2, 2, 3, 1])
0.47058823529411764
```
"""
function F1(y_actual::Vector, y_pred::Vector)
    prec, rec = precision(y_actual, y_pred), recall(y_actual, y_pred)
    return 2(prec * rec) / (prec + rec)
end

"""
    confusionmatrix(y_actual, y_pred)

Generate a confusion matrix

Examples
```julia-repl
julia> confusionmatrix([1, 1, 1, 1, 0], [1, 1, 1, 1, 0])
2×2 Matrix{Int8}:
 1  0
 0 4
julia> confusionmatrix([1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 0, 2])
3×3 Matrix{Int8}:
 1  0 0
 0 4 0
 0 0 1
```
"""
function confusionmatrix(y_actual::Vector, y_pred::Vector)
    confmat = zeros(Int8, length(Set(y_actual)), length(Set(y_actual)))

    # Recode since Julia is a 1-index language
    if minimum(y_actual) == 0
        y_actual .+= 1; y_pred .+= 1
    end

    for (predicted, actual) in zip(y_pred, y_actual)
        @inbounds confmat[predicted, actual] += 1
    end
    return confmat
end

end
