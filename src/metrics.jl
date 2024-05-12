using LinearAlgebra: diag, replace!

"""
    mse(y, ŷ)

Calculate the mean squared error

See also [`mae`](@ref).

Examples
```julia
mse([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
mse([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
```
"""
function mse(y, ŷ) 
    if length(y) !== length(ŷ)
        throw(DimensionMismatch("y and ̂y must be the same length"))
    end

    return @fastmath sum((y - ŷ).^2) / length(y)
end

"""
    mae(y, ŷ)

Calculate the mean absolute error

See also [`mse`](@ref).

# Examples
```julia
mae([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
mae([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
```
"""
function mae(y, ŷ) 
    if length(y) !== length(ŷ)
        throw(DimensionMismatch("y and ̂y must be the same length"))
    end

    return @fastmath sum(abs.(y .- ŷ)) / length(y)
end

"""
    accuracy(y, ŷ)

Calculate the accuracy for a classification task

# Examples
```julia
accuracy([1, 1, 1, 1], [0, 1, 1, 0])
accuracy([1, 2, 3, 4], [1, 1, 1, 1])
```
"""
function accuracy(y, ŷ)
    if length(y) !== length(ŷ)
        throw(DimensionMismatch("y and ̂y must be the same length"))
    end

    # Converting from one hot encoding to the original representation if y is multiclass
    if !isa(y, Vector)
        y, ŷ = vec(mapslices(argmax, y, dims=2)), vec(mapslices(argmax, ŷ, dims=2))
    end

    @fastmath differences = y .- ŷ
    return @fastmath length(differences[differences .== 0]) / length(ŷ)
end

"""
    precision(y, ŷ)

Calculate the precision for a classification task

See also [`recall`](@ref).

# Examples
```julia
precision([0, 1, 0, 0], [0, 1, 1, 0])
precision([0, 1, 0, 0], [0, 1, 0, 0])
```
"""
function Base.precision(y::Array{Int64}, ŷ::Array{Int64})
    confmat = confusion_matrix(y, ŷ)

    if size(confmat) == (2, 2)
        confmat[1, 1] == 0 && 0.0
        confmat[1, 2] == 0 && 1.0
        return @fastmath confmat[1, 1]/sum(confmat[1, :])
    else
        intermediate = @fastmath diag(confmat)./vec(sum(confmat, dims=2))
        replace!(intermediate, NaN=>0)
        return mean(intermediate)
    end
end

"""
    recall(y, ŷ)

Calculate the recall for a classification task

See also [`precision`](@ref).

# Examples
```julia
recall([1, 2, 1, 3, 0], [2, 2, 2, 3, 1])
recall([1, 2, 1, 3, 2], [2, 2, 2, 3, 1])
```
"""
function recall(y, ŷ)
    confmat = confusion_matrix(y, ŷ)

    if size(confmat) == (2, 2)
        confmat[1, 1] == 0 && 0.0
        confmat[2, 1] == 0 && 1.0
        return @fastmath confmat[1, 1]/sum(confmat[:, 1])
    else
        intermediate = @fastmath diag(confmat)./vec(sum(confmat, dims=1))
        replace!(intermediate, NaN=>0)
        return mean(intermediate)
    end
end

"""
    F1(y, ŷ)

Calculate the F1 score for a classification task

# Examples
```julia
F1([1, 2, 1, 3, 0], [2, 2, 2, 3, 1])
F1([1, 2, 1, 3, 2], [2, 2, 2, 3, 1])
```
"""
function F1(y, ŷ)
    prec, rec = precision(y, ŷ), recall(y, ŷ)
    return @fastmath 2(prec * rec) / (prec + rec)
end

"""
    confusion_matrix(y, ŷ)

Generate a confusion matrix

# Examples
```julia
confusion_matrix([1, 1, 1, 1, 0], [1, 1, 1, 1, 0])
confusion_matrix([1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 0, 2])
```
"""
function confusion_matrix(y, ŷ)
    # Converting from one hot encoding to the original representation if y is multiclass
    if !isa(y, Vector)
        y, ŷ = vec(mapslices(argmax, y, dims=2)), vec(mapslices(argmax, ŷ, dims=2))
    end

    # Recode since Julia is a 1-index language
    flor = minimum(vcat(y, ŷ))
    if flor < 1
        @fastmath y .+= (1 - flor)
        @fastmath ŷ .+= (1 - flor)
    end

    n = maximum(reduce(vcat, (y, ŷ)))
    confmat = zeros(Int64, n, n)

    for (predicted, actual) in zip(ŷ, y)
        @inbounds @fastmath confmat[predicted, actual] += 1
    end
    return confmat
end
