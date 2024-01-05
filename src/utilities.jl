mean(x::Vector{<:Real}) = sum(x)/size(x, 1)

function var(x::Vector{<:Real})
    x̄, n = mean(x), length(x)

    return sum((x .- x̄).^2)/(n-1)
end

# Helpers to subtract or add consecutive elements in a vector
consecutive(v::Vector{<:Real}) = [-(v[i+1], v[i]) for i = 1:length(v)-1]

# Used for multiclass classification
function one_hot_encode(x::Vector{<:Real})
    one_hot = permutedims(float(unique(x) .== reshape(x, (1, size(x, 1))))), (2, 1)
    return one_hot[1]
end
