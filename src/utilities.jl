mean(x::Vector{<:Real}) = sum(x)/length(x)

function var(x::Vector{<:Real})
    x̄, n = mean(x), length(x)

    return sum((x .- x̄).^2)/(n-1)
end

# Helpers to subtract or add consecutive elements in a vector
consecutive(v::Vector{<:Real}) = [-(v[i+1], v[i]) for i = 1:length(v)-1]
