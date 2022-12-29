function recode(ŷ::Array{Float64})
    rounded = round(ŷ)
    minimum(rounded) == 0 ? 
end