module Inference

using ..Estimators: EventStudy, CausalEstimator, estimatecausaleffect!
using ..Metalearners: Metalearner, estimatecausaleffect!

function generatenulldistribution(e::CausalEstimator, n::Integer=1000)
    model = deepcopy(e)
    nobs = size(model.T)
    results = Vector{Float64}(undef, n)
    
    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n 
        model.T = float(rand(0:1, nobs))
        estimatecausaleffect!(model)
        results[iter] = model.causal_effect
    end
    return sort(results)
end

end
