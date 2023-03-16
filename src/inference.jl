module Inference

using ..Estimators: CausalEstimator, EventStudy, estimatecausaleffect!

"""
    generatenulldistribution(e, n)

Generate a null distribution for the treatment effect of G-computation or doubly robust 
estimation.

This method estimates the same model that is provided using random permutations of the 
treatment assignment to generate a vector of estimated effects under different treatment
regimes.

Examples
```julia-repl
julia> x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
julia> g_computer = GComputation(x, y, t)
julia> estimatecausaleffect!(g_computer)
julia> generatenulldistribution(g_computer, 500)
[0.016297180690693656, 0.0635928694685571, 0.20004144093635673, 0.505893866040335, 
0.5130594630907543, 0.5432486130493388, 0.6181727442724846, 0.61838399963459, 
0.7038981488009489, 0.7043407710415689  …  21.909186142780246, 21.960498059428854, 
21.988553083790023, 22.285403459215363, 22.613625375395973, 23.382102081355548, 
23.52056245175936, 24.739658523175912, 25.30523686137909, 28.07474553316176]
```
"""
function generatenulldistribution(e::CausalEstimator, n::Integer=1000)
    model = deepcopy(e)
    nobs = size(model.T, 1)
    results = Vector{Float64}(undef, n)
    
    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n 
        model.T = float(rand(0:1, nobs))
        estimatecausaleffect!(model)
        results[iter] = model.causal_effect
    end
    return sort(results)
end

"""
    generatenulldistribution(e, n)

Generate a null distribution for the treatment effect in an event study design.

Instead of randomizing the assignment of units to the treamtent or control group, this 
method generates the null distribution by reestimating the event study with the intervention
set to different times.

Examples
```julia-repl
julia> x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
julia> g_computer = GComputation(x, y, t)
julia> estimatecausaleffect!(g_computer)
julia> generatenulldistribution(g_computer, 500)
[0.016297180690693656, 0.0635928694685571, 0.20004144093635673, 0.505893866040335, 
0.5130594630907543, 0.5432486130493388, 0.6181727442724846, 0.61838399963459, 
0.7038981488009489, 0.7043407710415689  …  21.909186142780246, 21.960498059428854, 
21.988553083790023, 22.285403459215363, 22.613625375395973, 23.382102081355548, 
23.52056245175936, 24.739658523175912, 25.30523686137909, 28.07474553316176]
```
"""
function generatenulldistribution(e::EventStudy, n::Integer=1000)
    model = deepcopy(e)
    nobs = size(model.Y₀, 1) + size(model.Y₁, 1)
    results = Vector{Float64}(undef, n)
    n = ifelse(n > nobs-1, nobs-1, n)

    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n 
        X, Y = vcat(model.X₀, model.X₁), vcat(model.Y₀, model.Y₁)
        model.T = float(rand(0:1, nobs))
        estimatecausaleffect!(model)
        results[iter] = model.causal_effect
    end
    return sort(results)
end

end
