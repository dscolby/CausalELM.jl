module Inference

using ..Estimators: CausalEstimator, EventStudy, estimatecausaleffect!, mean

"""
    generatenulldistribution(e, n)

Generate a null distribution for the treatment effect of G-computation or doubly robust 
estimation.

This method estimates the same model that is provided using random permutations of the 
treatment assignment to generate a vector of estimated effects under different treatment
regimes.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

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
    local model = deepcopy(e)
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
    generatenulldistribution(e, n, mean_effect)

Generate a null distribution for the treatment effect in an event study design. By default, 
this method generates a null distribution of mean differences. To generate a null 
distribution of cummulative differences, set the mean_effect argument to false.

Instead of randomizing the assignment of units to the treamtent or control group, this 
method generates the null distribution by reestimating the event study with the intervention
set to n intervals within the total study duration.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

Examples
```julia-repl
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> event_study = EventStudy(x₀, y₀, x₁, y₁)
julia> estimatecausaleffect!(event_study)
julia> generatenulldistribution(event_study, 10)
[-0.5012456678829079, -0.33790650529972194, -0.2534340182760628, -0.21030239864895905, 
-0.11672915615117885, -0.08149441936166794, -0.0685134758182695, -0.06217013151235991, 
-0.05905529159312335, -0.04927743270606937]
```
"""
function generatenulldistribution(e::EventStudy, n::Integer=1000, mean_effect=true)
    local model = deepcopy(e)
    nobs = size(model.Y₀, 1) + size(model.Y₁, 1)
    results = Vector{Float64}(undef, n)
    n -= 1

    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n

        # Find the index to split at the nth interval
        split_idx = floor(Int, iter*(nobs/n))-1
        X, Y = vcat(e.X₀, e.X₁), vcat(e.Y₀, e.Y₁)
        x₀, y₀ = X[1:split_idx, :], Y[1:split_idx]
        x₁, y₁ = X[split_idx+1:end, :], Y[split_idx+1:end]

        # Reestimate the model with the intervention now at the nth interval
        model.X₀, model.Y₀, model.X₁, model.Y₁ = x₀, y₀, x₁, y₁
        estimatecausaleffect!(model)
        results[iter] = ifelse(mean_effect, mean(model.abnormal_returns), 
            sum(model.abnormal_returns))
    end
    return sort(results)
end

end
