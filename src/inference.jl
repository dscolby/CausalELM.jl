using Random: shuffle

"""
    summarize(its, mean_effect)

Return a summary from an interrupted time series estimator.

p-values and standard errors are estimated using approximate randomization inference that
permutes the time of the intervention.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```jldoctest
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
1-element Vector{Float64}
 0.25714308
julia> summarize(m1)
 {"Task" => "Regression", "Regularized" => true, "Activation Function" => relu, 
 "Validation Metric" => "mse","Number of Neurons" => 2, 
 "Number of Neurons in Approximator" => 10, "β" => [0.25714308], 
 "Causal Effect" => -3.9101138, "Standard Error" => 1.903434356, "p-value" = 0.00123356}
```
"""
function summarize(its::InterruptedTimeSeries, n=1000, mean_effect=true)
    if !isdefined(its, :Δ)
        throw(ErrorException("call estimate_causal_effect! before calling summarize"))
    end

    effect = ifelse(mean_effect, mean(its.Δ), sum(its.Δ))

    p, stderr = quantities_of_interest(its, n, mean_effect)

    summary_dict = Dict()
    nicenames = ["Task", "Regularized", "Activation Function", "Validation Metric", 
        "Number of Neurons", "Number of Neurons in Approximator", "Causal Effect", 
        "Standard Error", "p-value"]

    values = ["Regression", its.regularized, its.activation, its.validation_metric, 
        its.num_neurons, its.approximator_neurons, effect, stderr, p]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

"""
    summarize(mod, n)

Return a summary from a CausalEstimator or Metalearner.

p-values and standard errors are estimated using approximate randomization inference.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = GComputation(X, T, Y)
julia> estimate_causal_effect!(m1)
 0.3100468253
julia> summarize(m1)
 {"Task" => "Regression", "Quantity of Interest" => "ATE", Regularized" => "true", 
 "Activation Function" => "relu", "Time Series/Panel Data" => "false", 
 "Validation Metric" => "mse","Number of Neurons" => "5", 
 "Number of Neurons in Approximator" => "10", "Causal Effect: 0.00589761, 
 "Standard Error" => 5.12900734, "p-value" => 0.479011245} 
```

```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = RLearner(X, T, Y)
julia> estimate_causal_effect(m1)
1-element Vector{Float64}
 [0.5804032956]
julia> summarize(m1)
 {"Task" => "Regression", "Quantity of Interest" => "ATE", Regularized" => "true", 
 "Activation Function" => "relu", "Validation Metric" => "mse", "Number of Neurons" => "5", 
 "Number of Neurons in Approximator" => "10", "Causal Effect" = 0.5804032956, 
 "Standard Error" => 2.129400324, "p-value" => 0.0008342356}
```

```jldoctest
julia> X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = SLearner(X, T, Y)
julia> estimate_causal_effect!(m1)
100-element Vector{Float64}
 0.20729633391630697
 0.20729633391630697
 0.20729633391630692
 ⋮
 0.20729633391630697
 0.20729633391630697
 0.20729633391630697
julia> summarise(m1)
 {"Task" => "Regression", Regularized" => "true", "Activation Function" => "relu", 
 "Validation Metric" => "mse", "Number of Neurons" => "5", 
 "Number of Neurons in Approximator" => "10", 
 "Causal Effect: [0.20729633391630697, 0.20729633391630697, 0.20729633391630692, 
 0.20729633391630697, 0.20729633391630697, 0.20729633391630697, 0.20729633391630697, 
 0.20729633391630703, 0.20729633391630697, 0.20729633391630697  …  0.20729633391630703, 
 0.20729633391630697, 0.20729633391630692, 0.20729633391630703, 0.20729633391630697, 
 0.20729633391630697, 0.20729633391630692, 0.20729633391630697, 0.20729633391630697, 
 0.20729633391630697], "Standard Error" => 5.3121435085, "p-value" => 0.0632454855}
```
"""
function summarize(mod, n=1000)
    if !isdefined(mod, :causal_effect) || mod.causal_effect === NaN
        throw(ErrorException("call estimate_causal_effect! before calling summarize"))
    end

    summary_dict = Dict()
    task = typeof(mod) == DoubleMachineLearning ? "regression" : mod.task
    nicenames = ["Task", "Quantity of Interest", "Regularized", "Activation Function", 
        "Time Series/Panel Data", "Validation Metric", "Number of Neurons", 
        "Number of Neurons in Approximator", "Causal Effect", "Standard Error", 
        "p-value"]
    
    p, stderr = quantities_of_interest(mod, n)

    values = [task, mod.quantity_of_interest, mod.regularized, mod.activation, mod.temporal, 
        mod.validation_metric, mod.num_neurons, mod.approximator_neurons, mod.causal_effect, 
        stderr, p]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

summarize(R::RLearner, n=1000) = summarize(R.dml, n)

summarize(S::SLearner, n=1000) = summarize(S.g, n)

"""
    generate_null_distribution(mod, n)

Generate a null distribution for the treatment effect of G-computation, double machine 
learning, or metalearning.

This method estimates the same model that is provided using random permutations of the 
treatment assignment to generate a vector of estimated effects under different treatment
regimes. When mod is a metalearner the null statistic is the difference is the ATE.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

Examples
```jldoctest
julia> x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
julia> g_computer = GComputation(x, t, y)
julia> estimate_causal_effect!(g_computer)
julia> generate_null_distribution(g_computer, 500)
500-element Vector{Float64}
500-element Vector{Float64}
 0.016297180690693656
 0.0635928694685571
 0.20004144093635673
 ⋮
 24.739658523175912
 25.30523686137909
 28.07474553316176
```
"""
function generate_null_distribution(mod, n)
    local m = deepcopy(mod)
    nobs = size(m.T, 1)
    results = Vector{Float64}(undef, n)
    
    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n 

        # Sample from a continuous distribution if the treatment is continuous
        if var_type(mod.T) isa Continuous
            m.T = (maximum(m.T)-minimum(m.T)).*rand(nobs).+minimum(m.T)
        else
            m.T = float(rand(unique(m.T), nobs))
        end
        
        estimate_causal_effect!(m)
        results[iter] = mod isa Metalearner ? mean(m.causal_effect) : m.causal_effect
    end
    return results
end

"""
    generate_null_distribution(its, n, mean_effect)

Generate a null distribution for the treatment effect in an interrupted time series 
analysis. By default, this method generates a null distribution of mean differences. To 
generate a null distribution of cummulative differences, set the mean_effect argument to 
false.

Randomization is conducted by randomly assigning observations to the pre and 
post-intervention periods, resestimating the causal effect, and repeating n times. The null 
distribution is the set of n casual effect estimates.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```jldoctest
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causale_ffect!(its)
julia> generate_null_distribution(its, 10)
10-element Vector{Float64}
 -0.5012456678829079
 -0.33790650529972194
 -0.2534340182760628
 ⋮
 -0.06217013151235991 
 -0.05905529159312335
 -0.04927743270606937
```
"""
function generate_null_distribution(its::InterruptedTimeSeries, n, mean_effect)
    local model = deepcopy(its)
    split_idx = size(model.Y₀, 1)
    results = Vector{Float64}(undef, n)
    data = reduce(hcat, (reduce(vcat, (its.X₀, its.X₁)), reduce(vcat, (its.Y₀, its.Y₁))))

    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n
        permuted_data = data[shuffle(1:end), :]
        permuted_x₀ = permuted_data[1:split_idx, 1:end-1]
        permuted_x₁ = permuted_data[split_idx+1:end, 1:end-1]
        permuted_y₀ = permuted_data[1:split_idx, end]
        permuted_y₁ = permuted_data[split_idx+1:end, end]

        # Reestimate the model with the intervention now at the nth interval
        model.X₀, model.Y₀ = permuted_x₀, permuted_y₀
        model.X₁, model.Y₁ = permuted_x₁, permuted_y₁
        estimate_causal_effect!(model)
        results[iter] = ifelse(mean_effect, mean(model.Δ), sum(model.Δ))
    end
    return results
end

"""
    quantities_of_interest(mod, n)

Generate a p-value and standard error through randomization inference

This method generates a null distribution of treatment effects by reestimating treatment 
effects from permutations of the treatment vector and estimates a p-value and standard from
the generated distribution.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

For a primer on randomization inference see:
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```jldoctest
julia> x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
julia> g_computer = GComputation(x, t, y)
julia> estimate_causal_effect!(g_computer)
julia> quantities_of_interest(g_computer, 1000)
 (0.114, 6.953133617011371)
```
"""
function quantities_of_interest(mod, n)
    local null_dist = generate_null_distribution(mod, n)
    local avg_effect = mod isa Metalearner ? mean(mod.causal_effect) : mod.causal_effect

    extremes = length(null_dist[abs(avg_effect) .< abs.(null_dist)])
    pvalue = extremes/n

    stderr = sqrt(sum([(avg_effect .- x)^2 for x in null_dist])/(n-1)) / sqrt(n)

    return pvalue, stderr
end

"""
    quantities_of_interest(mod, n)

Generate a p-value and standard error through randomization inference

This method generates a null distribution of treatment effects by reestimating treatment 
effects from permutations of the treatment vector and estimates a p-value and standard from 
the generated distribution. Randomization for event studies is done by creating time splits 
at even intervals and reestimating the causal effect.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

For a primer on randomization inference see:
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```jldoctest
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> quantities_of_interest(its, 10)
 (0.0, 0.07703275541001667)
```
"""
function quantities_of_interest(mod::InterruptedTimeSeries, n, mean_effect)
    local null_dist = generate_null_distribution(mod, n, mean_effect)
    local metric = ifelse(mean_effect, mean, sum)
    local effect = metric(mod.Δ)

    extremes = length(null_dist[effect .< abs.(null_dist)])
    pvalue = extremes/n

    stderr = (sum([(effect .- x)^2 for x in null_dist])/(n-1))/sqrt(n)

    return pvalue, stderr
end
