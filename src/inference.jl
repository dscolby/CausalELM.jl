using Random: shuffle

"""
    summarize(mod, kwargs...)

Get a summary from a CausalEstimator or Metalearner.

# Arguments
- `mod::Union{CausalEstimator, Metalearner}`: a model to summarize.

# Keywords
- `n::Int=1000`: the number of iterations to generate the numll distribution for 
    randomization inference if inference is true.
- `inference::Bool`=false: wheteher calculate p-values and standard errors.
- `mean_effect::Bool=true`: whether to estimate the mean or cumulative effect for an 
    interrupted time series estimator.

# Notes
p-values and standard errors are estimated using approximate randomization inference. If set 
to true, this procedure takes a long time due to repeated matrix inversions. You can greatly 
speed this up by setting to a lower number and launching Julia with more threads.

# References
For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> X, T, Y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)
julia> m1 = GComputation(X, T, Y)
julia> estimate_causal_effect!(m1)
julia> summarize(m1)

julia> m2 = RLearner(X, T, Y)
julia> estimate_causal_effect(m2)
julia> summarize(m2)

julia> m3 = SLearner(X, T, Y)
julia> estimate_causal_effect!(m3)
julia> summarise(m3)  # British spelling works too!
```
"""
function summarize(mod; kwargs...)
    if all(isnan, mod.causal_effect)
        throw(ErrorException("call estimate_causal_effect! before calling summarize"))
    end

    summary_dict = Dict()
    nicenames = [
        "Task",
        "Quantity of Interest",
        "Activation Function",
        "Sample Size",
        "Number of Machines",
        "Number of Features",
        "Number of Neurons",
        "Time Series/Panel Data",
        "Causal Effect",
        "Standard Error",
        "p-value",
        "Lower 2.5% CI",
        "Upper 97.5% CI"
    ]

    if haskey(kwargs, :inference) && kwargs[:inference] == true
        iters = haskey(kwargs, :n) ? kwargs[:n] : 1000
        p, stderr, lower_ci, upper_ci = quantities_of_interest(mod, iters)
    else
        p, stderr, lower_ci, upper_ci = NaN, NaN, NaN, NaN
    end

    values = [
        mod.task,
        mod.quantity_of_interest,
        mod.activation,
        mod.sample_size,
        mod.num_machines,
        mod.num_feats,
        mod.num_neurons,
        mod.temporal,
        mod.causal_effect,
        stderr,
        p,
        lower_ci,
        upper_ci
    ]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

function summarize(its::InterruptedTimeSeries; kwargs...)
    if all(isnan, its.causal_effect)
        throw(ErrorException("call estimate_causal_effect! before calling summarize"))
    end

    if haskey(kwargs, "mean_effect") && kwargs[:mean_effect] == true
        effect, qoi = mean(its.causal_effect), "Mean Difference"
    else
        effect, qoi = sum(its.causal_effect), "Cumulative Difference"
    end

    if haskey(kwargs, :inference) && kwargs[:inference] == true
        iters = haskey(kwargs, :n) ? kwargs[:n] : 100
        p, stderr, l, u = quantities_of_interest(its, iters, effect == "Mean Difference")
    else
        p, stderr, l, u = NaN, NaN, NaN, NaN
    end

    summary_dict = Dict()
    nicenames = [
        "Task",
        "Quantity of Interest",
        "Activation Function",
        "Sample Size",
        "Number of Machines",
        "Number of Features",
        "Number of Neurons",
        "Time Series/Panel Data",
        "Causal Effect",
        "Standard Error",
        "p-value",
        "Lower 2.5% CI",
        "Upper 97.5% CI"
    ]

    values = [
        its.task,
        qoi,
        its.activation,
        its.sample_size,
        its.num_machines,
        its.num_feats,
        its.num_neurons,
        its.temporal,
        effect,
        stderr,
        p,
        l,
        u
    ]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

"""
    generate_null_distribution(mod, n)
    generate_null_distribution(mod, n, mean_effect)

Generate a null distribution for the treatment effect of G-computation, double machine 
learning, or metalearning.

# Arguments
- `mod::Any`: model to summarize.
- `n::Int=100`: number of iterations to generate the null distribution for randomization 
    inference.
- `mean_effect::Bool=true`: whether to estimate the mean or cumulative effect for an 
    interrupted time series estimator.

# Notes
This method estimates the same model that is provided using random permutations of the 
treatment assignment to generate a vector of estimated effects under different treatment
regimes. When mod is a metalearner the null statistic is the difference is the ATE.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

# Examples
```julia
julia> x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
julia> g_computer = GComputation(x, t, y)
julia> estimate_causal_effect!(g_computer)
julia> generate_null_distribution(g_computer, 500)
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> generate_null_distribution(its, 10)
```
"""
function generate_null_distribution(mod, n)
    nobs = size(mod.T, 1)
    t_min, t_max = minimum(mod.T), maximum(mod.T)
    t_range = t_max - t_min
    results = Vector{Float64}(undef, n)

    # Generate random treatment assignments and estimate their causal effects
    Threads.@threads for i ∈ 1:n
        model = deepcopy(mod)

        # Sample from a continuous distribution if the treatment is continuous
        if var_type(mod.T) isa Continuous
            model.T = t_range .* rand(nobs) .+ t_min
        else
            model.T = float(rand(unique(mod.T), nobs))
        end

        estimate_causal_effect!(model)
        results[i] = mod isa Metalearner ? mean(model.causal_effect) : model.causal_effect
    end
    return results
end

function generate_null_distribution(its::InterruptedTimeSeries, n, mean_effect)
    data = reduce(hcat, (reduce(vcat, (its.X₀, its.X₁)), reduce(vcat, (its.Y₀, its.Y₁))))
    min_idx, max_idx = 2, size(data, 1) - 1
    indices = rand(min_idx:max_idx, n)
    results = Vector{Float64}(undef, n)
    
    # Reestimate the model with the intervention now at the nth time period
    Threads.@threads for iter in 1:n
        x₀, y₀ = data[1:indices[iter], 1:(end - 1)], data[1:indices[iter], end]
        x₁ = data[(indices[iter] + 1):end, 1:(end - 1)]
        y₁ = data[(indices[iter] + 1):end, end]
        model = deepcopy(its)
        model.X₀, model.Y₀, model.X₁, model.Y₁ = x₀, y₀, x₁, y₁
        estimate_causal_effect!(model)
        results[iter] = mean_effect ? mean(model.causal_effect) : sum(model.causal_effect)
    end
    return results
end

"""
    quantities_of_interest(mod, n)
    quantities_of_interest(mod, n, mean_effect)

Generate a p-value and standard error through randomization inference

This method generates a null distribution of treatment effects by reestimating treatment 
effects from permutations of the treatment vector and estimates a p-value and standard from
the generated distribution.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

For a primer on randomization inference see:
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
julia> g_computer = GComputation(x, t, y)
julia> estimate_causal_effect!(g_computer)
julia> quantities_of_interest(g_computer, 1000)
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> quantities_of_interest(its, 10)
```
"""
function quantities_of_interest(mod, n)
    null_dist = generate_null_distribution(mod, n)
    avg_effect = mod isa Metalearner ? mean(mod.causal_effect) : mod.causal_effect
    pvalue, stderr = p_value_and_std_err(null_dist, avg_effect)
    lb, ub = confidence_interval(null_dist)

    return pvalue, stderr, lb, ub
end

function quantities_of_interest(mod::InterruptedTimeSeries, n, mean_effect)
    null_dist = generate_null_distribution(mod, n, mean_effect)
    metric = ifelse(mean_effect, mean, sum)
    effect = metric(mod.causal_effect)
    pvalue, stderr = p_value_and_std_err(null_dist, effect)
    lb, ub = confidence_interval(null_dist)

    return pvalue, stderr, lb, ub
end

"""
    confidence_interval(null_dist)

Compute 95% confidence intervals via randomization inference.

This function should not be called directly by the user.

For a primer on randomization inference see:
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
julia> g_computer = GComputation(x, t, y)
julia> estimate_causal_effect!(g_computer)
julia> null_dist = CausalELM.generate_null_distribution(g_computer, 1000)
julia> confidence_interval(null_dist)
(-0.45147664642089147, 0.45147664642089147)
```
"""
function confidence_interval(null_dist)
    sorted_null_dist, n = sort(null_dist), length(null_dist)
    low_idx, high_idx = 0.025 * (n - 1), 0.975 * (n - 1)

    lb = if isinteger(low_idx)
        sorted_null_dist[Int(low_idx)]
    else
        mean(sorted_null_dist[floor(Int, low_idx):ceil(Int, low_idx)])
    end

    ub = if isinteger(high_idx)
        sorted_null_dist[Int(high_idx)]
    else
        mean(sorted_null_dist[floor(Int, high_idx):ceil(Int, high_idx)])
    end

    return lb, ub
end

"""
    p_value_and_std_err(null_dist, test_stat)

Compute the p-value for a given test statistic and null distribution.

This is an approximate method based on randomization inference that does not assume any 
parametric form of the null distribution.

For a primer on randomization inference see:
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
julia> g_computer = GComputation(x, t, y)
julia> estimate_causal_effect!(g_computer)
julia> null_dist = CausalELM.generate_null_distribution(g_computer, 1000)
julia> p_value_and_std_err(null_dist, CausalELM.mean(null_dist))
(0.3758916871866841, 0.1459779344550146)
```
"""
function p_value_and_std_err(null_dist, test_stat)
    n = length(null_dist)
    extremes = length(null_dist[abs(test_stat) .<= null_dist])
    pvalue = extremes / n
    stderr = (sum([(test_stat .- x)^2 for x in null_dist]) / (n - 1)) / sqrt(n)

    return pvalue, stderr
end
