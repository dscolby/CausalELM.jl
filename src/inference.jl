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
    ]

    if haskey(kwargs, :inference) && kwargs[:inference] == true
        iters = haskey(kwargs, :n) ? kwargs[:n] : 1000
        p, stderr = quantities_of_interest(mod, iters)
    else
        p, stderr = NaN, NaN
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
        p, stderr = quantities_of_interest(its, iters, effect == "Mean Difference")
    else
        p, stderr = NaN, NaN
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
    split_idx = size(its.Y₀, 1)
    results = Vector{Float64}(undef, n)
    data = reduce(hcat, (reduce(vcat, (its.X₀, its.X₁)), reduce(vcat, (its.Y₀, its.Y₁))))

    # Generate random treatment assignments and estimate the causal effects
    Threads.@threads for iter in 1:n
        permuted_data = data[shuffle(1:end), :]
        permuted_x₀ = permuted_data[1:split_idx, 1:(end - 1)]
        permuted_x₁ = permuted_data[(split_idx + 1):end, 1:(end - 1)]
        permuted_y₀ = permuted_data[1:split_idx, end]
        permuted_y₁ = permuted_data[(split_idx + 1):end, end]
        model = deepcopy(its)

        # Reestimate the model with the intervention now at the nth interval
        model.X₀, model.Y₀ = permuted_x₀, permuted_y₀
        model.X₁, model.Y₁ = permuted_x₁, permuted_y₁
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

    extremes = length(null_dist[abs(avg_effect) .< abs.(null_dist)])
    pvalue = extremes / n

    stderr = sqrt(sum([(avg_effect .- x)^2 for x in null_dist]) / (n - 1)) / sqrt(n)

    return pvalue, stderr
end

function quantities_of_interest(mod::InterruptedTimeSeries, n, mean_effect)
    local null_dist = generate_null_distribution(mod, n, mean_effect)
    local metric = ifelse(mean_effect, mean, sum)
    local effect = metric(mod.causal_effect)

    extremes = length(null_dist[effect .< abs.(null_dist)])
    pvalue = extremes / n

    stderr = (sum([(effect .- x)^2 for x in null_dist]) / (n - 1)) / sqrt(n)

    return pvalue, stderr
end
