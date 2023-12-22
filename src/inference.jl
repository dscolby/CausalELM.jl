using Random: shuffle

"""
    summarize(its, mean_effect)

Return a summary from an interrupted time series estimator.

p-values and standard errors are estimated using approximate randomization inference that
permutes the time of the intervention.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```julia-repl
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimatetreatmenteffect!(m1)
[0.25714308]
julia> summarize(m1)
{"Task" => "Regression", "Regularized" => true, "Activation Function" => relu, 
"Validation Metric" => "mse","Number of Neurons" => 2, 
"Number of Neurons in Approximator" => 10, "β" => [0.25714308], 
"Causal Effect" => -3.9101138, "Standard Error" => 1.903434356, "p-value" = 0.00123356}
```
"""
function summarize(its::InterruptedTimeSeries, n::Integer=1000, mean_effect::Bool=true)
    if !isdefined(its, :Δ)
        throw(ErrorException("call estimatecausaleffect! before calling summarize"))
    end


    effect = ifelse(mean_effect, mean(its.Δ), sum(its.Δ))

    p, stderr = quantities_of_interest(its, n, mean_effect)

    summary_dict = Dict()
    nicenames = ["Task", "Regularized", "Activation Function", "Validation Metric", 
        "Number of Neurons", "Number of Neurons in Approximator", "β", "Causal Effect", 
        "Standard Error", "p-value"]

    values = [its.task, its.regularized, its.activation, its.validation_metric, 
        its.num_neurons, its.approximator_neurons, its.β, effect, stderr, p]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

"""
    summarize(g)

Return a summary from a G-Computation estimator.

p-values and standard errors are estimated using approximate randomization inference.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = GComputation(X, Y, T)
julia> estimate_causal_effect!(m1)
[0.3100468253]
julia> summarize(m1)
{"Task" => "Regression", "Quantity of Interest" => "ATE", Regularized" => "true", 
"Activation Function" => "relu", "Time Series/Panel Data" => "false", 
"Validation Metric" => "mse","Number of Neurons" => "5", 
"Number of Neurons in Approximator" => "10", "β" => "[0.3100468253]",
"Causal Effect: 0.00589761, "Standard Error" => 5.12900734, "p-value" => 0.479011245} 
```
"""
function summarize(g::GComputation, n::Integer=1000)
    summary_dict = Dict()
    nicenames = ["Task", "Quantity of Interest", "Regularized", "Activation Function", 
        "Time Series/Panel Data", "Validation Metric", "Number of Neurons", 
        "Number of Neurons in Approximator", "β", "Causal Effect", "Standard Error", 
        "p-value"]
    
    p, stderr = quantities_of_interest(g, n)

    values = [g.task, g.quantity_of_interest, g.regularized, g.activation, g.temporal, 
        g.validation_metric, g.num_neurons, g.approximator_neurons, g.β, g.causal_effect,
        stderr, p]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

"""
    summarize(dml, n)

Return a summary from a doubly robust estimator.

p-values and standard errors are estimated using approximate randomization inference.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = DoubleMachineLearning(X, X, Y, T)
julia> estimate_causal_effect(m1)
[0.5804032956]
julia> summarize(m1)
{"Task" => "Regression", "Quantity of Interest" => "ATE", Regularized" => "true", 
"Activation Function" => "relu", "Validation Metric" => "mse", "Number of Neurons" => "5", 
"Number of Neurons in Approximator" => "10", "Causal Effect" = 0.5804032956, 
"Standard Error" => 2.129400324, "p-value" => 0.0008342356}
```
"""
function summarize(dml::DoubleMachineLearning, n::Integer=1000)
    summary_dict = Dict()
    nicenames = ["Task", "Quantity of Interest", "Regularized", "Activation Function", 
        "Validation Metric", "Number of Neurons", "Number of Neurons in Approximator", 
        "Causal Effect", "Standard Error", "p-value"]

    p, stderr = quantities_of_interest(dml, n)

    values = [dml.task, "ATE", dml.regularized, dml.activation,  dml.validation_metric, 
        dml.num_neurons, dml.approximator_neurons, dml.causal_effect, stderr, p]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end


"""
    summarize(m, n)

Return a summary from a metalearner.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = SLearner(X, Y, T)
julia> estimate_causal_effect!(m1)
[0.20729633391630697, 0.20729633391630697, 0.20729633391630692, 0.20729633391630697, 
0.20729633391630697, 0.20729633391630697, 0.20729633391630697, 0.20729633391630703, 
0.20729633391630697, 0.20729633391630697  …  0.20729633391630703, 0.20729633391630697, 
0.20729633391630692, 0.20729633391630703, 0.20729633391630697, 0.20729633391630697, 
0.20729633391630692, 0.20729633391630697, 0.20729633391630697, 0.20729633391630697]
julia> summarise(m1)
{"Task" => "Regression", Regularized" => "true", "Activation Function" => "relu", 
"Validation Metric" => "mse", "Number of Neurons" => "5", 
"Number of Neurons in Approximator" => "10", 
"β" => "[0.3100468253]", "Causal Effect: [0.20729633391630697, 0.20729633391630697, 
0.20729633391630692, 0.20729633391630697, 0.20729633391630697, 0.20729633391630697, 
0.20729633391630697, 0.20729633391630703, 0.20729633391630697, 0.20729633391630697  …  
0.20729633391630703, 0.20729633391630697, 0.20729633391630692, 0.20729633391630703, 
0.20729633391630697, 0.20729633391630697, 0.20729633391630692, 0.20729633391630697, 
0.20729633391630697, 0.20729633391630697], "Standard Error" => 5.3121435085, 
"p-value" => 0.0632454855}
```
"""
function summarize(m::Metalearner, n::Integer=1000)
    summary_dict = Dict()
    nicenames = ["Task", "Regularized", "Activation Function",  "Validation Metric", 
        "Number of Neurons", "Number of Neurons in Approximator", "Causal Effect", 
        "Standard Error", "p-value"]

    p, stderr = quantities_of_interest(m, n)

    values = [m.task, m.regularized, m.activation, m.validation_metric, m.num_neurons, 
        m.approximator_neurons, m.causal_effect, stderr, p]

    for (nicename, value) in zip(nicenames, values)
        summary_dict[nicename] = value
    end

    return summary_dict
end

"""
    generate_null_distribution(e, n)

Generate a null distribution for the treatment effect of G-computation, doubly robust 
estimation, or metalearning.

This method estimates the same model that is provided using random permutations of the 
treatment assignment to generate a vector of estimated effects under different treatment
regimes. When e is a metalearner the null statistic is the difference is the ATE.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

Examples
```julia-repl
julia> x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
julia> g_computer = GComputation(x, y, t)
julia> estimate_causal_effect!(g_computer)
julia> generate_null_distribution(g_computer, 500)
[0.016297180690693656, 0.0635928694685571, 0.20004144093635673, 0.505893866040335, 
0.5130594630907543, 0.5432486130493388, 0.6181727442724846, 0.61838399963459, 
0.7038981488009489, 0.7043407710415689  …  21.909186142780246, 21.960498059428854, 
21.988553083790023, 22.285403459215363, 22.613625375395973, 23.382102081355548, 
23.52056245175936, 24.739658523175912, 25.30523686137909, 28.07474553316176]
```
"""
function generate_null_distribution(e::Union{CausalEstimator, Metalearner}, n::Integer=1000)
    local m = deepcopy(e)
    nobs = size(m.T, 1)
    results = Vector{Float64}(undef, n)
    
    # Generate random treatment assignments and estimate the causal effects
    for iter in 1:n 
        m.T = float(rand(unique(m.T), nobs))
        estimate_causal_effect!(m)
        results[iter] = e isa Metalearner ? mean(m.causal_effect) : m.causal_effect
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
```julia-repl
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causale_ffect!(its)
julia> generate_null_distribution(its, 10)
[-0.5012456678829079, -0.33790650529972194, -0.2534340182760628, -0.21030239864895905, 
-0.11672915615117885, -0.08149441936166794, -0.0685134758182695, -0.06217013151235991, 
-0.05905529159312335, -0.04927743270606937]
```
"""
function generate_null_distribution(its::InterruptedTimeSeries, n::Integer=1000, 
    mean_effect::Bool=true)
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
    quantities_of_interest(model, n)

Generate a p-value and standard error through randomization inference

This method generates a null distribution of treatment effects by reestimating treatment 
effects from permutations of the treatment vector and estimates a p-value and standard from
the generated distribution.

Note that lowering the number of iterations increases the probability of failing to reject
the null hypothesis.

For a primer on randomization inference see:
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```julia-repl
julia> x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
julia> g_computer = GComputation(x, y, t)
julia> estimate_causal_effect!(g_computer)
julia> quantities_of_interest(g_computer, 1000)
(0.114, 6.953133617011371)
```
"""
function quantities_of_interest(m::Union{CausalEstimator, Metalearner}, n::Integer=1000)
    local null_dist = generate_null_distribution(m, n)
    local avg_effect = m isa Metalearner ? mean(m.causal_effect) : m.causal_effect

    extremes = length(null_dist[abs(avg_effect) .< abs.(null_dist)])
    pvalue = extremes/n

    stderr = sqrt(sum([(avg_effect .- x)^2 for x in null_dist])/(n-1)) / sqrt(n)

    return pvalue, stderr
end

"""
    quantities_of_interest(model, n)

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
```julia-repl
julia> x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> quantities_of_interest(its, 10)
(0.0, 0.07703275541001667)
```
"""
function quantities_of_interest(model::InterruptedTimeSeries, n::Integer=1000, 
    mean_effect::Bool=true)
    local null_dist = generate_null_distribution(model, n, mean_effect)
    local metric = ifelse(mean_effect, mean, sum)
    local effect = metric(model.Δ)

    extremes = length(null_dist[effect .< abs.(null_dist)])
    pvalue = extremes/n

    stderr = (sum([(effect .- x)^2 for x in null_dist])/(n-1))/sqrt(n)

    return pvalue, stderr
end
