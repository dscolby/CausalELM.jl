module Assumptions

using ..Estimators: InterruptedTimeSeries, estimatecausaleffect!

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
function testassumptions(its::InterruptedTimeSeries)
end

"""
    testcovariateindependence(its; n)

Test for independence between covariates and the event or intervention.

If the covariates used to predict the counterfactual outcomes are effected by the 
event/intervention then they will not be able to predict the counterfactual outcomes. 
p-values from this test represent the proportion of times that a placebo treatment had a 
greater estimated effect on a covariate than the actual treatment assignment. Thus, the
the lower the p-value is for a variable the more evidence there is that the variable was 
effected by the event/treatment and is not useful in predicting counterfactual outcomes.

Examples
```julia-repl
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimatecausaleffect!(its)
julia> testcovariateindependence(its)
Dict("Column 1 p-value" => 0.421, "Column 5 p-value" => 0.07, "Column 3 p-value" => 0.01, 
"Column 2 p-value" => 0.713, "Column 4 p-value" => 0.043)
```
"""
function testcovariateindependence(its::InterruptedTimeSeries, n::Int=1000)
    y₀ = reduce(hcat, (its.X₀[:, 1:end-1], zeros(size(its.X₀, 1))))
    y₁ = reduce(hcat, (its.X₁[:, 1:end-1], ones(size(its.X₁, 1))))
    all_vars = [reduce(vcat, (y₀, y₁)) ones(size(y₀, 1) + size(y₁, 1))]
    x = all_vars[:, end-1:end]
    results = Dict{String, Float64}()

    for i in 1:size(all_vars, 2)-2
        y = all_vars[:, i]
        β = first(x\y)
        p = pval(x, y, β, n=n)
        results["Column " * string(i) * " p-value"] = p
    end
    return results
end

"""
    omittedvariable(its; iterations)

See how an omitted variable would affect the results of an interrupted time series analysis.

This method reestimates interrupted time series models with random variables simulated from 
normal distributions. If the included covariates are good predictors of the 
counterfactual outcome, including a random variable should make little to no difference on 
the estimated treatment effect.

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
function omittedvariable(its::InterruptedTimeSeries)
end

"""
    pval(x, y, β; n)

Estimate the p-value for the hypothesis that an event had a statistically significant effect 
on the slope of a covariate using randomization inference.

Examples
```julia-repl
julia> x, y, β = reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 0.5
julia> pval(x, y, β)
0.98
julia> pval(x, y, β, n=100)
0.08534054
```
"""
function pval(x::Array{Float64}, y::Array{Float64}, β::Float64; n::Int=1000)
    m2 = "the first column of x should be a treatment vector of 0s and 1s"
    if size(x, 2) !== 2
        throw(ArgumentError("x should only contain treatment and intercept columns"))
    end

    if sort(union(x[:, 1], [0, 1])) != [0, 1]
        throw(ArgumentError(m2))
    end

    l = unique(x[:, 2])

    if length(l) !== 1 | (length(l) === 1 && l[1] !== 1.0)
        throw(ArgumentError("the second column in x should be an intercept with all 1s"))
    end

    x_copy = deepcopy(x)
    nulldist = Vector{Float64}(undef, n)
    for i in 1:n
        x_copy[:, 1] = float(rand(0:1, size(x, 1)))  # Random treatment vector
        nulldist[i] = first(x_copy\y)
    end
    p = length(nulldist[abs(β) .< abs.(nulldist)])/n

    return p
end
    
end
