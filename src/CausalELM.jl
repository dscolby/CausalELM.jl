"""
Macros, functions, and structs for applying Extreme Learning Machines to causal inference
tasks where the counterfactual is unavailable or biased and must be predicted. Provides 
macros for event study designs, parametric G-computation, doubly robust estimation, and 
metalearners. Additionally, these tasks can be performed with or without L2 penalization and
will automatically choose the best number of neurons and L2 penalty. 

For more details on Extreme Learning Machines see:
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.
"""
module CausalELM

export InterruptedTimeSeries, GComputation, DoublyRobust 
export SLearner, TLearner, XLearner
export estimate_causal_effect!, summarize

"""
    estimate_causal_effect!(its)

Estimate the effect of an event relative to a predicted counterfactual.

Examples
```julia-repl
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
0.25714308
```
"""

"""
    estimate_causal_effect!(g)

Estimate a causal effect of interest using G-Computation.

If treatents are administered at multiple time periods, the effect will be estimated as the 
average difference between the outcome of being treated in all periods and being treated in 
no periods.For example, given that individuals 1, 2, ..., i ∈ I recieved either a treatment 
or a placebo in p different periods, the model would estimate the average treatment effect 
as E[Yᵢ|T₁=1, T₂=1, ... Tₚ=1, Xₚ] - E[Yᵢ|T₁=0, T₂=0, ... Tₚ=0, Xₚ].

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m2 = GComputation(X, Y, T)
julia> estimate_causal_effect!(m2)
0.31067439
```
"""

"""
    estimate_causal_effect!(DRE)

Estimate a causal effect of interest using doubly robust estimation.

Unlike other estimators, this method does not support time series or panel data. This method 
also does not work as well with smaller datasets because it estimates separate outcome 
models for the treatment and control groups.

Examples
```julia-repl
julia> X, Xₚ, Y, T =  rand(100, 5), rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m3 = DoublyRobust(X, Xₚ, Y, T)
julia> estimate_causal_effect!(m3)
0.31067439
```
"""

"""
    estimate_causal_effect!(m)

Estimate the CATE using a metalearner.

For an overview of meatlearning see:

    Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
    estimating heterogeneous treatment effects using machine learning." Proceedings of the 
    national academy of sciences 116, no. 10 (2019): 4156-4165.

Examples
```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m4 = SLearner(X, Y, T)
julia> estimate_causal_effect!(m4)
[0.20729633391630697, 0.20729633391630697, 0.20729633391630692, 0.20729633391630697, 
0.20729633391630697, 0.20729633391630697, 0.20729633391630697, 0.20729633391630703, 
0.20729633391630697, 0.20729633391630697  …  0.20729633391630703, 0.20729633391630697, 
0.20729633391630692, 0.20729633391630703, 0.20729633391630697, 0.20729633391630697, 
0.20729633391630692, 0.20729633391630697, 0.20729633391630697, 0.20729633391630697]
```

```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m5 = TLearner(X, Y, T)
julia> estimatecausaleffect!(m5)
[0.0493951571746305, 0.049395157174630444, 0.0493951571746305, 0.049395157174630444, 
0.04939515717463039, 0.04939515717463039, 0.04939515717463039, 0.04939515717463039, 
0.049395157174630444, 0.04939515717463061  …  0.0493951571746305, 0.04939515717463039, 
0.0493951571746305, 0.04939515717463039, 0.0493951571746305, 0.04939515717463039, 
0.04939515717463039, 0.049395157174630444, 0.04939515717463039, 0.049395157174630444]
```

```julia-repl
julia> X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]
julia> m1 = XLearner(X, Y, T)
julia> estimatecausaleffect!(m1)
[-0.025012644892878473, -0.024634294305967294, -0.022144246680543364, -0.023983138957276127, 
-0.024756239357838557, -0.019409519377053822, -0.02312807640357356, -0.016967113188439076, 
-0.020188871831409317, -0.02546526148141366  …  -0.019811641136866287, 
-0.020780821058711863, -0.013588359417922776, -0.020438648396328824, -0.016169487825519843, 
-0.024031422484491572, -0.01884713946778991, -0.021163590874553318, -0.014607310062509895, 
-0.022449034332142046]
```
"""
function estimate_causal_effect!() end

function summarize() end

mean(x::Vector{<:Real}) = sum(x)/length(x)

function var(x::Vector{<:Real})
    x̄, n = mean(x), length(x)

    return sum((x .- x̄).^2)/(n-1)
end

const summarise = summarize

# Helpers to subtract or add consecutive elements in a vector
consecutive(v::Vector{<:Real}) = [-(v[i+1], v[i]) for i = 1:length(v)-1]

include("activation.jl")
include("models.jl")
include("metrics.jl")
include("crossval.jl")
include("estimators.jl")
include("metalearners.jl")
include("inference.jl")
include("model_validation.jl")

end
