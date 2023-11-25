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

export InterruptedTimeSeries, GComputation, DoubleMachineLearning 
export SLearner, TLearner, XLearner
export estimate_causal_effect!, summarize

# Allows dispatching e_value methods based on the type of outcome variable
struct Discrete end
struct Continuous end

# Check the type of outcome data
function variable_type(y::Vector{<:Real})
    y_sorted = sort(y)
    
    if Set(y_sorted) == Set([0, 1]) || Set(y_sorted) == Set(round.(y_sorted))
        return Discrete()
    else
        return Continuous()
    end
end

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
