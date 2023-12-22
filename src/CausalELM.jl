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
 
export validate
export hard_tanh, elish, fourier
export SLearner, TLearner, XLearner
export estimate_causal_effect!, summarize
export mse, mae, accuracy, precision, recall, F1
export InterruptedTimeSeries, GComputation, DoubleMachineLearning
export binary_step, σ, tanh, relu, leaky_relu, swish, softmax, softplus, gelu, gaussian

include("utilities.jl")
include("activation.jl")
include("models.jl")
include("metrics.jl")
include("crossval.jl")
include("estimators.jl")
include("metalearners.jl")
include("inference.jl")
include("model_validation.jl")

# So that it works with British spelling
const summarise = summarize

end
