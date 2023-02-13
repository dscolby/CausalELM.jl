"""
Macros, functions, and structs for applying Extreme Learning Machines to causal inference
tasks where the counterfactual is unavailable or biased and must be predicted. Provides 
macros for event study designs, parametric G-computation, doubly robust machine learning, and 
metalearners. Additionally, these tasks can be performed with or without L2 penalization and
will automatically choose the best number of neurons and L2 penalty. 

For more details on Extreme Learning Machines see:
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.
"""
module CausalELM

export summarise, summarize

include("activation.jl")
using .ActivationFunctions
export binarystep, σ, tanh, relu, leakyrelu, swish, softmax, softplus, gelu, gaussian, 
    hardtanh, elish, fourier

include("models.jl")
using .Models
export ExtremeLearningMachine, ExtremeLearner, RegularizedExtremeLearner, fit!, predict, 
    predictcounterfactual!, placebotest!

include("metrics.jl")
using .Metrics
export mse, mae, accuracy, precision, recall, F1

include("crossval.jl")
using .CrossValidation
export recode, traintest, validate, crossvalidate, bestsize

include("estimators.jl")
using .Estimators
export EventStudy, GComputation, DoublyRobust, estimatecausaleffect!, summarize

include("metalearners.jl")
using .Metalearners
export Metalearner, SLearner, TLearner, XLearner, estimatecausaleffect!, summarize

const summarise = summarize

end
