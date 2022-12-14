"""
Macros, functions, and structs for applying Extreme Learning Machines to causal inference
tasks where the counterfactual is unavailable or biased and must be predicted. Provides 
macros for event study designs, IPW, parametric G-computation, and doubly robust machine
learning. Additionally, these tasks can be performed with or without L2 penalization and
will automatically choose the best number of neurons and L2 penalty. For more details on
Extreme Learning Machines see:
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.
"""
module CausalELM

export binarystep, σ, tanh, relu, leakyrelu, swish, softmax, softplus, gelu, gaussian,
    hardtanh, elish, fourier, ExtremeLearningMachine, ExtremeLearner, 
    RegularizedExtremeLearner, fit!, predict, predictcounterfactual!, placebotest!, mse, 
    mae, accuracy, precision, recall, F1, recode, traintest, validate, crossvalidate, 
    bestsize, EventStudy, estimatetreatmenteffect!

include("activation.jl")
include("models.jl")
include("metrics.jl")
include("crossval.jl")
include("estimators.jl")

end
