"""
Macros, functions, and structs for applying Extreme Learning Machines to causal inference
tasks where the counterfactual is unavailable or biased and must be predicted. Supports 
causal inference via interrupted time series designs, parametric G-computation, double 
machine learning, and S-learning, T-learning, X-learning, R-learning, and doubly robust 
estimation. Additionally, these tasks can be performed with or without L2 penalization and
will automatically choose the best number of neurons and L2 penalty. 

For more details on Extreme Learning Machines see:
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory 
    and applications." Neurocomputing 70, no. 1-3 (2006): 489-501.
"""
module CausalELM

export validate
export gelu, gaussian
export hard_tanh, elish, fourier
export binary_step, σ, tanh, relu
export leaky_relu, swish, softmax, softplus
export mse, mae, accuracy, precision, recall, F1
export estimate_causal_effect!, summarize, summarise
export InterruptedTimeSeries, GComputation, DoubleMachineLearning
export SLearner, TLearner, XLearner, RLearner, DoublyRobustLearner

include("activation.jl")
include("utilities.jl")
include("models.jl")
include("metrics.jl")
include("estimators.jl")
include("metalearners.jl")
include("inference.jl")
include("model_validation.jl")

# So that it works with British spelling
const summarise = summarize

end
