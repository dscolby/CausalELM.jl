module CausalELM

export binarystep, σ, tanh, relu, leakyrelu, swish, softmax, softplus, gelu, gaussian,
    hardtanh, elish, fourier, ExtremeLearner, RegularizedExtremeLearner, fit!, predict,
    predictcounterfactual!, placebotest!, mse, mae

include("activation.jl")
include("models.jl")
include("metrics.jl")

end
