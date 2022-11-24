module CausalELM

export binarystep, σ, tanh, relu, leakyrelu, swish, softmax, softplus, gelu, gaussian,
    hardtanh, elish, fourier, ExtremeLearner, RegularixedExtremeLearner, fit!, predict,
    predictcounterfactual!, placebotest!

include("activation.jl")
include("models.jl")

end
