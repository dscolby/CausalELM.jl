module CausalELM

export binarystep, σ, tanh, relu, leakyrelu, swish, softmax, softplus, gelu, gaussian,
    hardtanh, elish, Elm, fit, predict

include("activation.jl")
include("models.jl")

end
