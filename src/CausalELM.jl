module CausalELM

export binarystep, σ, tanh, relu, leakyrelu, swish, softmax, softplus, gelu, gaussian,
    hardtanh, elish, fourier, Elm, fit, predict

include("activation.jl")
include("models.jl")

end
