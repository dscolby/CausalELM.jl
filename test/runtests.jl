using Test
using Aqua
using Documenter
using CausalELM

include("test_activation.jl")
include("test_models.jl")
include("test_metrics.jl")
include("test_estimators.jl")
include("test_metalearners.jl")
include("test_inference.jl")
include("test_model_validation.jl")
include("test_utilities.jl")

Aqua.test_all(CausalELM)

DocMeta.setdocmeta!(CausalELM, :DocTestSetup, :(using CausalELM); recursive=true)
doctest(CausalELM)
