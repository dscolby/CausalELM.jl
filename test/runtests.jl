using CausalELM
using Test

@testset "Average" begin
    @test mean([1, 2, 3]) == 2
end

include("test_activation.jl")
include("test_models.jl")
include("test_metrics.jl")
include("test_crossval.jl")
include("test_estimators.jl")
include("test_metalearners.jl")
include("test_inference.jl")
include("test_assumptions.jl")
