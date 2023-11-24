using Test
import CausalELM

@testset "Moments" begin
    @test CausalELM.mean([1, 2, 3]) == 2
    @test CausalELM.var([1, 2, 3]) == 1
end

@testset "Add and Subtract Consecutive Elements" begin
    @test CausalELM.consecutive([1, 2, 3, 4, 5]) == [1, 1, 1, 1]
end

@testset "Outcome Variable Types" begin
    @test CausalELM.variable_type([1, 0, 0, 1]) isa CausalELM.Discrete
    @test CausalELM.variable_type([1, 2, 3]) isa CausalELM.Discrete
    @test CausalELM.variable_type([1.1, 2.2, 3.3]) isa CausalELM.Continuous
end

include("test_activation.jl")
include("test_models.jl")
include("test_metrics.jl")
include("test_crossval.jl")
include("test_estimators.jl")
include("test_metalearners.jl")
include("test_inference.jl")
include("test_model_validation.jl")
