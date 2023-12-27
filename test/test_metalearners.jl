using Test
using CausalELM

include("../src/models.jl")

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
slearner1, slearner2 = SLearner(x, y, t), SLearner(x, y, t, regularized=true)
estimate_causal_effect!(slearner1); estimate_causal_effect!(slearner2)

tlearner1, tlearner2 = TLearner(x, y, t), TLearner(x, y, t, regularized=true)
estimate_causal_effect!(tlearner1); estimate_causal_effect!(tlearner2)

xlearner1 = XLearner(x, y, t)
xlearner1.num_neurons = 5
CausalELM.stage1!(xlearner1)
stage21 = CausalELM.stage2!(xlearner1)

xlearner2 = XLearner(x, y, t, regularized=true)
xlearner2.num_neurons = 5
CausalELM.stage1!(xlearner2); CausalELM.stage2!(xlearner2)
stage22 = CausalELM.stage2!(xlearner1)

xlearner3 = XLearner(x, y, t)
estimate_causal_effect!(xlearner3)

xlearner4 = XLearner(x, y, t, regularized=true)
estimate_causal_effect!(xlearner4)

rlearner = RLearner(x, y, t)
estimate_causal_effect!(rlearner)

@testset "S-Learners" begin
    @testset "S-Learner Structure" begin
        @test slearner1.X !== Nothing
        @test slearner1.Y !== Nothing
        @test slearner1.T !== Nothing
        @test slearner2.X !== Nothing
        @test slearner2.Y !== Nothing
        @test slearner2.T !== Nothing
    end

    @testset "S-Learner Estimation" begin
        @test isa(slearner1.causal_effect, Array{Float64})
        @test isa(slearner2.causal_effect, Array{Float64})
    end
end

@testset "T-Learners" begin
    @testset "T-Learner Structure" begin
        @test tlearner1.X !== Nothing
        @test tlearner1.Y !== Nothing
        @test tlearner1.T !== Nothing
        @test tlearner2.X !== Nothing
        @test tlearner2.Y !== Nothing
        @test tlearner2.T !== Nothing
    end

    @testset "T-Learner Estimation" begin
        @test isa(tlearner1.causal_effect, Array{Float64})
        @test isa(tlearner2.causal_effect, Array{Float64})
    end
end

@testset "X-Learners" begin
    @testset "First Stage X-Learner" begin
        @test typeof(xlearner1.μ₀) <: CausalELM.ExtremeLearningMachine
        @test typeof(xlearner1.μ₁) <: CausalELM.ExtremeLearningMachine
        @test xlearner1.ps isa Array{Float64}
        @test xlearner1.μ₀.__fit === true
        @test xlearner1.μ₁.__fit === true
        @test typeof(xlearner2.μ₀) <: CausalELM.ExtremeLearningMachine
        @test typeof(xlearner2.μ₁) <: CausalELM.ExtremeLearningMachine
        @test xlearner2.ps isa Array{Float64}
        @test xlearner2.μ₀.__fit === true
        @test xlearner2.μ₁.__fit === true
    end

    @testset "Second Stage X-Learner" begin
        @test length(stage21) == 2
        @test eltype(stage21) <: CausalELM.ExtremeLearningMachine
        @test length(stage22) == 2
        @test eltype(stage22) <: CausalELM.ExtremeLearningMachine
    end

    @testset "X-Learner Structure" begin
        @test xlearner3.X !== Nothing
        @test xlearner3.Y !== Nothing
        @test xlearner3.T !== Nothing
        @test xlearner4.X !== Nothing
        @test xlearner4.Y !== Nothing
        @test xlearner4.T !== Nothing
    end

    @testset "X-Learner Estimation" begin
        @test typeof(xlearner3.μ₀) <: CausalELM.ExtremeLearningMachine
        @test typeof(xlearner3.μ₁) <: CausalELM.ExtremeLearningMachine
        @test xlearner3.ps isa Array{Float64}
        @test xlearner3.causal_effect isa Array{Float64}
    end
end

@testset "R-learning" begin
    @testset "R-learner Structure" begin
        @test rlearner.dml !== Nothing
    end

    @testset "R-learner estimation" begin
        @test rlearner.causal_effect isa Vector
        @test length(rlearner.causal_effect) == length(y)
        @test eltype(rlearner.causal_effect) == Float64
    end
end

@testset "Task Errors" begin
    @test_throws ArgumentError SLearner(x, y, t, task="abc")
    @test_throws ArgumentError TLearner(x, y, t, task="def")
    @test_throws ArgumentError XLearner(x, y, t, task="xyz")
end
