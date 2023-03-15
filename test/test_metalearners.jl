using CausalELM.Metalearners: SLearner, TLearner, XLearner, estimatecausaleffect!, stage1!, 
    stage2!, summarize
using CausalELM.Models: ExtremeLearningMachine
using Test

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
slearner1, slearner2 = SLearner(x, y, t), SLearner(x, y, t, regularized=true)
estimatecausaleffect!(slearner1); estimatecausaleffect!(slearner2)
summary1 = summarize(slearner1)
summary2 = summarize(slearner2)

tlearner1, tlearner2 = TLearner(x, y, t), TLearner(x, y, t, regularized=true)
estimatecausaleffect!(tlearner1); estimatecausaleffect!(tlearner2)
summary1t = summarize(tlearner1)
summary2t = summarize(tlearner2)

xlearner1 = XLearner(x, y, t)
xlearner1.num_neurons = 5
stage1!(xlearner1); stage2!(xlearner1)

xlearner2 = XLearner(x, y, t, regularized=true)
xlearner2.num_neurons = 5
stage1!(xlearner2); stage2!(xlearner2)

xlearner3 = XLearner(x, y, t)
estimatecausaleffect!(xlearner3)
summary3 = summarize(xlearner3)

xlearner4 = XLearner(x, y, t, regularized=true)
estimatecausaleffect!(xlearner4)
summary4 = summarize(xlearner3)

@testset "S-Learner Structure" begin
    @test slearner1.X !== Nothing
    @test slearner1.Y !== Nothing
    @test slearner1.T !== Nothing
    @test slearner2.X !== Nothing
    @test slearner2.Y !== Nothing
    @test slearner2.T !== Nothing
end

@testset "S-Learner Estimation" begin
    @test isa(slearner1.β, Array)
    @test isa(slearner1.causal_effect, Array{Float64})
    @test isa(slearner2.β, Array)
    @test isa(slearner2.causal_effect, Array{Float64})
end

@testset "T-Learner Structure" begin
    @test tlearner1.X !== Nothing
    @test tlearner1.Y !== Nothing
    @test tlearner1.T !== Nothing
    @test tlearner2.X !== Nothing
    @test tlearner2.Y !== Nothing
    @test tlearner2.T !== Nothing
end

@testset "T-Learner Estimation" begin
    @test typeof(tlearner1.μ₀) <: ExtremeLearningMachine
    @test typeof(tlearner1.μ₁) <: ExtremeLearningMachine
    @test isa(tlearner1.causal_effect, Array{Float64})
    @test typeof(tlearner2.μ₀) <: ExtremeLearningMachine
    @test typeof(tlearner2.μ₁) <: ExtremeLearningMachine
    @test isa(tlearner2.causal_effect, Array{Float64})
end

@testset "First Stage X-Learner" begin
    @test typeof(xlearner1.g) <: ExtremeLearningMachine
    @test typeof(xlearner1.μ₀) <: ExtremeLearningMachine
    @test typeof(xlearner1.μ₁) <: ExtremeLearningMachine
    @test xlearner1.gᵢ isa Array{Float64}
    @test xlearner1.μ₀.__fit === true
    @test xlearner1.μ₁.__fit === true
    @test typeof(xlearner2.g) <: ExtremeLearningMachine
    @test typeof(xlearner2.μ₀) <: ExtremeLearningMachine
    @test typeof(xlearner2.μ₁) <: ExtremeLearningMachine
    @test xlearner2.gᵢ isa Array{Float64}
    @test xlearner2.μ₀.__fit === true
    @test xlearner2.μ₁.__fit === true
end

@testset "Second Stage X-Learner" begin
    @test typeof(xlearner1.μχ₀) <: ExtremeLearningMachine
    @test typeof(xlearner1.μχ₁) <: ExtremeLearningMachine
    @test typeof(xlearner2.μχ₀) <: ExtremeLearningMachine
    @test typeof(xlearner2.μχ₁) <: ExtremeLearningMachine
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
    @test typeof(xlearner3.g) <: ExtremeLearningMachine
    @test typeof(xlearner3.μ₀) <: ExtremeLearningMachine
    @test typeof(xlearner3.μ₁) <: ExtremeLearningMachine
    @test xlearner3.gᵢ isa Array{Float64}
    @test xlearner3.causal_effect isa Array{Float64}
    @test typeof(xlearner3.μχ₀) <: ExtremeLearningMachine
    @test typeof(xlearner3.μχ₁) <: ExtremeLearningMachine
end

@testset "Metalearners Summary" begin
    for (k, v) in summary1
        @test v isa String
    end

    for (k, v) in summary2
        @test v isa String
    end

    for (k, v) in summary1t
        @test v isa String
    end

    for (k, v) in summary2t
        @test v isa String
    end

    for (k, v) in summary3
        @test v isa String
    end

    for (k, v) in summary4
        @test v isa String
    end
end

@testset "Task Errors" begin
    @test_throws AssertionError SLearner(x, y, t, task="abc")
    @test_throws AssertionError TLearner(x, y, t, task="def")
    @test_throws AssertionError XLearner(x, y, t, task="xyz")
end
