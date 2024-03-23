using Test
using CausalELM
using DataFrames

include("../src/models.jl")

x, t, y = rand(100, 5), [rand()<0.4 for i in 1:100], rand(1:100, 100, 1)
slearner1, slearner2 = SLearner(x, t, y), SLearner(x, t, y, regularized=true)
estimate_causal_effect!(slearner1); estimate_causal_effect!(slearner2)

# S-learner initialized with DataFrames
x_df = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(t=rand(0:1, 100)), DataFrame(y=rand(100))
s_learner_df = SLearner(x_df, t_df, y_df)

tlearner1, tlearner2 = TLearner(x, t, y), TLearner(x, t, y, regularized=true)
estimate_causal_effect!(tlearner1); estimate_causal_effect!(tlearner2)

# T-learner initialized with DataFrames
t_learner_df = TLearner(x_df, t_df, y_df)

xlearner1 = XLearner(x, t, y)
xlearner1.num_neurons = 5
CausalELM.stage1!(xlearner1)
stage21 = CausalELM.stage2!(xlearner1)

xlearner2 = XLearner(x, t, y, regularized=true)
xlearner2.num_neurons = 5
CausalELM.stage1!(xlearner2); CausalELM.stage2!(xlearner2)
stage22 = CausalELM.stage2!(xlearner1)

xlearner3 = XLearner(x, t, y)
estimate_causal_effect!(xlearner3)

xlearner4 = XLearner(x, t, y, regularized=true)
estimate_causal_effect!(xlearner4)

# Testing initialization with DataFrames
x_learner_df = XLearner(x_df, y_df, y_df)

rlearner = RLearner(x, t, y)
estimate_causal_effect!(rlearner)

# R-learner with categorical treatment
rlearner_t_cat = RLearner(x, rand(1:4, 100), y)
estimate_causal_effect!(rlearner_t_cat)

# R-learner with categorical outcome
rlearner_y_cat = RLearner(x, t, rand(1:4, 100))
estimate_causal_effect!(rlearner_y_cat)

# Testing initialization with DataFrames
r_learner_df = RLearner(x_df, t_df, y_df)

@testset "S-Learners" begin
    @testset "S-Learner Structure" begin
        @test slearner1.g isa GComputation
        @test slearner2.g isa GComputation
        @test s_learner_df.g isa GComputation
    end

    @testset "S-Learner Estimation" begin
        @test isa(slearner1.causal_effect, Array{Float64})
        @test isa(slearner2.causal_effect, Array{Float64})
    end
end

@testset "T-Learners" begin
    @testset "T-Learner Structure" begin
        @test tlearner1.X !== Nothing
        @test tlearner1.T !== Nothing
        @test tlearner1.Y !== Nothing
        @test tlearner2.X !== Nothing
        @test tlearner2.T !== Nothing
        @test tlearner2.Y !== Nothing
        @test t_learner_df.X !== Nothing
        @test t_learner_df.T !== Nothing
        @test t_learner_df.Y !== Nothing
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
        @test xlearner3.T !== Nothing
        @test xlearner3.Y !== Nothing
        @test xlearner4.X !== Nothing
        @test xlearner4.T !== Nothing
        @test xlearner4.Y !== Nothing
        @test x_learner_df.X !== Nothing
        @test x_learner_df.T !== Nothing
        @test x_learner_df.Y !== Nothing
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
        @test r_learner_df.dml !== Nothing
    end

    @testset "R-learner estimation" begin
        @test rlearner.causal_effect isa Vector
        @test length(rlearner.causal_effect) == length(y)
        @test eltype(rlearner.causal_effect) == Float64
        @test length(rlearner_t_cat.causal_effect) == length(y)
        @test length(rlearner_y_cat.causal_effect) == length(y)
    end
end

@testset "Task Errors" begin
    @test_throws ArgumentError SLearner(x, t, y, task="abc")
    @test_throws ArgumentError TLearner(x, t, y, task="def")
    @test_throws ArgumentError XLearner(x, t, y, task="xyz")
end
