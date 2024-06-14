using CausalELM
using Test
using DataFrames

include("../src/models.jl")

x, t, y = rand(100, 5), Float64.([rand() < 0.4 for i in 1:100]), vec(rand(1:100, 100, 1))
slearner1, slearner2 = SLearner(x, t, y), SLearner(x, t, y; regularized=true)
estimate_causal_effect!(slearner1);
estimate_causal_effect!(slearner2);

# S-learner with a binary outcome
s_learner_binary = SLearner(x, y, t)
estimate_causal_effect!(s_learner_binary)

# S-learner initialized with DataFrames
x_df = DataFrame(; x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(; t=rand(0:1, 100)), DataFrame(; y=rand(100))

s_learner_df = SLearner(x_df, t_df, y_df)

tlearner1, tlearner2 = TLearner(x, t, y), TLearner(x, t, y; regularized=false)
estimate_causal_effect!(tlearner1);
estimate_causal_effect!(tlearner2);

# T-learner initialized with DataFrames
t_learner_df = TLearner(x_df, t_df, y_df)

# Testing with a binary outcome
t_learner_binary = TLearner(x, t, Float64.([rand() < 0.8 for i in 1:100]))
estimate_causal_effect!(t_learner_binary)

xlearner1 = XLearner(x, t, y)
xlearner1.num_neurons = 5
CausalELM.stage1!(xlearner1)
stage21 = CausalELM.stage2!(xlearner1)

xlearner2 = XLearner(x, t, y; regularized=true)
xlearner2.num_neurons = 5
CausalELM.stage1!(xlearner2);
CausalELM.stage2!(xlearner2);
stage22 = CausalELM.stage2!(xlearner1)

xlearner3 = XLearner(x, t, y)
estimate_causal_effect!(xlearner3)

xlearner4 = XLearner(x, t, y; regularized=true)
estimate_causal_effect!(xlearner4)

# Testing initialization with DataFrames
x_learner_df = XLearner(x_df, t_df, y_df)

# Testing with binary outcome
x_learner_binary = XLearner(x, t, Float64.([rand() < 0.4 for i in 1:100]))
estimate_causal_effect!(x_learner_binary)

rlearner = RLearner(x, t, y)
estimate_causal_effect!(rlearner)

# Testing with a W arguments
r_learner_w = RLearner(x, t, y; W=rand(100, 4))
estimate_causal_effect!(r_learner_w)

# Testing initialization with DataFrames
r_learner_df = RLearner(x_df, t_df, y_df)

# Doubly Robust Estimation
dr_learner = DoublyRobustLearner(x, t, y; W=rand(100, 4))
X_T, Y = generate_folds(
    reduce(hcat, (dr_learner.X, dr_learner.T, dr_learner.W)), dr_learner.Y, 2
)
X = [fl[:, 1:size(dr_learner.X, 2)] for fl in X_T]
T = [fl[:, size(dr_learner.X, 2) + 1] for fl in X_T]
W = [fl[:, (size(dr_learner.W, 2) + 2):end] for fl in X_T]
τ̂ = CausalELM.estimate_effect!(dr_learner, X, T, Y, reduce(hcat, (W, X)))
estimate_causal_effect!(dr_learner)

# Doubly Robust Estimation with no regularization
dr_no_reg = DoublyRobustLearner(x, t, y; W=rand(100, 4), regularized=false)
estimate_causal_effect!(dr_no_reg)

# Testing Doubly Robust Estimation with a binary outcome
dr_learner_binary = DoublyRobustLearner(x, t, Float64.([rand() < 0.8 for i in 1:100]))
estimate_causal_effect!(dr_learner_binary)

# Doubly robust estimation with DataFrames
dr_learner_df = DoublyRobustLearner(x_df, t_df, y_df)
estimate_causal_effect!(dr_learner_df)

@testset "S-Learners" begin
    @testset "S-Learner Structure" begin
        @test slearner1.g isa GComputation
        @test slearner2.g isa GComputation
        @test s_learner_df.g isa GComputation
    end

    @testset "S-Learner Estimation" begin
        @test isa(slearner1.causal_effect, Array{Float64})
        @test isa(slearner2.causal_effect, Array{Float64})
        @test isa(s_learner_binary.causal_effect, Array{Float64})
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
        @test isa(t_learner_binary.causal_effect, Array{Float64})
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
        @test x_learner_binary.causal_effect isa Array{Float64}
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
        @test r_learner_w.causal_effect isa Vector
        @test length(r_learner_w.causal_effect) == length(y)
        @test eltype(r_learner_w.causal_effect) == Float64
    end
end

@testset "Doubly Robust Learners" begin
    @testset "Doubly Robust Learner Structure" begin
        for field in fieldnames(typeof(dr_learner))
            @test getfield(dr_learner, field) !== Nothing
        end

        for field in fieldnames(typeof(dr_learner_df))
            @test getfield(dr_learner_df, field) !== Nothing
        end
    end

    @testset "Calling estimate_effect!" begin
        @test length(τ̂) === length(dr_learner.Y)
    end

    @testset "Doubly Robust Learner Estimation" begin
        @test dr_learner.causal_effect isa Vector
        @test length(dr_learner.causal_effect) === length(y)
        @test eltype(dr_learner.causal_effect) == Float64
        @test dr_learner_df.causal_effect isa Vector
        @test length(dr_learner_df.causal_effect) === length(y)
        @test eltype(dr_learner_df.causal_effect) == Float64
        @test dr_learner_binary.causal_effect isa Vector
        @test length(dr_learner_binary.causal_effect) === length(y)
        @test eltype(dr_learner_binary.causal_effect) == Float64
        @test dr_no_reg.causal_effect isa Vector
        @test length(dr_no_reg.causal_effect) === length(y)
        @test eltype(dr_no_reg.causal_effect) == Float64
    end
end

@testset "Task Errors" begin
    @test_throws ArgumentError SLearner(x, t, y, task="abc")
    @test_throws ArgumentError TLearner(x, t, y, task="def")
    @test_throws ArgumentError XLearner(x, t, y, task="xyz")
end
