using Test
using CausalELM
using DataFrames
using LinearAlgebra

import Random

include("../src/models.jl")

Random.seed!(17)

# Make sure to avoid singular exceptions
x = randn(50, 7)
Q, R = qr(x)
x, w = Q[:, 1:4], Q[:, 5:7]
t, y = Float64.([rand()<0.8 for i in 1:50]), Q[:, 36]

x₀, y₀, x₁, y₁ = rand(100, 5), rand(100), rand(10, 5), rand(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)

# ITS with a DataFrame
x₀_df, y₀_df = DataFrame(x1=x[:, 1], x2=x[:, 2], x3=x[:, 3]), DataFrame(y=rand(50))
x₁_df, y₁_df = DataFrame(x1=Q[:, 50], x2=Q[:, 49], x3=Q[:, 48]), DataFrame(y=rand(50))
its_df = InterruptedTimeSeries(x₀_df, y₀_df, x₁_df, y₁_df)

# No autoregressive term
its_no_ar = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its_no_ar)

# Testing without regularization
its_noreg = InterruptedTimeSeries(x₀, y₀, x₁, y₁, regularized=false)
estimate_causal_effect!(its_noreg)

g_computer = GComputation(x, t, y, temporal=false)
estimate_causal_effect!(g_computer)

# Testing with a binary outcome
g_computer_binary_out = GComputation(x, y, t, temporal=false)
estimate_causal_effect!(g_computer_binary_out)

# G-computation with a DataFrame
x_df = DataFrame(x1=randn(50), x2=randn(50), x3=randn(50), x4=randn(50))
t_df, y_df = DataFrame(t=rand(0:1, 50)), DataFrame(y=rand(50))
g_computer_df = GComputation(x_df, t_df, y_df)

gcomputer_att = GComputation(x, t, y, quantity_of_interest="ATT", temporal=false)
estimate_causal_effect!(gcomputer_att)

gcomputer_noreg = GComputation(x, t, y, regularized=false)
estimate_causal_effect!(gcomputer_noreg)

# Make sure the data isn't shuffled
g_computer_ts = GComputation(float.(hcat([1:10;], 11:20)), 
    Float64.([rand()<0.4 for i in 1:10]), rand(10))

dm = DoubleMachineLearning(x, t, y)
estimate_causal_effect!(dm)

# Testing with a binary outcome
dm_binary_out = DoubleMachineLearning(x, y, t)
estimate_causal_effect!(dm_binary_out)

# With dataframes instead of arrays
dm_df = DoubleMachineLearning(x_df, t_df, y_df)

# No regularization
dm_noreg = DoubleMachineLearning(x, t, y, regularized=false)
estimate_causal_effect!(dm_noreg)

# Specifying W
dm_w = DoubleMachineLearning(x, t, y, W=w)
estimate_causal_effect!(dm_w)

# Calling estimate_effect!
dm_estimate_effect = DoubleMachineLearning(x, t, y)
CausalELM.estimate_effect!(dm_estimate_effect)

# Generating folds
x_fold, t_fold, w_fold, y_fold = CausalELM.make_folds(dm)

# Test predicting residuals
x_train, x_test = x[1:15, :], x[26:end, :]
t_train, t_test = t[1:15], t[26:50]
y_train, y_test = y[1:15], y[26:end]
residual_predictor = DoubleMachineLearning(x, t, y)
residual_predictor.num_neurons = 10
residuals = CausalELM.predict_residuals(residual_predictor, x_train, x_test, y_train, 
                                        y_test, t_train, t_test, x_train, x_test)

# Estimating the CATE
cate_estimator = DoubleMachineLearning(x, t, y, regularized=false)
cate_estimator.num_neurons = 5
cate_predictors = CausalELM.estimate_effect!(cate_estimator, true)

@testset "Interrupted Time Series Estimation" begin
    @testset "Interrupted Time Series Structure" begin
        @test its.X₀ !== Nothing
        @test its.Y₀ !== Nothing
        @test its.X₁ !== Nothing
        @test its.Y₁ !== Nothing

        # No autocorrelation term
        @test its_no_ar.X₀ !== Nothing
        @test its_no_ar.Y₀ !== Nothing
        @test its_no_ar.X₁ !== Nothing
        @test its_no_ar.Y₁ !== Nothing

        # When initializing with a DataFrame
        @test its_df.X₀ !== Nothing
        @test its_df.Y₀ !== Nothing
        @test its_df.X₁ !== Nothing
        @test its_df.Y₁ !== Nothing
    end

    @testset "Interrupted Time Series Estimation" begin
        @test isa(its.Δ, Array)

        # Without autocorrelation
        @test isa(its_no_ar.Δ, Array)

        # Without regularization
        @test isa(its_noreg.Δ, Array)
    end
end

@testset "G-Computation" begin
    @testset "G-Computation Structure" begin
        @test g_computer.X !== Nothing
        @test g_computer.T !== Nothing
        @test g_computer.Y !== Nothing

        # Make sure temporal data isn't shuffled
        @test g_computer_ts.X[1, 1] === 1.0
        @test g_computer_ts.X[2, 1] === 2.0
        @test g_computer_ts.X[9, 2] === 19.0
        @test g_computer_ts.X[10, 2] === 20.0

        # G-computation Initialized with a DataFrame
        @test g_computer_df.X !== Nothing
        @test g_computer_df.T !== Nothing
        @test g_computer_df.Y !== Nothing
    end

    @testset "G-Computation Estimation" begin
        @test isa(g_computer.causal_effect, Float64)

        # Estimation without regularization
        @test isa(gcomputer_noreg.causal_effect, Float64)

        @test isa(g_computer_binary_out.causal_effect, Float64)

        # Check that the estimats for ATE and ATT are different
        @test g_computer.causal_effect !== gcomputer_att.causal_effect
    end
end

@testset "Double Machine Learning" begin
    @testset "Double Machine Learning Structure" begin
        @test dm.X !== Nothing
        @test dm.T !== Nothing
        @test dm.Y !== Nothing

        # No regularization
        @test dm_noreg.X !== Nothing
        @test dm_noreg.T !== Nothing
        @test dm_noreg.Y !== Nothing

        # Intialized with dataframes
        @test dm_df.X !== Nothing
        @test dm_df.T !== Nothing
        @test dm_df.Y !== Nothing
    end

    @testset "Double Machine Learning Estimation Helpers" begin
        @test dm_estimate_effect.causal_effect isa Float64
        @test size(x_fold[1], 2) == size(dm.X, 2)
        @test size(w_fold[1], 2) == size(dm.W, 2)
        @test y_fold isa Vector{Vector{Float64}}
        @test t_fold isa Vector{Vector{Float64}}
        @test length(t_fold) == dm.folds
        @test residuals[1] isa Vector
        @test residuals[2] isa Vector
    end

    @testset "CATE Estimation" begin
        @test cate_predictors isa Vector
        @test length(cate_predictors) == length(cate_estimator.Y)
        @test eltype(cate_predictors) == Float64
    end

    @testset "Double Machine Learning Post-estimation Structure" begin
        @test dm.causal_effect isa Float64
        @test dm_binary_out.causal_effect isa Float64
        @test dm_noreg.causal_effect isa Float64
        dm_w.causal_effect isa Float64
    end
end

@testset "Summarization and Inference" begin
    @testset "Quanities of Interest Errors" begin
        @test_throws ArgumentError GComputation(x, y, t, quantity_of_interest="abc")
    end

    @testset "Task Errors" begin
        @test_throws ArgumentError GComputation(x, y, t, task="abc")
    end

    @testset "Moving Averages" begin
        @test CausalELM.moving_average(Float64[]) isa Array{Float64}
        @test CausalELM.moving_average([1.0]) == [1.0]
        @test CausalELM.moving_average([1.0, 2.0, 3.0]) == [1.0, 1.5, 2.0]
    end
end
