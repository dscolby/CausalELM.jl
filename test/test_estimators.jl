using Test
using CausalELM

include("../src/models.jl")

x₀, y₀, x₁, y₁ = Float64.(rand(1:100, 100, 5)), rand(100), rand(10, 5), rand(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)

# No autoregressive term
its_no_ar = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its_no_ar)

x, y, t = rand(100, 5), vec(rand(1:100, 100, 1)), Float64.([rand()<0.4 for i in 1:100])
g_computer = GComputation(x, y, t, temporal=false)
estimate_causal_effect!(g_computer)

gcomputer_att = GComputation(x, y, t, quantity_of_interest="ATT", temporal=false)
estimate_causal_effect!(gcomputer_att)

# Mak sure the data isn't shuffled
g_computer_ts = GComputation(float.(hcat([1:10;], 11:20)), rand(10), 
    Float64.([rand()<0.4 for i in 1:10]))

dm = DoubleMachineLearning(x, y, t)
estimate_causal_effect!(dm)

# No regularization
dm_noreg = DoubleMachineLearning(x, y, t, regularized=false)
estimate_causal_effect!(dm_noreg)

# Calling estimate_effect!
dm_estimate_effect = DoubleMachineLearning(x, y, t)
dm_estimate_effect.num_neurons = 5
CausalELM.estimate_effect!(dm_estimate_effect)

# Test predicting residuals
x_train, x_test = x[1:80, :], x[81:end, :]
y_train, y_test = y[1:80], y[81:end]
t_train, t_test = t[1:80], t[81:100]
residual_predictor = DoubleMachineLearning(x, y, t)
residual_predictor.num_neurons = 5
residuals = CausalELM.predict_residuals(residual_predictor, x_train, x_test, y_train, 
    y_test, t_train, t_test)

# Estimating the CATE
cate_estimator = DoubleMachineLearning(x, y, t)
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
    end

    @testset "Interrupted Time Series Estimation" begin
        @test isa(its.β, Array)
        @test isa(its.Ŷ, Array)
        @test isa(its.Δ, Array)

        # Without autocorrelation
        @test isa(its_no_ar.β, Array)
        @test isa(its_no_ar.Ŷ, Array)
        @test isa(its_no_ar.Δ, Array)
    end
end

@testset "G-Computation" begin
    @testset "G-Computation Structure" begin
        @test g_computer.X !== Nothing
        @test g_computer.Y !== Nothing
        @test g_computer.T !== Nothing

        # Make sure temporal data isn't shuffled
        @test g_computer_ts.X[1, 1] === 1.0
        @test g_computer_ts.X[2, 1] === 2.0
        @test g_computer_ts.X[9, 2] === 19.0
        @test g_computer_ts.X[10, 2] === 20.0
    end

    @testset "G-Computation Estimation" begin
        @test isa(g_computer.β, Array)
        @test isa(g_computer.causal_effect, Float64)

        # Check that the estimats for ATE and ATT are different
        @test g_computer.causal_effect !== gcomputer_att.causal_effect
    end
end

@testset "Double Machine Learning" begin
    @testset "Double Machine Learning Structure" begin
        @test dm.X !== Nothing
        @test dm.Y !== Nothing
        @test dm.T !== Nothing

        # No regularization
        @test dm_noreg.X !== Nothing
        @test dm_noreg.Y !== Nothing
        @test dm_noreg.T !== Nothing
    end

    @testset "Double Machine Learning Estimation Helpers" begin
        @test dm_estimate_effect.causal_effect isa Float64
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
        @test dm_noreg.causal_effect isa Float64
    end
end

@testset "Summarization and Inference" begin
    @testset "Quanities of Interest Errors" begin
        @test_throws ArgumentError GComputation(x, y, t, quantity_of_interest="abc")
    end

    @testset "Task Errors" begin
        @test_throws ArgumentError InterruptedTimeSeries(x₀, y₀, x₁, y₁, task="abc")
        @test_throws ArgumentError GComputation(x, y, t, task="abc")
        @test_throws ArgumentError DoubleMachineLearning(x, y, t, task="xyz")
    end

    @testset "Moving Averages" begin
        @test CausalELM.moving_average(Float64[]) isa Array{Float64}
        @test CausalELM.moving_average([1.0]) == [1.0]
        @test CausalELM.moving_average([1.0, 2.0, 3.0]) == [1.0, 1.5, 2.0]
    end
end
