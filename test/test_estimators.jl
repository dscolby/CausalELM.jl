using Test
using CausalELM
using DataFrames

include("../src/models.jl")

x₀, y₀, x₁, y₁ = Float64.(rand(1:200, 100, 5)), rand(100), rand(10, 5), rand(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)

# ITS with a DataFrame
x₀_df, y₀_df = DataFrame(; x1=rand(100), x2=rand(100), x3=rand(100)),
DataFrame(; y=rand(100))
x₁_df, y₁_df = DataFrame(; x1=rand(100), x2=rand(100), x3=rand(100)),
DataFrame(; y=rand(100))
its_df = InterruptedTimeSeries(x₀_df, y₀_df, x₁_df, y₁_df)

# No autoregressive term
its_no_ar = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its_no_ar)

x, t, y = rand(100, 5), rand(0:1, 100), vec(rand(1:100, 100, 1))
g_computer = GComputation(x, t, y; temporal=false)
estimate_causal_effect!(g_computer)

# Testing with a binary outcome
g_computer_binary_out = GComputation(x, y, t; temporal=false)
estimate_causal_effect!(g_computer_binary_out)

# G-computation with a DataFrame
x_df = DataFrame(; x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100))
t_df, y_df = DataFrame(; t=rand(0:1, 100)), DataFrame(; y=rand(100))
g_computer_df = GComputation(x_df, t_df, y_df)
gcomputer_att = GComputation(x, t, y; quantity_of_interest="ATT", temporal=false)
estimate_causal_effect!(gcomputer_att)

# Make sure the data isn't shuffled
g_computer_ts = GComputation(
    float.(hcat([1:10;], 11:20)), Float64.([rand() < 0.4 for i in 1:10]), rand(10)
)

big_x, big_t, big_y = rand(10000, 8), rand(0:1, 10000), vec(rand(1:100, 10000, 1))
dm = DoubleMachineLearning(big_x, big_t, big_y)
estimate_causal_effect!(dm)

# Testing with a binary outcome
dm_binary_out = DoubleMachineLearning(x, y, t)
estimate_causal_effect!(dm_binary_out)

# With dataframes instead of arrays
dm_df = DoubleMachineLearning(x_df, t_df, y_df)

# Test predicting residuals
x_train, x_test = x[1:80, :], x[81:end, :]
t_train, t_test = float(t[1:80]), float(t[81:end])
y_train, y_test = float(y[1:80]), float(y[81:end])
residual_predictor = DoubleMachineLearning(x, t, y, num_neurons=5)
residuals = CausalELM.predict_residuals(
    residual_predictor, x_train, x_test, y_train, y_test, t_train, t_test
)

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
        @test isa(its.causal_effect, Array)

        # Without autocorrelation
        @test isa(its_no_ar.causal_effect, Array)
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

        # Intialized with dataframes
        @test dm_df.X !== Nothing
        @test dm_df.T !== Nothing
        @test dm_df.Y !== Nothing
    end

    @testset "Generating Residuals" begin
        @test residuals[1] isa Vector
        @test residuals[2] isa Vector
    end

    @testset "Double Machine Learning Post-estimation Structure" begin
        @test dm.causal_effect isa Float64
    end
end

@testset "Miscellaneous Tests" begin
    @testset "Quanities of Interest Errors" begin
        @test_throws ArgumentError GComputation(x, y, t, quantity_of_interest="abc")
    end

    @testset "Moving Averages" begin
        @test CausalELM.moving_average(Float64[]) isa Array{Float64}
        @test CausalELM.moving_average([1.0]) == [1.0]
        @test CausalELM.moving_average([1.0, 2.0, 3.0]) == [1.0, 1.5, 2.0]
    end
end
