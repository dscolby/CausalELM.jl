using CausalELM.Estimators: InterruptedTimeSeries, GComputation, DoubleMachineLearning, 
    estimate_causal_effect!, mean, crossfitting_sets, first_stage!, ate!, 
    predict_propensity_score, predict_control_outcomes, predict_treatment_outcomes, 
    moving_average
using CausalELM.Models: ExtremeLearningMachine
using Test

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

dm = DoubleMachineLearning(x, x, y, t)
dm.num_neurons = 5
ps_mod, control_mod = first_stage!(dm, x₀, x, t, y₀)
treat_mod = ate!(dm, x₁, y₁)
dm.num_neurons = 0
estimate_causal_effect!(dm)
x_folds, xₚ_folds, y_folds, t_folds = crossfitting_sets(dm)

# No regularization
dm_noreg = DoubleMachineLearning(x, x, y, t, regularized=false)
estimate_causal_effect!(dm_noreg)

# Estimating the ATT instead of the ATE
dm_att = DoubleMachineLearning(x, x, y, t, quantity_of_interest="ATT")
estimate_causal_effect!(dm_att)

# Estimating the ATT without regularization
dm_att_noreg = DoubleMachineLearning(x, x, y, t, regularized=false, quantity_of_interest="ATT")
estimate_causal_effect!(dm_att_noreg)

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

        @test g_computer.risk_ratio isa Real
        @test g_computer.fit = true
    end
end

@testset "Double Machine Learning" begin
    @testset "Doubly Machine Learning Structure" begin
        @test dm.X !== Nothing
        @test dm.Y !== Nothing
        @test dm.T !== Nothing

        # No regularization
        @test dm_noreg.X !== Nothing
        @test dm_noreg.Y !== Nothing
        @test dm_noreg.T !== Nothing

        # Using the ATT
        @test dm_att.X !== Nothing
        @test dm_att.Y !== Nothing
        @test dm_att.T !== Nothing

        # Using the ATT without regularization
        @test dm_att_noreg.X !== Nothing
        @test dm_att_noreg.Y !== Nothing
        @test dm_att_noreg.T !== Nothing
    end

    @testset "Double Machine Learning First Stage" begin
        @test ps_mod isa ExtremeLearningMachine
        @test control_mod isa ExtremeLearningMachine
    end

    @testset "Double Machine Learning Second Stage" begin
        @test treat_mod isa ExtremeLearningMachine
    end

    @testset "Double Machine Learning Predictions" begin
        @test predict_propensity_score(ps_mod, x₀) isa Array{Float64}
        @test predict_control_outcomes(control_mod, x₀) isa Array{Float64}
        @test predict_treatment_outcomes(treat_mod, x₀) isa Array{Float64}
    end

    @testset "X and Xₚ Different Size Error" begin
        @test_throws ArgumentError DoubleMachineLearning(x, rand(50, 5), y, t)
    end

    @testset "Doubly Robust Post-estimation Structure" begin
        @test dm.ps isa Array{Float64}
        @test dm.μ₀ isa Array{Float64}
        @test dm.μ₁ isa Array{Float64}
        @test dm.causal_effect isa Float64

        # No regularization
        @test dm_noreg.ps isa Array{Float64}
        @test dm_noreg.μ₀ isa Array{Float64}
        @test dm_noreg.μ₁ isa Array{Float64}
        @test dm_noreg.causal_effect isa Float64

        # Using the ATT
        @test dm_att.ps isa Array{Float64}
        @test dm_att.μ₀ isa Array{Float64}
        @test dm_att.causal_effect isa Float64

        # Using the ATT with no regularization
        @test dm_att_noreg.ps isa Array{Float64}
        @test dm_att_noreg.μ₀ isa Array{Float64}
        @test dm_att_noreg.causal_effect isa Float64

        # Estimating the risk ratio
        @test dm_att_noreg.risk_ratio isa Real
        @test dm.risk_ratio isa Real

        # Testing the fit boolean
        @test dm_att_noreg.fit = true
        @test dm_att.fit = true
        @test dm_noreg.fit = true
    end

    @testset "Generating Folds for Cross Fitting" begin
        @test size(x_folds, 1) === 5
        @test size(xₚ_folds, 1) === 5
        @test size(y_folds, 1) === 5
        @test size(t_folds, 1) === 5
        @test size(x_folds[1], 1) === 20
        @test size(xₚ_folds[1], 1) === 20
        @test size(y_folds[1], 1) === 20
        @test size(t_folds[1], 1) === 20
    end
end

@testset "Summarization and Inference" begin
    @testset "Quanities of Interest Errors" begin
        @test_throws ArgumentError GComputation(x, y, t, quantity_of_interest="abc")
        @test_throws ArgumentError DoubleMachineLearning(x, x, y, t, quantity_of_interest="xyz")
    end

    @testset "Task Errors" begin
        @test_throws ArgumentError InterruptedTimeSeries(x₀, y₀, x₁, y₁, task="abc")
        @test_throws ArgumentError GComputation(x, y, t, task="abc")
        @test_throws ArgumentError DoubleMachineLearning(x, x, y, t, task="xyz")
    end

    @testset "Moving Averages" begin
        @test moving_average(Float64[]) isa Array{Float64}
        @test moving_average([1.0]) == [1.0]
        @test moving_average([1.0, 2.0, 3.0]) == [1.0, 1.5, 2.0]
    end
end
