using CausalELM.Estimators: InterruptedTimeSeries, GComputation, DoublyRobust, 
    estimatecausaleffect!, mean, crossfittingsets, firststage!, ate!, 
    predictpropensityscore, predictcontroloutcomes, predicttreatmentoutcomes, movingaverage
using CausalELM.Models: ExtremeLearningMachine
using Test

x₀, y₀, x₁, y₁ = Float64.(rand(1:100, 100, 5)), rand(100), rand(10, 5), rand(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimatecausaleffect!(its)

# No autoregressive term
its_no_ar = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimatecausaleffect!(its_no_ar)

x, y, t = rand(100, 5), vec(rand(1:100, 100, 1)), Float64.([rand()<0.4 for i in 1:100])
g_computer = GComputation(x, y, t)
estimatecausaleffect!(g_computer)

gcomputer_att = GComputation(x, y, t, quantity_of_interest="ATT")
estimatecausaleffect!(gcomputer_att)

# Mak sure the data isn't shuffled
g_computer_ts = GComputation(float.(hcat([1:10;], 11:20)), rand(10), 
    Float64.([rand()<0.4 for i in 1:10]), temporal=true)

dr = DoublyRobust(x, x, y, t)
dr.num_neurons = 5
ps_mod, control_mod = firststage!(dr, x₀, x, t, y₀)
treat_mod = ate!(dr, x₁, y₁)
dr.num_neurons = 0
estimatecausaleffect!(dr)
x_folds, xₚ_folds, y_folds, t_folds = crossfittingsets(dr)

# No regularization
dr_noreg = DoublyRobust(x, x, y, t, regularized=false)
estimatecausaleffect!(dr_noreg)

# Estimating the ATT instead of the ATE
dr_att = DoublyRobust(x, x, y, t, quantity_of_interest="ATT")
estimatecausaleffect!(dr_att)

# Estimating the ATT without regularization
dr_att_noreg = DoublyRobust(x, x, y, t, regularized=false, quantity_of_interest="ATT")
estimatecausaleffect!(dr_att_noreg)

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
    @test isa(its.placebo_test, Tuple{Vector{Float64}, Vector{Float64}})

    # Without autocorrelation
    @test isa(its_no_ar.β, Array)
    @test isa(its_no_ar.Ŷ, Array)
    @test isa(its_no_ar.Δ, Array)
    @test isa(its_no_ar.placebo_test, Tuple{Vector{Float64}, Vector{Float64}})
end

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

@testset "Doubly Robust Estimation Structure" begin
    @test dr.X !== Nothing
    @test dr.Y !== Nothing
    @test dr.T !== Nothing

    # No regularization
    @test dr_noreg.X !== Nothing
    @test dr_noreg.Y !== Nothing
    @test dr_noreg.T !== Nothing

    # Using the ATT
    @test dr_att.X !== Nothing
    @test dr_att.Y !== Nothing
    @test dr_att.T !== Nothing

    # Using the ATT without regularization
    @test dr_att_noreg.X !== Nothing
    @test dr_att_noreg.Y !== Nothing
    @test dr_att_noreg.T !== Nothing
end

@testset "DRE First Stage" begin
    @test ps_mod isa ExtremeLearningMachine
    @test control_mod isa ExtremeLearningMachine
end

@testset "DRE Second Stage" begin
    @test treat_mod isa ExtremeLearningMachine
end

@testset "DRE Predictions" begin
    @test predictpropensityscore(ps_mod, x₀) isa Array{Float64}
    @test predictcontroloutcomes(control_mod, x₀) isa Array{Float64}
    @test predicttreatmentoutcomes(treat_mod, x₀) isa Array{Float64}
end

@testset "Doubly Robust Estimation" begin
    @test dr.ps isa Array{Float64}
    @test dr.μ₀ isa Array{Float64}
    @test dr.μ₁ isa Array{Float64}
    @test dr.causal_effect isa Float64

    # No regularization
    @test dr_noreg.ps isa Array{Float64}
    @test dr_noreg.μ₀ isa Array{Float64}
    @test dr_noreg.μ₁ isa Array{Float64}
    @test dr_noreg.causal_effect isa Float64

    # Using the ATT
    @test dr_att.ps isa Array{Float64}
    @test dr_att.μ₀ isa Array{Float64}
    @test dr_att.causal_effect isa Float64

    # Using the ATT with no regularization
    @test dr_att_noreg.ps isa Array{Float64}
    @test dr_att_noreg.μ₀ isa Array{Float64}
    @test dr_att_noreg.causal_effect isa Float64
end

@testset "Quanities of Interest Errors" begin
    @test_throws ArgumentError GComputation(x, y, t, quantity_of_interest="abc")
    @test_throws ArgumentError DoublyRobust(x, x, y, t, quantity_of_interest="xyz")
end

@testset "Task Errors" begin
    @test_throws ArgumentError InterruptedTimeSeries(x₀, y₀, x₁, y₁, task="abc")
    @test_throws ArgumentError GComputation(x, y, t, task="abc")
    @test_throws ArgumentError DoublyRobust(x, x, y, t, task="xyz")
end

@testset "X and Xₚ Different Size Error" begin
    @test_throws ArgumentError DoublyRobust(x, rand(50, 5), y, t)
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

@testset "Moving Averages" begin
    @test movingaverage(Float64[]) isa Array{Float64}
    @test movingaverage([1.0]) == [1.0]
    @test movingaverage([1.0, 2.0, 3.0]) == [1.0, 1.5, 2.0]
end
