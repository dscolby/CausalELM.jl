using CausalELM.Estimators: EventStudy, GComputation, DoublyRobust, estimatecausaleffect!, 
    mean
using Test

x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
event_study = EventStudy(x₀, y₀, x₁, y₁)
estimatecausaleffect!(event_study)

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]
g_computer = GComputation(x, y, t)
estimatecausaleffect!(g_computer)

gcomputer_att = GComputation(x, y, t, quantity_of_interest="ATT")
estimatecausaleffect!(gcomputer_att)

dr = DoublyRobust(x, x, y, t)
estimatecausaleffect!(dr)

# No regularization
dr_noreg = DoublyRobust(x, x, y, t, regularized=false)
estimatecausaleffect!(dr_noreg)

# Estimating the ATT instead of the ATE
dr_att = DoublyRobust(x, x, y, t, quantity_of_interest="ATT")
estimatecausaleffect!(dr_att)

# Estimating the ATT without regularization
dr_att_noreg = DoublyRobust(x, x, y, t, regularized=false, quantity_of_interest="ATT")
estimatecausaleffect!(dr_att_noreg)

@testset "Event Study Structure" begin
    @test event_study.X₀ !== Nothing
    @test event_study.Y₀ !== Nothing
    @test event_study.X₁ !== Nothing
    @test event_study.Y₁ !== Nothing
end

@testset "Event Study Estimation" begin
    @test isa(event_study.β, Array)
    @test isa(event_study.Ŷ, Array)
    @test isa(event_study.abnormal_returns, Array)
    @test isa(event_study.placebo_test, Tuple{Vector{Float64}, Vector{Float64}})
end

@testset "G-Computation Structure" begin
    @test g_computer.X !== Nothing
    @test g_computer.Y !== Nothing
    @test g_computer.T !== Nothing
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
    @test_throws ArgumentError EventStudy(x₀, y₀, x₁, y₁, task="abc")
    @test_throws ArgumentError GComputation(x, y, t, task="abc")
    @test_throws ArgumentError DoublyRobust(x, x, y, t, task="xyz")
end
