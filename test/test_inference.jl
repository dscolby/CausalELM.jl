using CausalELM.Inference: generatenulldistribution, quantitiesofinterest
using CausalELM.Estimators: CausalEstimator, EventStudy
using CausalELM.Metalearners: SLearner, Metalearner, estimatecausaleffect!
using Test

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]

g_computer = GComputation(x, y, t)
estimatecausaleffect!(g_computer)
g_inference = generatenulldistribution(g_computer)
p1, stderr1 = quantitiesofinterest(g_computer)

x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
event_study = EventStudy(x₀, y₀, x₁, y₁)
estimatecausaleffect!(event_study)

# Null distributions for the mean and cummulative changes
event_study_inference1 = generatenulldistribution(event_study, 10)
event_study_inference2 = generatenulldistribution(event_study, 10, false)
p2, stderr2 = quantitiesofinterest(event_study, 10)

slearner = SLearner(x, y, t)
estimatecausaleffect!(slearner)
slearner_inference = generatenulldistribution(slearner)

@testset "Generating Null Distributions" begin
    @test size(g_inference, 1) === 1000
    @test g_inference isa Array{Float64}
    @test size(event_study_inference1, 1) === 10
    @test event_study_inference1 isa Array{Float64}
    @test size(event_study_inference2, 1) === 10
    @test event_study_inference2 isa Array{Float64}
    @test size(slearner_inference, 1) === 1000
    @test slearner_inference isa Array{Float64}
end

@testset "More Splits Than Observations" begin
    @test_throws BoundsError generatenulldistribution(event_study, 101)
end

@testset "P-values and Standard Errors" begin
    @test 1 >= p1 >= 0
    @test stderr1 > 0
    @test 1 >= p2 >= 0
    @test stderr2 > 0
end
