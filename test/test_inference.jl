using CausalELM.Inference: generatenulldistribution, quantitiesofinterest, summarize
using CausalELM.Estimators: CausalEstimator, EventStudy, GComputation, DoublyRobust
using CausalELM.Metalearners: SLearner, TLearner, XLearner, Metalearner, 
    estimatecausaleffect!
using Test

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]

g_computer = GComputation(x, y, t)
estimatecausaleffect!(g_computer)
g_inference = generatenulldistribution(g_computer)
p1, stderr1 = quantitiesofinterest(g_computer)
summary1 = summarize(g_computer)

dr = DoublyRobust(x, x, y, t)
estimatecausaleffect!(dr)
dr_inference = generatenulldistribution(dr)
p2, stderr2 = quantitiesofinterest(dr)
summary2 = summarize(dr)

x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
event_study = EventStudy(x₀, y₀, x₁, y₁)
estimatecausaleffect!(event_study)
summary3 = summarize(event_study, 10)

# Null distributions for the mean and cummulative changes
event_study_inference1 = generatenulldistribution(event_study, 10)
event_study_inference2 = generatenulldistribution(event_study, 10, false)
p3, stderr3 = quantitiesofinterest(event_study, 10)

slearner = SLearner(x, y, t)
estimatecausaleffect!(slearner)
slearner_inference = generatenulldistribution(slearner)
p4, stderr4 = quantitiesofinterest(slearner)
summary4 = summarize(slearner)

tlearner = TLearner(x, y, t)
estimatecausaleffect!(tlearner)
tlearner_inference = generatenulldistribution(tlearner)
p5, stderr5 = quantitiesofinterest(tlearner)
summary5 = summarize(tlearner)

xlearner = XLearner(x, y, t)
estimatecausaleffect!(xlearner)
xlearner_inference = generatenulldistribution(xlearner)
p6, stderr6 = quantitiesofinterest(xlearner)
summary6 = summarize(xlearner)

@testset "Generating Null Distributions" begin
    @test size(g_inference, 1) === 1000
    @test g_inference isa Array{Float64}
    @test size(dr_inference, 1) === 1000
    @test dr_inference isa Array{Float64}
    @test size(event_study_inference1, 1) === 10
    @test event_study_inference1 isa Array{Float64}
    @test size(event_study_inference2, 1) === 10
    @test event_study_inference2 isa Array{Float64}
    @test size(slearner_inference, 1) === 1000
    @test slearner_inference isa Array{Float64}
    @test size(tlearner_inference, 1) === 1000
    @test tlearner_inference isa Array{Float64}
    @test size(xlearner_inference, 1) === 1000
    @test xlearner_inference isa Array{Float64}
end

@testset "More Splits Than Observations" begin
    @test_throws BoundsError generatenulldistribution(event_study, 101)
    @test_throws BoundsError generatenulldistribution(event_study, 100)
end

@testset "P-values and Standard Errors" begin
    @test 1 >= p1 >= 0
    @test stderr1 > 0
    @test 1 >= p2 >= 0
    @test stderr2 > 0
    @test 1 >= p3 >= 0
    @test stderr3 > 0
    @test 1 >= p4 >= 0
    @test stderr4 > 0
    @test 1 >= p5 >= 0
    @test stderr5 > 0
    @test 1 >= p6 >= 0
    @test stderr6 > 0
end

@testset "Full Summaries" begin
    # G-Computation
    for (k, v) in summary1
        @test !isnothing(v)
    end

    # Doubly Robust Estimation
    for (k, v) in summary2
        @test !isnothing(v)
    end

    # Event Study
    for (k, v) in summary3
        @test !isnothing(v)
    end

    # S-Learners
    for (k, v) in summary4
        @test !isnothing(v)
    end

    # T-Learners
    for (k, v) in summary5
        @test !isnothing(v)
    end

    # X-Learners
    for (k, v) in summary6
        @test !isnothing(v)
    end
end

@test mean([1, 2, 3]) == 2
