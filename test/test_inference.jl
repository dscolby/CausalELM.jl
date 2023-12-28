using Test
using CausalELM

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]

g_computer = GComputation(x, y, t)
estimate_causal_effect!(g_computer)
g_inference = CausalELM.generate_null_distribution(g_computer)
p1, stderr1 = CausalELM.quantities_of_interest(g_computer)
summary1 = summarize(g_computer)

dm = DoubleMachineLearning(x, y, t)
estimate_causal_effect!(dm)
dm_inference = CausalELM.generate_null_distribution(dm)
p2, stderr2 = CausalELM.quantities_of_interest(dm)
summary2 = summarize(dm)

# With a continuous treatment variable
dm_continuous = DoubleMachineLearning(x, rand(1:4, 100), t)
estimate_causal_effect!(dm_continuous)
dm_continuous_inference = CausalELM.generate_null_distribution(dm_continuous)
p3, stderr3 = CausalELM.quantities_of_interest(dm_continuous)
summary3 = summarize(dm_continuous)

x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)
summary4 = summarize(its, 10)

# Null distributions for the mean and cummulative changes
its_inference1 = CausalELM.generate_null_distribution(its, 10)
its_inference2 = CausalELM.generate_null_distribution(its, 10, false)
p4, stderr4 = CausalELM.quantities_of_interest(its, 10)

slearner = SLearner(x, y, t)
estimate_causal_effect!(slearner)
slearner_inference = CausalELM.generate_null_distribution(slearner)
p5, stderr5 = CausalELM.quantities_of_interest(slearner)
summary5 = summarize(slearner)

tlearner = TLearner(x, y, t)
estimate_causal_effect!(tlearner)
tlearner_inference = CausalELM.generate_null_distribution(tlearner)
p6, stderr6 = CausalELM.quantities_of_interest(tlearner)
summary6 = summarize(tlearner)

xlearner = XLearner(x, y, t)
estimate_causal_effect!(xlearner)
xlearner_inference = CausalELM.generate_null_distribution(xlearner)
p7, stderr7 = CausalELM.quantities_of_interest(xlearner)
summary7 = summarize(xlearner)
summary8 = summarise(xlearner)

rlearner = RLearner(x, y, t)
estimate_causal_effect!(rlearner)
summary9 = summarize(rlearner)

@testset "Generating Null Distributions" begin
    @test size(g_inference, 1) === 1000
    @test g_inference isa Array{Float64}
    @test size(dm_inference, 1) === 1000
    @test dm_inference isa Array{Float64}
    @test size(dm_continuous_inference, 1) === 1000
    @test dm_continuous_inference isa Array{Float64}
    @test size(its_inference1, 1) === 10
    @test its_inference1 isa Array{Float64}
    @test size(its_inference2, 1) === 10
    @test its_inference2 isa Array{Float64}
    @test size(slearner_inference, 1) === 1000
    @test slearner_inference isa Array{Float64}
    @test size(tlearner_inference, 1) === 1000
    @test tlearner_inference isa Array{Float64}
    @test size(xlearner_inference, 1) === 1000
    @test xlearner_inference isa Array{Float64}
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
    @test 1 >= p7 >= 0
    @test stderr7 > 0
end

@testset "Full Summaries" begin
    # G-Computation
    for (k, v) in summary1
        @test !isnothing(v)
    end

    # Double Machine Learning
    for (k, v) in summary2
        @test !isnothing(v)
    end

    # Double Machine Learning with continuous treatment
    for (k, v) in summary3
        @test !isnothing(v)
    end

    # Interrupted Time Series
    for (k, v) in summary4
        @test !isnothing(v)
    end

    # S-Learners
    for (k, v) in summary5
        @test !isnothing(v)
    end

    # T-Learners
    for (k, v) in summary6
        @test !isnothing(v)
    end

    # X-Learners
    for (k, v) in summary7
        @test !isnothing(v)
    end

    # Testing the British spelling of summarise
    for (k, v) in summary8
        @test !isnothing(v)
    end

    # R-Learners
    for (k, v) in summary9
        @test !isnothing(v)
    end
end

@testset "Error Handling" begin
    @test_throws ErrorException summarize(InterruptedTimeSeries(x₀, y₀, x₁, y₁), 10)
    @test_throws ErrorException summarize(GComputation(x, y, t))
    @test_throws ErrorException summarize(TLearner(x, y, t))
end

@test CausalELM.mean([1, 2, 3]) == 2
