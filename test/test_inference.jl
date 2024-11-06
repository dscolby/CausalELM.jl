using Test
using CausalELM

x, t, y = rand(100, 5),
[rand() < 0.4 for i in 1:100],
Float64.([rand() < 0.4 for i in 1:100])

g_computer = GComputation(x, t, y)
estimate_causal_effect!(g_computer)
g_inference = CausalELM.generate_null_distribution(g_computer, 1000)
p1, stderr1 = CausalELM.p_value_and_std_err(g_inference, CausalELM.mean(g_inference))
lb1, ub1 = CausalELM.confidence_interval(g_inference)
p11, stderr11, lb11, ub11 = CausalELM.quantities_of_interest(g_computer, 100)
summary1 = summarize(g_computer, n=100, inference=true)

dm = DoubleMachineLearning(x, t, y)
estimate_causal_effect!(dm)
dm_inference = CausalELM.generate_null_distribution(dm, 1000)
p2, stderr2 = CausalELM.p_value_and_std_err(dm_inference, CausalELM.mean(dm_inference))
lb2, ub2 = CausalELM.confidence_interval(dm_inference)
summary2 = summarize(dm, n=100)

# With a continuous treatment variable
dm_continuous = DoubleMachineLearning(x, t, rand(1:4, 100))
estimate_causal_effect!(dm_continuous)
dm_continuous_inference = CausalELM.generate_null_distribution(dm_continuous, 1000)
p3, stderr3 = CausalELM.p_value_and_std_err(
    dm_continuous_inference, CausalELM.mean(dm_continuous_inference)
)
lb3, ub3 = CausalELM.confidence_interval(dm_continuous_inference)
summary3 = summarize(dm_continuous, n=100)

x₀, y₀, x₁, y₁ = rand(1:100, 100, 5), rand(100), rand(10, 5), rand(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)
summary4 = summarize(its, n=100)
summary4_inference = summarize(its, n=100, inference=true)

# Null distributions for the mean and cummulative changes
its_inference1 = CausalELM.generate_null_distribution(its, 1000, true)
its_inference2 = CausalELM.generate_null_distribution(its, 10, false)
lb4, ub4 = CausalELM.confidence_interval(its_inference1)
p4, stderr4 = CausalELM.p_value_and_std_err(its_inference1, CausalELM.mean(its_inference1))
p44, stderr44, lb44, ub44 = CausalELM.quantities_of_interest(its, 100, true)

slearner = SLearner(x, t, y)
estimate_causal_effect!(slearner)
summary5 = summarize(slearner, n=100)

tlearner = TLearner(x, t, y)
estimate_causal_effect!(tlearner)
tlearner_inference = CausalELM.generate_null_distribution(tlearner, 1000)
lb6, ub6 = CausalELM.confidence_interval(tlearner_inference)
p6, stderr6 = CausalELM.p_value_and_std_err(
    tlearner_inference, CausalELM.mean(tlearner_inference)
)
p66, stderr66, lb66, ub66 = CausalELM.quantities_of_interest(tlearner, 100)
summary6 = summarize(tlearner, n=100)

xlearner = XLearner(x, t, y)
estimate_causal_effect!(xlearner)
xlearner_inference = CausalELM.generate_null_distribution(xlearner, 1000)
lb7, ub7 = CausalELM.confidence_interval(xlearner_inference)
p7, stderr7 = CausalELM.p_value_and_std_err(
    xlearner_inference, CausalELM.mean(xlearner_inference)
)
summary7 = summarize(xlearner, n=100)
summary8 = summarise(xlearner, n=100)

rlearner = RLearner(x, t, y)
estimate_causal_effect!(rlearner)
summary9 = summarize(rlearner, n=100)

dr_learner = DoublyRobustLearner(x, t, y)
estimate_causal_effect!(dr_learner)
dr_learner_inference = CausalELM.generate_null_distribution(dr_learner, 1000)
lb8, ub8 = CausalELM.confidence_interval(dr_learner_inference)
p8, stderr8 = CausalELM.p_value_and_std_err(
    dr_learner_inference, CausalELM.mean(dr_learner_inference)
)
summary10 = summarize(dr_learner, n=100)

@testset "Generating Null Distributions" begin
    @test size(g_inference, 1) === 1000
    @test g_inference isa Array{Float64}
    @test size(dm_inference, 1) === 1000
    @test dm_inference isa Array{Float64}
    @test size(dm_continuous_inference, 1) === 1000
    @test dm_continuous_inference isa Array{Float64}
    @test size(its_inference1, 1) === 1000
    @test its_inference1 isa Array{Float64}
    @test size(its_inference2, 1) === 10
    @test its_inference2 isa Array{Float64}
    @test size(tlearner_inference, 1) === 1000
    @test tlearner_inference isa Array{Float64}
    @test size(xlearner_inference, 1) === 1000
    @test xlearner_inference isa Array{Float64}
    @test size(dr_learner_inference, 1) === 1000
    @test dr_learner_inference isa Array{Float64}
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
    @test 1 >= p6 >= 0
    @test stderr6 > 0
    @test 1 >= p7 >= 0
    @test stderr7 > 0
    @test 1 >= p8 >= 0
    @test stderr8 > 0
end

@testset "Confidence Intervals" begin
    @test lb1 < g_computer.causal_effect < ub1
    @test lb2 < dm.causal_effect < ub2
    @test lb3 < dm_continuous.causal_effect < ub3
    @test lb4 < CausalELM.mean(its.causal_effect) < ub4
    @test lb6 < CausalELM.mean(tlearner.causal_effect) < ub6
    @test lb7 < CausalELM.mean(xlearner.causal_effect) < ub7
    @test lb8 < CausalELM.mean(dr_learner.causal_effect) < ub8
end

@testset "All Quantities of Interest" begin
    @test lb11 < g_computer.causal_effect < ub11
    @test 1 >= p11 >= 0
    @test stderr11 > 0
    @test lb44 < CausalELM.mean(its.causal_effect) < ub44
    @test 1 >= p44 >= 0
    @test stderr44 > 0
    @test lb66 < CausalELM.mean(tlearner.causal_effect) < ub66
    @test 1 >= p66 >= 0
    @test stderr66 > 0
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

    # Interrupted Time Series with randomization inference
    @test summary4_inference["Standard Error"] !== NaN
    @test summary4_inference["p-value"] !== NaN

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

    # Doubly robust learner
    for (k, v) in summary10
        @test !isnothing(v)
    end
end

@testset "Error Handling" begin
    @test_throws ErrorException summarize(InterruptedTimeSeries(x₀, y₀, x₁, y₁), n=10)
    @test_throws ErrorException summarize(GComputation(x, y, t))
    @test_throws ErrorException summarize(TLearner(x, y, t))
end

@test CausalELM.mean([1, 2, 3]) == 2
