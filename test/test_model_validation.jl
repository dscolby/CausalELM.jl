using Test
using CausalELM.Estimators: InterruptedTimeSeries, GComputation, estimatecausaleffect!
using CausalELM.ModelValidation: pval, testcovariateindependence, testomittedpredictor, 
    supwald, validate, sdam, scdm, gvf, split_vector_ways, consecutive, jenks, 
    fake_treatments,counterfactualconsistency

x₀, y₀, x₁, y₁ = Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), randn(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimatecausaleffect!(its)
its_independence = testcovariateindependence(its)
wald_test = supwald(its)
ovb = testomittedpredictor(its)
its_validation = validate(its)

x, y, t = rand(100, 5), vec(rand(1:100, 100, 1)), Float64.([rand()<0.4 for i in 1:100])
g_computer = GComputation(x, y, t, temporal=false)
estimatecausaleffect!(g_computer)
test_outcomes = g_computer.Y[g_computer.T .== 1]

# Test splits for Jenks breaks
three_split = Vector{Vector{Real}}[[[1], [2], [3, 4, 5]], 
    [[1], [2, 3], [4, 5]], [[1], [2, 3, 4], [5]], [[1, 2], [3], [4, 5]], 
    [[1, 2], [3, 4], [5]], [[1, 2, 3], [4], [5]]]

two_split = Vector{Vector{Real}}[[[1], [2, 3, 4, 5]], [[1, 2], [3, 4, 5]], 
    [[1, 2, 3], [4, 5]], [[1, 2, 3, 4], [5]]]

@testset "p-value Argument Validation" begin
    @test_throws ArgumentError pval(rand(10, 1), rand(10), 0.5)
    @test_throws ArgumentError pval(rand(10, 3), rand(10), 0.5)
    @test_throws ArgumentError pval(reduce(hcat, (rand(10), ones(10))), rand(10), 0.5)
    @test_throws ArgumentError pval(reduce(hcat, (float(rand(0:1, 10)), rand(10))), 
        rand(10), 0.5)
end

@testset "p-values for OLS" begin
    @test 0 <= pval(reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 0.5) <= 1
    @test 0 <= pval(reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 0.5, 
        n=100) <= 1
    @test 0 <= pval(reduce(hcat, (reduce(vcat, (zeros(5), ones(5))), ones(10))), randn(10), 
        0.5) <= 1
end

@testset "Interrupted Time Series Assumptions" begin

    # Test testcovariateindependence method
    @test length(its_independence) === 5
    @test all(0 .<= values(its_independence) .<= 1) === true

    # Test supwald method
    @test wald_test isa Dict{String, Real}
    @test wald_test["Hypothesized Break Point"] === size(x₀, 1)
    @test wald_test["Predicted Break Point"] > 0
    @test wald_test["Wald Statistic"] >= 0
    @test 0 <= wald_test["p-value"] <= 1

    # Test omittedvariable method
    # The first test should throw an error because estimatecausaleffect! has not been called
    @test_throws ErrorException testomittedpredictor(InterruptedTimeSeries(x₀, y₀, x₁, y₁))
    @test ovb isa Dict{String, Float64}
    @test isa.(values(ovb), Float64) == Bool[1, 1, 1, 1]

    # All assumptions at once
    @test_throws ErrorException validate(InterruptedTimeSeries(x₀, y₀, x₁, y₁))
    @test its_validation isa Tuple
    @test length(its_validation) === 3
end

# Examples taken from https://www.ehdp.com/methods/jenks-natural-breaks-2.htm
@testset "Jenks Breaks" begin
    @test sdam([5, 4, 9, 10]) == 26
    @test scdm([[4], [5, 9, 10]]) == 14
    @test gvf([[4, 5], [9, 10]]) ≈ 0.96153846153
    @test gvf([[4], [5], [9, 10]]) ≈ 0.9807692307692307
    @test split_vector_ways([1, 2, 3, 4, 5]; n=3) == three_split
    @test split_vector_ways([1, 2, 3, 4, 5]; n=2) == two_split
    @test length(collect(Base.Iterators.flatten(jenks(collect(1:10), 5)))) == 10

    for vec in jenks(collect(1:10), 5)
        @test !isempty(vec)
    end
    setdiff
end

@testset "Counterfactual Consistency" begin
    @test length(fake_treatments(test_outcomes)) == length(test_outcomes)
    @test setdiff(Set(sort(unique(fake_treatments(test_outcomes)))), [1, 2, 3, 4]) == Set()
    @test counterfactualconsistency(g_computer) isa Real
end
