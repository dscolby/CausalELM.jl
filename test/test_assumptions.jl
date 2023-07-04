using CausalELM.Assumptions: pval, testcovariateindependence, testomittedpredictor
using CausalELM.Estimators: InterruptedTimeSeries, estimatecausaleffect!
using Test

x₀, y₀, x₁, y₁ = Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), randn(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimatecausaleffect!(its)
its_independence = testcovariateindependence(its)
ovb = testomittedpredictor(its)

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

    # Test omittedvariable method
    # The first test should throw an error because estimatecausaleffect! has not been called
    @test_throws ErrorException testomittedpredictor(InterruptedTimeSeries(x₀, y₀, x₁, y₁))
    @test ovb isa Dict{String, Float64}
    @test isa.(values(ovb), Float64) == Bool[1, 1, 1, 1]
end
