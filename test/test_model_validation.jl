using Test
using CausalELM.Estimators: InterruptedTimeSeries, GComputation, estimate_causal_effect!
using CausalELM.ModelValidation: pval, testcovariateindependence, testomittedpredictor, 
    supwald, validate, counterfactualconsistency, sumsofsquares, classpointers, 
    backtrack_to_find_breaks, jenksbreaks, faketreatments, sdam, scdm, gvf, bestsplits, 
    groupbyclass, variance

x₀, y₀, x₁, y₁ = Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), randn(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)
its_independence = testcovariateindependence(its)
wald_test = supwald(its)
ovb = testomittedpredictor(its)
its_validation = validate(its)

x, y, t = rand(100, 5), vec(rand(1:100, 100, 1)), Float64.([rand()<0.4 for i in 1:100])
g_computer = GComputation(x, y, t, temporal=false)
estimate_causal_effect!(g_computer)
test_outcomes = g_computer.Y[g_computer.T .== 1]

# Used to test helper functions for Jenks breaks
sum_of_squares2 = sumsofsquares([1, 2, 3, 4, 5], 2)
sum_of_squares3 = sumsofsquares([1, 2, 3, 4, 5], 3)

# Generate synthetic data with three distinct clusters
function generate_synthetic_data()
    cluster1 = rand(1:10, 50) .+ randn(50)
    cluster2 = rand(20:30, 60) .+ randn(60)
    cluster3 = rand(40:50, 40) .+ randn(40)
    data = vcat(cluster1, cluster2, cluster3)
    return data
end

# Generate synthetic data
data = generate_synthetic_data()

# Find the best number of breaks using the Jenks Natural Breaks algorithm
num_breaks = length(unique(bestsplits(data, 6)))

@testset "p-values" begin
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
        @test 0 <= pval(reduce(hcat, (reduce(vcat, (zeros(5), ones(5))), ones(10))), 
            randn(10), 0.5) <= 1
end
end

@testset "Interrupted Time Series Assumptions" begin

    @testset "Covariate Independence Assumption" begin
        # Test testcovariateindependence method
        @test length(its_independence) === 5
        @test all(0 .<= values(its_independence) .<= 1) === true
    end

    @testset "Wald Supremeum Test for Alternative Change Point" begin
        # Test supwald method
        @test wald_test isa Dict{String, Real}
        @test wald_test["Hypothesized Break Point"] === size(x₀, 1)
        @test wald_test["Predicted Break Point"] > 0
        @test wald_test["Wald Statistic"] >= 0
        @test 0 <= wald_test["p-value"] <= 1
    end

    @testset "Sensitivity to Omitted Predictors" begin
        # Test omittedvariable method
        # The first test should throw an error since estimatecausaleffect! was not called
        @test_throws ErrorException testomittedpredictor(InterruptedTimeSeries(x₀, y₀, x₁, y₁))
        @test ovb isa Dict{String, Float64}
        @test isa.(values(ovb), Float64) == Bool[1, 1, 1, 1]
    end

    @testset "All Three Assumptions" begin
        # All assumptions at once
        @test_throws ErrorException validate(InterruptedTimeSeries(x₀, y₀, x₁, y₁))
        @test its_validation isa Tuple
        @test length(its_validation) === 3
    end
end


@testset "Jenks Breaks" begin
    @testset "Helper Functions for Finding Breaks" begin
        @test sum_of_squares2[1, 1] == 0.0
        @test sum_of_squares2[5, 2] == 1.75
        @test sum_of_squares3[1, 1] == 0.0
        @test sum_of_squares3[1, 2] == 0.0
        @test sum_of_squares3[5, 3] == 1.6666666666666665
        @test classpointers([1, 2, 3, 4, 5], 2, sum_of_squares2)[:, 1] == ones(Int, 5)
        @test length(classpointers([1, 2, 3, 4, 5], 2, 
            sumsofsquares([1, 2, 3, 4, 5], 2))) == 10
        @test classpointers([1, 2, 3, 4, 5], 3, sum_of_squares3)[:, 1] == ones(Int, 5)
        @test length(classpointers([1, 2, 3, 4, 5], 3, 
            sumsofsquares([1, 2, 3, 4, 5], 3))) == 15
        @test length(classpointers([1, 2, 3, 4, 5], 3, 
            sumsofsquares([1, 2, 3, 4, 5], 3))) == 15
        @test length(backtrack_to_find_breaks([1, 2, 3, 4, 5], 
            [1 1 1 1 1; 2 2 3 4 5])) == 5
        @test variance([1, 2, 3, 4, 5]) == 2.0
    end

    @testset "Jenks Breaks Function" begin
        @test length(unique(jenksbreaks(data, num_breaks))) == num_breaks
    end

    @testset "Helpers to Find the Best Number of Breaks" begin
        @test length(unique(faketreatments([1, 2, 3, 4, 5], 3))) == 3
        @test groupbyclass([1, 2, 3, 4, 5], [1, 1, 1, 2, 3]) == [[1, 2, 3], [4], [5]]
        @test sdam([5, 4, 9, 10]) == 26
        @test scdm([[4], [5, 9, 10]]) == 14
        @test gvf([[4, 5], [9, 10]]) ≈ 0.96153846153
        @test gvf([[4], [5], [9, 10]]) ≈ 0.9807692307692307
        @test length(bestsplits(test_outcomes, 5)) == length(test_outcomes)
        @test setdiff(Set(sort(unique(faketreatments(test_outcomes, 3)))), 
            [1, 2, 3]) == Set()
    end
end

@testset "Counterfactual Consistency" begin
    @test counterfactualconsistency(g_computer) isa Real
end
