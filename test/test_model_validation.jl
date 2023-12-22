using Test
using CausalELM

x₀, y₀, x₁, y₁ = Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), randn(10)
its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
estimate_causal_effect!(its)
its_independence = CausalELM.covariate_independence(its)
wald_test = CausalELM.sup_wald(its)
ovb = CausalELM.omitted_predictor(its)
its_validation = validate(its)

x, y, t = rand(100, 5), vec(rand(1:100, 100, 1)), Float64.([rand()<0.4 for i in 1:100])
g_computer = GComputation(x, y, t, temporal=false)
estimate_causal_effect!(g_computer)
test_outcomes = g_computer.Y[g_computer.T .== 1]

# Create binary GComputation to test E-values with a binary outcome
x1, y1 = rand(100, 5), Float64.([rand()<0.4 for i in 1:100])
t1 = Float64.([rand()<0.4 for i in 1:100])
binary_g_computer = GComputation(x1, y1, t1, temporal=false)
estimate_causal_effect!(binary_g_computer)

# Create binary GComputation to test E-values with a count outcome
x2, y2 = rand(100, 5), rand(1.0:5.0, 100)
t2 = Float64.([rand()<0.4 for i in 1:100])
count_g_computer = GComputation(x2, y2, t2, temporal=false)
estimate_causal_effect!(count_g_computer)

# Create double machine learning estimator
dml = DoubleMachineLearning(x, y, t)
estimate_causal_effect!(dml)

# Initialize an S-learner
s_learner = SLearner(x, y, t)
estimate_causal_effect!(s_learner)

# Initialize a T-learner
t_learner = TLearner(x, y, t)
estimate_causal_effect!(t_learner)

# Initialize an X-learner
x_learner = XLearner(x, y, t)
estimate_causal_effect!(x_learner)

# Used to test helper functions for Jenks breaks
sum_of_squares2 = CausalELM.sums_of_squares([1, 2, 3, 4, 5], 2)
sum_of_squares3 = CausalELM.sums_of_squares([1, 2, 3, 4, 5], 3)

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
num_breaks = length(unique(CausalELM.best_splits(data, 6)))

@testset "p-values" begin
    @testset "p-value Argument Validation" begin
        @test_throws ArgumentError CausalELM.p_val(rand(10, 1), rand(10), 0.5)
        @test_throws ArgumentError CausalELM.p_val(rand(10, 3), rand(10), 0.5)
        @test_throws ArgumentError CausalELM.p_val(reduce(hcat, (rand(10), ones(10))), 
            rand(10), 0.5)
        @test_throws ArgumentError CausalELM.p_val(reduce(hcat, (float(rand(0:1, 10)), 
            rand(10))), rand(10), 0.5)
    end

    @testset "p-values for OLS" begin
        @test 0 <= CausalELM.p_val(reduce(hcat, (float(rand(0:1, 10)), ones(10))), 
            rand(10), 0.5) <= 1
        @test 0 <= CausalELM.p_val(reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 
            0.5, n=100) <= 1
        @test 0 <= CausalELM.p_val(reduce(hcat, (reduce(vcat, (zeros(5), ones(5))), 
            ones(10))), randn(10), 0.5) <= 1
end
end

@testset "Interrupted Time Series Assumptions" begin

    @testset "Covariate Independence Assumption" begin
        # Test covariate_independence method
        @test length(its_independence) === 5
        @test all(0 .<= values(its_independence) .<= 1) === true
    end

    @testset "Wald Supremeum Test for Alternative Change Point" begin
        # Test sup_wald method
        @test wald_test isa Dict{String, Real}
        @test wald_test["Hypothesized Break Point"] === size(x₀, 1)
        @test wald_test["Predicted Break Point"] > 0
        @test wald_test["Wald Statistic"] >= 0
        @test 0 <= wald_test["p-value"] <= 1
    end

    @testset "Sensitivity to Omitted Predictors" begin
        # Test omittedvariable method
        # The first test should throw an error since estimatecausaleffect! was not called
        @test_throws ErrorException CausalELM.omitted_predictor(InterruptedTimeSeries(x₀, 
            y₀, x₁, y₁))
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
        @test CausalELM.class_pointers([1, 2, 3, 4, 5], 2, sum_of_squares2)[:, 1] == ones(
                                                                                        Int, 
                                                                                        5
                                                                                        )
        @test length(CausalELM.class_pointers([1, 2, 3, 4, 5], 2, 
            CausalELM.sums_of_squares([1, 2, 3, 4, 5], 2))) == 10
        @test CausalELM.class_pointers([1, 2, 3, 4, 5], 3, sum_of_squares3)[:, 1] == ones(
                                                                                        Int, 
                                                                                        5
                                                                                        )
        @test length(CausalELM.class_pointers([1, 2, 3, 4, 5], 3, 
            CausalELM.sums_of_squares([1, 2, 3, 4, 5], 3))) == 15
        @test length(CausalELM.class_pointers([1, 2, 3, 4, 5], 3, 
            CausalELM.sums_of_squares([1, 2, 3, 4, 5], 3))) == 15
        @test length(CausalELM.backtrack_to_find_breaks([1, 2, 3, 4, 5], 
            [1 1 1 1 1; 2 2 3 4 5])) == 5
        @test CausalELM.variance([1, 2, 3, 4, 5]) == 2.0
    end

    @testset "Jenks Breaks Function" begin
        @test 2 <= length(unique(CausalELM.jenks_breaks(data, num_breaks))) <= num_breaks
    end

    @testset "Helpers to Find the Best Number of Breaks" begin
        @test length(unique(CausalELM.fake_treatments([1, 2, 3, 4, 5], 3))) == 3
        @test CausalELM.group_by_class([1, 2, 3, 4, 5], [1, 1, 1, 2, 3]) == [[1, 2, 3], [4], 
                                                                                [5]]
        @test CausalELM.sdam([5, 4, 9, 10]) == 26
        @test CausalELM.scdm([[4], [5, 9, 10]]) == 14
        @test CausalELM.gvf([[4, 5], [9, 10]]) ≈ 0.96153846153
        @test CausalELM.gvf([[4], [5], [9, 10]]) ≈ 0.9807692307692307
        @test length(CausalELM.best_splits(test_outcomes, 5)) == length(test_outcomes)
        @test setdiff(Set(sort(unique(CausalELM.fake_treatments(test_outcomes, 3)))), 
            [1, 2, 3]) == Set()
    end
end

@testset "E-values" begin
    @testset "Generating E-values" begin
        @test CausalELM.e_value(binary_g_computer) isa Real
        @test CausalELM.e_value(count_g_computer) isa Real
        @test CausalELM.e_value(g_computer) isa Real
        @test CausalELM.e_value(dml) isa Real
        @test CausalELM.e_value(s_learner) isa Real
        @test CausalELM.e_value(t_learner) isa Real
        @test CausalELM.e_value(x_learner) isa Real
    end
end

@testset "G-Computation Assumptions" begin
    @testset "Counterfactual Consistency" begin
        @test CausalELM.counterfactual_consistency(g_computer) isa Real
    end

    @testset "CausalELM.exchangeability" begin
        @test CausalELM.exchangeability(binary_g_computer) isa Real
        @test CausalELM.exchangeability(count_g_computer) isa Real
        @test CausalELM.exchangeability(g_computer) isa Real
    end

    @testset "CausalELM.positivity" begin
        @test size(CausalELM.positivity(binary_g_computer), 2) == size(binary_g_computer.X, 
                                                                    2)+1
        @test size(CausalELM.positivity(count_g_computer), 2) == size(count_g_computer.X, 
                                                                    2)+1
        @test size(CausalELM.positivity(g_computer), 2) == size(g_computer.X, 2)+1
    end
end

@testset "All Assumptions for G-computation" begin
    @test_throws ErrorException validate(GComputation(x, y, t))
    @test length(validate(binary_g_computer)) == 3
    @test length(validate(count_g_computer)) == 3
    @test length(validate(g_computer)) == 3
end

@testset "Double Machine Learning Assumptions" begin
    @test CausalELM.counterfactual_consistency(dml) isa Real
    @test CausalELM.exchangeability(dml) isa Real
    @test size(CausalELM.positivity(dml), 2) == size(dml.X, 2)+1
    @test length(validate(dml)) == 3
end

@testset "Metalearner Assumptions" begin
    @test CausalELM.counterfactual_consistency(s_learner) isa Real
    @test CausalELM.exchangeability(s_learner) isa Real
    @test size(CausalELM.positivity(s_learner), 2) == size(s_learner.X, 2)+1
    @test CausalELM.counterfactual_consistency(t_learner) isa Real
    @test CausalELM.exchangeability(t_learner) isa Real
    @test size(CausalELM.positivity(t_learner), 2) == size(t_learner.X, 2)+1
    @test CausalELM.counterfactual_consistency(x_learner) isa Real
    @test CausalELM.exchangeability(x_learner) isa Real
    @test size(CausalELM.positivity(x_learner), 2) == size(x_learner.X, 2)+1
    @test length(validate(s_learner)) == 3
    @test length(validate(t_learner)) == 3
    @test length(validate(x_learner)) == 3
end
