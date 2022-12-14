using CausalELM.Models: ExtremeLearner, RegularizedExtremeLearner, fit!, predict,
    predictcounterfactual!, placebotest
using CausalELM.ActivationFunctions: σ
using Test

# Test classification functionality using a simple XOR test borrowed from 
# ExtremeLearning.jl
x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
y = [0.0, 1.0, 0.0, 1.0]
x_test = [1.0 1.0; 0.0 1.0; 0.0 0.0]

m1 = ExtremeLearner(x, y, 10, σ)
f1 = fit!(m1)
predictions1 = predict(m1, x_test)
predictcounterfactual!(m1, x_test)
placebo1 = placebotest(m1)

m2 = RegularizedExtremeLearner(x, y, 10, σ)
f2 = fit!(m2)
predictions2 = predict(m2, x_test)
predictcounterfactual!(m2, x_test)
placebo2 = placebotest(m2)

 @testset "Model Fit" begin
    @test length(m1.β) == 10
    @test size(m1.weights) == (2, 10)
 end

 @testset "Model Predictions" begin
    @test predictions1[1] < 0.1
    @test predictions1[2] > 0.9
    @test predictions1[3] < 0.1

    # Regularized case
    @test predictions1[1] < 0.1
    @test predictions1[2] > 0.9
    @test predictions1[3] < 0.1

    # Ensure the counterfactual attribute gets step
    @test m1.counterfactual == predictions1
    @test m2.counterfactual == predictions2
 end

 @testset "Placebo Test" begin
    @test length(placebo1) == 2
    @test length(placebo2) == 2
 end
