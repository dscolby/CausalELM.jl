using CausalELM.Models: ExtremeLearner, RegularizedExtremeLearner, fit!, predict
using CausalELM.ActivationFunctions: σ
using Test

# Test classification functionality using a simple XOR test borrowed from 
# ExtremeLearning.jl

x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
y = [0.0, 1.0, 0.0, 1.0]
x_test = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]

m1 = ExtremeLearner(x, y, 10, σ)
f1 = fit!(m1)
predictions1 = predict(m1, x_test)

m2 = RegularizedExtremeLearner(x, y, 10, σ)
f2 = fit!(m2)
predictions2 = predict(m2, x_test)

 @testset "Model Fit" begin
    @test length(m1.β) == 11
    @test size(m1.weights) == (2, 10)
 end

 @testset "Model Predictions" begin
    @test predictions1[1] < 0.1
    @test predictions1[2] > 0.9
    @test predictions1[3] < 0.1
    @test predictions1[4] > 0.9

    # These will be terrible because we are using an L2 penalty with only four data points
    @test -2 < predictions2[1] < 2
    @test -2 < predictions2[2] < 2
    @test -2 < predictions2[3] < 2
    @test -2 < predictions2[4] < 2
 end