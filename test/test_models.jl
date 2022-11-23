using CausalELM.Models: Elm, fit!, predict
using CausalELM.ActivationFunctions: σ
using Test

# Test classification functionality using a simple XOR test borrowed from 
# ExtremeLearning.jl

x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
y = [0.0, 1.0, 0.0, 1.0]
x_test = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]

m1 = Elm(x, y, 10, σ)
f1 = fit!(m1)
predictions = predict(m1, x_test)

 @testset "Model Fit" begin
    @test length(m1.β) == 11
    @test size(m1.weights) == (2, 10)
 end

 @testset "Model Predictions" begin
    @test predictions[1] < 0.1
    @test predictions[2] > 0.9
    @test predictions[3] < 0.1
    @test predictions[4] > 0.9
 end