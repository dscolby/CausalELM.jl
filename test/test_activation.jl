using CausalELM.ActivationFunctions: binarystep, σ, tanh, relu, leakyrelu, swish, softmax,
    softplus, gelu, gaussian, hardtanh, elish
using Test

@testset "Binary Step Activation" begin

    # Single numbers
    @test binarystep(-1000.) == 0
    @test binarystep(-0.00001) == 0
    @test binarystep(0.,) == 1
    @test binarystep(0.00001) == 1
    @test binarystep(100.) == 1
    
    # Vectors
    @test binarystep([-100]) == [0]
    @test binarystep([-100., -100., -100.]) == [0, 0, 0]
    @test binarystep([-0.00001, -0.00001, -0.00001]) == [0, 0, 0]
    @test binarystep([0, 0, 0]) == [1, 1, 1]
    @test binarystep([0.00001, 0.00001, 0.00001]) == [1, 1, 1]
    @test binarystep([100, 100, 100]) == [1, 1, 1]
    @test binarystep([-1000, 100, 1, 0, -0.001, -3]) == [0, 1, 1, 1, 0, 0]
end

@testset "Sigmoid Activation" begin
    @test σ(1) == 0.7310585786300049
    @test σ(0) == 0.5
    @test σ(-1) == 0.2689414213699951
    @test σ(100) == 1
    @test σ(-100) == 3.720075976020836e-44

    @test σ([1]) == [0.7310585786300049]
    @test σ([1, 0]) == [0.7310585786300049, 0.5]
    @test σ([1, 0, -1]) == [0.7310585786300049, 0.5, 0.2689414213699951]
    @test σ([1, 0, -1, 100]) == [0.7310585786300049, 0.5, 0.2689414213699951, 1]
    @test σ([1, 0, -1, 100, -100]) == [0.7310585786300049, 0.5, 
        0.2689414213699951, 1, 3.720075976020836e-44]
end

@testset "tanh Activation" begin
    @test tanh([1]) == [0.7615941559557649]
    @test tanh([1, 0]) == [0.7615941559557649, 0.0]
    @test tanh([1, 0, -1]) == [0.7615941559557649, 0.0, -0.7615941559557649]
    @test tanh([1, 0, -1, 100]) == [0.7615941559557649, 0.0, -0.7615941559557649, 1.0]
    @test tanh([1, 0, -1, 100, -100]) == [0.7615941559557649, 0.0, -0.7615941559557649, 1.0, -1.0]
end

@testset "ReLU Activation" begin
    @test relu(1) == 1
    @test relu(0) == 0
    @test relu(-1) == 0
    @test relu([1, 0, -1]) == [1, 0, 0]
end

@testset "Leaky ReLU Activation" begin
    @test leakyrelu(-1) == -0.01
    @test leakyrelu(0) == 0
    @test leakyrelu(1) == 1
    @test leakyrelu([-1, 0, 1]) == [-0.01, 0, 1]
end

@testset "swish Activation" begin
    @test swish(1) == 0.7310585786300049
    @test swish(0) == 0
    @test swish(-1) == -0.2689414213699951
    @test swish(5) == 4.966535745378576
    @test swish(-5) == -0.03346425462142428
    @test swish([1, 0, -1]) == [0.7310585786300049, 0, -0.2689414213699951]
end

@testset "softmax Activation" begin
    @test softmax(1) == 2.718281828459045
    @test softmax(-1) == -0.36787944117144233
    @test softmax([1, -1]) == [2.718281828459045, -0.36787944117144233]
end

@testset "softplus Activation" begin
    @test softplus(-1) == 0.31326168751822286
    @test softplus(1) == 1.3132616875182228
    @test softplus(0) == 0.6931471805599453
    @test softplus(5) == 5.006715348489118
    @test softplus(-5) == 0.006715348489118068
    @test softplus([-1, 1, 0]) == [0.31326168751822286, 1.3132616875182228, 0.6931471805599453]
end

@testset "GeLU Activation" begin
    @test gelu(-1) == -0.15880800939172324
    @test gelu(0) == 0
    @test gelu(1) == 0.8411919906082768
    @test gelu(5) == 4.999999770820381
    @test gelu(-5) == -2.2917961972623857e-7
    @test gelu([-1, 0, 1]) == [-0.15880800939172324, 0, 0.8411919906082768]
end

@testset "Gaussian Activation" begin
    @test gaussian(1) == 0.11443511435028261
    @test gaussian(-1) == 0.00010675115367571714
    @test gaussian(0) == 0.19947114020071635
    @test gaussian(-5) == 0
    @test gaussian(5) == 1.2471245010500615e-6
    @test gaussian([1, -1, 0]) == [0.11443511435028261, 0.00010675115367571714, 
        0.19947114020071635]
end

@testset "hardtanh Activation" begin
    @test hardtanh(-2) == -1
    @test hardtanh(0) == 0
    @test hardtanh(2) == 1
    @test hardtanh([-1, 0, 1]) == [-1, 0, 1]
end

@testset "ELiSH Activation" begin
    @test elish(-5) == -0.006647754849484245
    @test elish(-1) == -0.1700034015685479
    @test elish(-0.5) == -0.14855067788365744
    @test elish(0.5) == 0.3112296656009273
    @test elish(1) == 0.7310585786300049
    @test elish(5) == 4.966535745378576
    @test elish([-5, -1, -0.5]) == [-0.006647754849484245, -0.1700034015685479, 
        -0.14855067788365744]
end
