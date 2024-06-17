using Test
using CausalELM

@testset "Binary Step Activation" begin

    # Single numbers
    @test binary_step(-1000.0) == 0
    @test binary_step(-0.00001) == 0
    @test binary_step(0.0) == 1
    @test binary_step(0.00001) == 1
    @test binary_step(100.0) == 1

    # Vectors
    @test binary_step([-100.0]) == [0]
    @test binary_step([-100.0, -100.0, -100.0]) == [0, 0, 0]
    @test binary_step([-0.00001, -0.00001, -0.00001]) == [0, 0, 0]
    @test binary_step([0.0, 0.0, 0.0]) == [1, 1, 1]
    @test binary_step([0.00001, 0.00001, 0.00001]) == [1, 1, 1]
    @test binary_step([100.0, 100.0, 100.0]) == [1, 1, 1]
    @test binary_step([-1000.0, 100.0, 1.0, 0.0, -0.001, -3]) == [0, 1, 1, 1, 0, 0]
end

@testset "Sigmoid Activation" begin
    @test σ(1.0) == 0.7310585786300049
    @test σ(0.0) == 0.5
    @test σ(-1.0) == 0.2689414213699951
    @test σ(100.0) == 1
    @test σ(-100.0) == 3.720075976020836e-44

    @test σ([1.0]) == [0.7310585786300049]
    @test σ([1.0, 0.0]) == [0.7310585786300049, 0.5]
    @test σ([1.0, 0.0, -1.0]) == [0.7310585786300049, 0.5, 0.2689414213699951]
    @test σ([1.0, 0.0, -1.0, 100.0]) == [0.7310585786300049, 0.5, 0.2689414213699951, 1]
    @test σ([1.0, 0.0, -1.0, 100.0, -100.0]) ==
        [0.7310585786300049, 0.5, 0.2689414213699951, 1, 3.720075976020836e-44]
end

@testset "tanh Activation" begin
    @test CausalELM.tanh([1.0]) == [0.7615941559557649]
    @test CausalELM.tanh([1.0, 0.0]) == [0.7615941559557649, 0.0]
    @test CausalELM.tanh([1.0, 0.0, -1.0]) == [0.7615941559557649, 0.0, -0.7615941559557649]
    @test CausalELM.tanh([1.0, 0.0, -1.0, 100.0]) ==
        [0.7615941559557649, 0.0, -0.7615941559557649, 1.0]
    @test CausalELM.tanh([1.0, 0.0, -1.0, 100.0, -100.0]) ==
        [0.7615941559557649, 0.0, -0.7615941559557649, 1.0, -1.0]
end

@testset "ReLU Activation" begin
    @test relu(1.0) == 1
    @test relu(0.0) == 0
    @test relu(-1.0) == 0
    @test relu([1.0, 0.0, -1.0]) == [1, 0, 0]
end

@testset "Leaky ReLU Activation" begin
    @test leaky_relu(-1.0) == -0.01
    @test leaky_relu(0.0) == 0
    @test leaky_relu(1.0) == 1
    @test leaky_relu([-1.0, 0.0, 1.0]) == [-0.01, 0, 1]
end

@testset "swish Activation" begin
    @test swish(1.0) == 0.7310585786300049
    @test swish(0.0) == 0
    @test swish(-1.0) == -0.2689414213699951
    @test swish(5.0) == 4.966535745378576
    @test swish(-5.0) == -0.03346425462142428
    @test swish([1.0, 0.0, -1.0]) == [0.7310585786300049, 0, -0.2689414213699951]
end

@testset "softmax Activation" begin
    @test softmax(1.0) == 1.0
    @test softmax(-1.0) == 1.0
    @test sum(softmax([1.0, -0.5])) == 1.0
    @test sum(softmax([1.0, 0])) == 1.0
    @test vec(mapslices(sum, softmax(rand(3, 3)); dims=2)) ≈ [1, 1, 1]
end

@testset "softplus Activation" begin
    @test softplus(-1.0) == 0.3132616875182228
    @test softplus(1.0) == 1.3132616875182228
    @test softplus(0.0) == 0.6931471805599453
    @test softplus(5.0) == 5.006715348489118
    @test softplus(-5.0) == 0.006715348489118068
    @test softplus([-1.0, 1.0, 0.0]) ==
        [0.3132616875182228, 1.3132616875182228, 0.6931471805599453]
end

@testset "GeLU Activation" begin
    @test gelu(-1.0) == -0.15880800939172324
    @test gelu(0.0) == 0
    @test gelu(1.0) == 0.8411919906082768
    @test gelu(5.0) == 4.999999770820381
    @test gelu(-5.0) == -2.2917961972623857e-7
    @test gelu([-1.0, 0.0, 1.0]) == [-0.15880800939172324, 0, 0.8411919906082768]
end

@testset "Gaussian Activation" begin
    @test gaussian(1.0) ≈ 0.36787944117144233
    @test gaussian(-1.0) ≈ 0.36787944117144233
    @test gaussian(0.0) ≈ 1.0
    @test gaussian(-5.0) ≈ 1.3887943864964021e-11
    @test gaussian(5.0) ≈ 1.3887943864964021e-11
    @test gaussian([1.0, -1.0, 0.0]) ≈ [0.36787944117144233, 0.36787944117144233, 1.0]
end

@testset "hard_tanh Activation" begin
    @test hard_tanh(-2.0) == -1
    @test hard_tanh(0.0) == 0
    @test hard_tanh(2.0) == 1
    @test hard_tanh([-1.0, 0.0, 1.0]) == [-1, 0, 1]
end

@testset "ELiSH Activation" begin
    @test elish(-5.0) == -0.006647754849484245
    @test elish(-1.0) == -0.17000340156854793
    @test elish(-0.5) == -0.1485506778836575
    @test elish(0.5) == 0.3112296656009273
    @test elish(1.0) == 0.7310585786300049
    @test elish(5.0) == 4.966535745378576
    @test elish([-5.0, -1.0, -0.5]) ==
        [-0.006647754849484245, -0.17000340156854793, -0.1485506778836575]
end

@testset "Fourier Activation" begin
    @test fourier(1.0) ≈ 0.8414709848078965
    @test fourier(0.0) == 0
    @test fourier(-1.0) ≈ -0.8414709848078965
    @test fourier([1.0, 0.0, -1.0]) ≈ [0.8414709848078965, 0, -0.8414709848078965]
end
