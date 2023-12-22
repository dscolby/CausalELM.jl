using Test
using CausalELM

length4, length5  = rand(4), rand(5)

@testset "Mean squared error" begin
    @test mse([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0
    @test mse([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]) == 4
    @test mse([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 1
end

@testset "Mean absolute error" begin
    @test mae([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0
    @test mae([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]) == 2
    @test mae([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 1
end

@testset "Confusion Matrix" begin
    @test CausalELM.confusion_matrix([1, 1, 1, 1, 0], [1, 1, 1, 1, 0]) == [1 0; 0 4]
    @test CausalELM.confusion_matrix([1, 0, 1, 0], [0, 1, 0, 1]) == [0 2; 2 0]
    @test CausalELM.confusion_matrix([1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 0, 2]) == [1 0 0; 
                                                                                 0 4 0; 
                                                                                 0 0 1]
end

@testset "Accuracy" begin
    @test accuracy([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]) == 0
    @test accuracy([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]) == 1
    @test accuracy([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]) == 0.5
    @test accuracy([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]) == 0.25
end

@testset "Precision" begin
    @test precision([0, 1, 0, 0], [0, 1, 1, 0]) == 0.5
    @test precision([0, 1, 0, 0], [0, 1, 0, 0]) == 1
    @test precision([1, 2, 1, 3, 0], [2, 2, 2, 3, 1]) ≈ 0.333333333
    @test precision([1, 2, 1, 3, 2], [2, 2, 2, 3, 1]) ≈ 0.444444444
end

@testset "Recall" begin
    @test recall([0, 1, 0, 0], [0, 1, 1, 0]) == 1
    @test recall([0, 1, 0, 0], [0, 1, 0, 0]) == 1
    @test recall([1, 2, 1, 3, 0], [2, 2, 2, 3, 1]) == 0.5
    @test recall([1, 2, 1, 3, 2], [2, 2, 2, 3, 1]) == 0.5
end

@testset "F1 Score" begin
    @test F1([0, 1, 0, 0], [0, 1, 1, 0]) ≈ 0.6666666666
    @test F1([0, 1, 0, 0], [0, 1, 0, 0]) == 1
    @test F1([1, 2, 1, 3, 0], [2, 2, 2, 3, 1]) == 0.4
    @test F1([1, 2, 1, 3, 2], [2, 2, 2, 3, 1]) == 0.47058823529411764
end

@testset "Dimension Mismatch" begin
    @test_throws DimensionMismatch mse(length4, length5)
    @test_throws DimensionMismatch mae(length4, length5)
    @test_throws DimensionMismatch accuracy(length4, length5)
end
