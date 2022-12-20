using CausalELM.Metrics: mse, mae

@testset "Mean squared error" begin
    @test mse([0, 0, 0], [0, 0, 0]) == 0
    @test mse([-1, -1, -1], [1, 1, 1]) == 4
    @test mse([1, 1, 1], [2, 2, 2]) == 1
end

@testset "Mean absolute error" begin
    @test mae([0, 0, 0], [0, 0, 0]) == 0
    @test mae([-1, -1, -1], [1, 1, 1]) == 2
    @test mae([1, 1, 1], [2, 2, 2]) == 1
end
