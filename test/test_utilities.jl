using Test

include("../src/utilities.jl")

@testset "Moments" begin
    @test mean([1, 2, 3]) == 2
    @test var([1, 2, 3]) == 1
end

@testset "Add and Subtract Consecutive Elements" begin
    @test consecutive([1, 2, 3, 4, 5]) == [1, 1, 1, 1]
end

@testset "One Hot Encoding" begin
    @test one_hot_encode([1, 2, 3]) == [1 0 0; 0 1 0; 0 0 1]
end