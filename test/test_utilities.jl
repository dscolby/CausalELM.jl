using Test

include("../src/utilities.jl")

struct Binary end
struct Count end

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

@testset "Clipping" begin
    @test clip_if_binary([1.2, -0.02], Binary()) == [0.9999999, 1.0e-7]
    @test clip_if_binary([1.2, -0.02], Count()) == [1.2, -0.02]
end