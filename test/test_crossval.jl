using Test
using CausalELM

using CausalELM: relu
include("../src/crossval.jl")

x, y = shuffle_data(rand(100, 5), rand(100))
xfolds, yfolds = generate_folds(zeros(20, 2), zeros(20), 5)
xfolds_ts, yfolds_ts = generate_temporal_folds(float.(hcat([1:10;], 11:20)), 
    [1.0:1.0:10.0;], 5)

@testset "Fold Generation" begin
    @test_throws ArgumentError generate_folds(zeros(5, 2), zeros(5), 6)
    @test_throws ArgumentError generate_folds(zeros(5, 2), zeros(5), 5)
    @test size(xfolds, 1) == 5
    @test size(xfolds[1], 1) == 4
    @test size(xfolds[2], 2) == 2
    @test length(yfolds) == 5
    @test size(yfolds[1], 1) == 4
    @test size(yfolds[2], 2) == 1
    @test isa(xfolds, Array)
    @test isa(yfolds, Array)

    # Time series or panel data
    # Testing incorrect input
    @test_throws ArgumentError generate_temporal_folds(zeros(5, 2), zeros(5), 6)
    @test_throws ArgumentError generate_temporal_folds(zeros(5, 2), zeros(5), 5)
    @test_throws ArgumentError generate_temporal_folds(zeros(10, 2), zeros(5), 6)
    @test_throws ArgumentError generate_temporal_folds(zeros(10, 2), zeros(5), 5)

    @test size(xfolds_ts, 1) == 5
    @test size(xfolds_ts[1], 1) == 2
    @test size(xfolds_ts[2], 2) == 2
    @test length(yfolds_ts) == 5
    @test size(yfolds_ts[1], 1) == 2
    @test size(yfolds_ts[2], 2) == 1
    @test isa(xfolds_ts, Array)
    @test isa(yfolds_ts, Array)
end

@testset "Single cross validation iteration" begin

    # Regression: Not TS L2, TS L2
    @test isa(validation_loss(rand(100, 5), rand(100), rand(20, 5), rand(20), 5, mse), Float64)
    @test isa(validation_loss(rand(100, 5), rand(100), rand(20, 5), rand(20), 5, mse), Float64)
    @test isa(validation_loss(rand(100, 5), rand(100), rand(20, 5), rand(20), 5, mse, 
        regularized=false), Float64)
    @test isa(validation_loss(rand(100, 5), rand(100), rand(20, 5), rand(20), 5,  mse, 
        regularized=false, activation=gelu), Float64)

    # Classification: Not TS L2, TS L2
    @test isa(validation_loss(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy), Float64)
    @test isa(validation_loss(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy), Float64)
    @test isa(validation_loss(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy, regularized=false, activation=gelu), 
        Float64)

    @test isa(validation_loss(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy, regularized=false), Float64)
end

@testset "Cross validation" begin

    # Regression
    @test isa(cross_validate(rand(100, 5), rand(100), 5, mse, relu, true, 5, false), 
        Float64)
    @test isa(cross_validate(rand(100, 5), rand(100), 5, mse, relu, false, 5, true), 
        Float64)

    # Classification
    @test isa(cross_validate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy, 
        relu, true, 5, false), Float64)
    @test isa(cross_validate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy, 
        relu, false, 5, true), Float64)
end

@testset "Best network size" begin
    @test 100>= best_size(rand(100, 7), Float64.(rand(100) .> 0.5), accuracy, 
        "classification", relu, 1, 10, true, 5, false, 10, 2) >= 1

    @test 100 >= best_size(rand(100, 5), rand(100), mse, "regression", relu, 1, 10, false, 
        5, true, 10, 2) >= 1
end

@testset "Data Shuffling" begin
    @test size(x) === (100, 5)
    @test x isa Array{Float64}
    @test size(y, 1) === 100
    @test y isa Vector{Float64}
end
