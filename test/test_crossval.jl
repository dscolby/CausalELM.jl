using CausalELM.CrossValidation: recode, generatefolds, validate, crossvalidate, bestsize
using CausalELM.Metrics: mse, accuracy
using CausalELM.ActivationFunctions: gelu
using Test

xfolds, yfolds = generatefolds(zeros(20, 2), zeros(20), 5)
xfolds_ts, yfolds_ts = generatefolds(float.(hcat([1:10;], 11:20)), [1.0:1.0:10.0;], 5)

@testset "Recode" begin
    @test recode([-0.7, 0.2, 1.1]) == [1, 2, 3]
    @test recode([0.1, 0.2, 0.3]) == [1, 1, 1]
    @test recode([1.1, 1.51, 1.8]) == [1, 2, 2]
end

@testset "Fold Generation" begin
    @test_throws ArgumentError generatefolds(zeros(5, 2), zeros(5), 6)
    @test_throws ArgumentError generatefolds(zeros(5, 2), zeros(5), 5)
    @test size(xfolds, 1) == 5
    @test size(xfolds[1], 1) == 4
    @test size(xfolds[2], 2) == 2
    @test length(yfolds) == 5
    @test size(yfolds[1], 1) == 4
    @test size(yfolds[2], 2) == 1
    @test isa(xfolds, Array)
    @test isa(yfolds, Array)

    # Time series or panel data
    @test_throws ArgumentError generatefolds(zeros(5, 2), zeros(5), 6)
    @test_throws ArgumentError generatefolds(zeros(5, 2), zeros(5), 5)
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
    @test isa(validate(rand(100, 5), rand(100), rand(20, 5), rand(20), 5, mse), Float64)
    @test isa(validate(rand(100, 5), rand(100), rand(20, 5), rand(20), 5, mse), Float64)
    @test isa(validate(rand(100, 5), rand(100), rand(20, 5), rand(20), 5, mse, 
        regularized=false), Float64)
    @test isa(validate(rand(100, 5), rand(100), rand(20, 5), rand(20), 5,  mse, 
        regularized=false, activation=gelu), Float64)

    # Classification: Not TS L2, TS L2
    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy), Float64)
    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy), Float64)
    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy, regularized=false, activation=gelu), 
        Float64)

    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), rand(20, 5), 
        Float64.(rand(20) .> 0.5), 5, accuracy, regularized=false), Float64)
end

@testset "Cross validation" begin

    # Regression
    @test isa(crossvalidate(rand(100, 5), rand(100), 5, mse), Float64)
    @test isa(crossvalidate(rand(100, 5), rand(100), 5, mse), Float64)

    # Classification
    @test isa(crossvalidate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy), Float64)
    @test isa(crossvalidate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy), Float64)
end

@testset "Best network size" begin
    @test 100>= bestsize(rand(100, 5), Float64.(rand(100) .> 0.5), accuracy, 
        "classification") >= 1

    @test 100 >= bestsize(rand(100, 5), rand(100), mse, "regression") >= 1
end
