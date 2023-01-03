using CausalELM.CrossValidation: recode, traintest, validate, crossvalidate, bestsize
using CausalELM.Metrics: mse, accuracy
using Test

xtrain, ytrain, xtest, ytest = traintest(zeros(20, 2), zeros(20), 5)
xtrain_ts, ytrain_ts, xtest_ts, ytest_ts = traintest(float.(hcat([1:10;], 11:20)), 
    [1.0:1.0:10.0;], 5, 1)

xtrain_ts2, ytrain_ts2, xtest_ts2, ytest_ts2 = traintest(float.(hcat([1:10;], 11:20)), 
    [1.0:1.0:10.0;], 5, 4)

@testset "Recode" begin
    @test recode([-0.7, 0.2, 1.1]) == [1, 2, 3]
    @test recode([0.1, 0.2, 0.3]) == [1, 1, 1]
    @test recode([1.1, 1.51, 1.8]) == [1, 2, 2]
end

@testset "Train-test split" begin
    @test_throws AssertionError traintest(zeros(5, 2), zeros(5), 6)
    @test size(xtrain, 1) == 16
    @test length(ytrain) == 16
    @test size(xtest, 1) == 4
    @test size(ytest, 1) == 4
    @test isa(xtrain, Array)

    # Time series or panel data
    @test_throws AssertionError traintest(zeros(5, 2), zeros(5), 6, 1)
    @test_throws AssertionError traintest(zeros(5, 2), zeros(5), 5, 5)
    @test isa(xtrain_ts, Array)
    @test xtrain_ts == [1.0 11.0; 2.0 12.0]
    @test ytrain_ts == [1.0, 2.0]
    @test xtest_ts == [3. 13.; 4. 14.; 5. 15.; 6. 16.; 7. 17.; 8. 18.; 9. 19.; 10. 20.]
    @test ytest_ts == [3., 4., 5., 6., 7., 8., 9., 10.]
    @test xtrain_ts2 == [1. 11.; 2. 12.; 3. 13.; 4. 14.; 5. 15.; 6. 16.; 7. 17.; 8. 18.]
    @test ytrain_ts2 == [1., 2., 3., 4., 5., 6., 7., 8.]
    @test xtest_ts2 == [9.0 19.0; 10.0 20.0]
    @test ytest_ts2 == [9.0, 10.0]
end

@testset "Single cross validation iteration" begin

    # Regression: Not TS L2, TS L2
    @test isa(validate(rand(100, 5), rand(100), 5, mse), Float64)
    @test isa(validate(rand(100, 5), rand(100), 5, mse, 3), Float64)
    @test isa(validate(rand(100, 5), rand(100), 5, mse, regularized=false), Float64)
    @test isa(validate(rand(100, 5), rand(100), 5, mse, 3, regularized=false), Float64)

    # Classification: Not TS L2, TS L2
    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy), Float64)
    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy, 3), Float64)
    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy, 
        regularized=false), Float64)

    @test isa(validate(rand(100, 5), Float64.(rand(100) .> 0.5), 5, accuracy, 3, 
        regularized=false), Float64)
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
