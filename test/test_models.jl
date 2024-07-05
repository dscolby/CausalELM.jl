using Test
using CausalELM

include("../src/models.jl")

# Test classification functionality using a simple XOR test borrowed from 
# ExtremeLearning.jl
x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
y = [0.0, 1.0, 0.0, 1.0]
x_test = [1.0 1.0; 0.0 1.0; 0.0 0.0]

big_x, big_y = rand(10000, 7), rand(10000)

x1 = rand(20, 5)
y1 = rand(20)
x1test = rand(30, 5)

mock_model = ExtremeLearner(x, y, 10, σ)

m1 = ExtremeLearner(x, y, 10, σ)
f1 = fit!(m1)
predictions1 = predict(m1, x_test)
predict_counterfactual!(m1, x_test)
placebo1 = placebo_test(m1)

m3 = ExtremeLearner(x1, y1, 10, σ)
fit!(m3)
predictions3 = predict(m3, x1test)

m4 = ExtremeLearner(rand(100, 5), rand(100), 5, relu)
fit!(m4)

nofit = ExtremeLearner(x1, y1, 10, σ)
set_weights_biases(nofit)

ensemble = ELMEnsemble(big_x, big_y, 10000, 100, 5, 10, relu)
fit!(ensemble)
predictions = predict(ensemble, big_x)

@testset "Extreme Learning Machines" begin
    @testset "Extreme Learning Machine Structure" begin
        @test mock_model.X isa Array{Float64}
        @test mock_model.Y isa Array{Float64}
        @test mock_model.training_samples == size(x, 1)
        @test mock_model.hidden_neurons == 10
        @test mock_model.activation == σ
        @test mock_model.__fit == false
    end

    @testset "Model Fit" begin
        @test length(m1.β) == 10
        @test size(m1.weights) == (2, 10)
        @test length(m4.β) == size(m4.X, 2)
    end

    @testset "Model Predictions" begin
        @test predictions1[1] < 0.1
        @test predictions1[2] > 0.9
        @test predictions1[3] < 0.1

        # Ensure the counterfactual attribute gets step
        @test m1.counterfactual == predictions1

        # Ensure we can predict with a test set with more data points than the training set
        @test isa(predictions3, Array{Float64})
    end

    @testset "Placebo Test" begin
        @test length(placebo1) == 2
    end

    @testset "Predict Before Fit" begin
        @test isdefined(nofit, :H) == true
        @test_throws ErrorException predict(nofit, x1test)
        @test_throws ErrorException placebo_test(nofit)
    end

    @testset "Print Models" begin
        msg1, msg2 = "Extreme Learning Machine with ", "hidden neurons"
        msg3 = "Regularized " * msg1
        @test sprint(print, m1) === msg1 * string(m1.hidden_neurons) * " " * msg2
    end
end

@testset "Extreme Learning Machine Ensembles" begin
    @testset "Initializing Ensembles" begin
        @test ensemble isa ELMEnsemble
        @test ensemble.X isa Array{Float64}
        @test ensemble.Y isa Array{Float64}
        @test ensemble.elms isa Array{ExtremeLearner}
        @test length(ensemble.elms) == 100
        @test ensemble.feat_indices isa Vector{Vector{Int64}}
        @test length(ensemble.feat_indices) == 100
    end
    
    @testset "Ensemble Fitting and Prediction" begin
        @test all([elm.__fit for elm in ensemble.elms]) == true
        @test predictions isa Vector{Float64}
        @test length(predictions) == 10000
    end

    @testset "Print Models" begin
        msg1, msg2 = "Extreme Learning Machine Ensemble with ", "learners"
        msg3 = "Regularized " * msg1
        @test sprint(print, ensemble) === msg1 * string(length(ensemble.elms)) * " " * msg2
    end
end
