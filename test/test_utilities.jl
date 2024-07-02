using Test
using CausalELM

# Variables for checking the output of the model_config macro because it is difficult
model_config_avg_expr = @macroexpand CausalELM.@model_config average_effect
model_config_ind_expr = @macroexpand CausalELM.@model_config individual_effect
model_config_avg_idx = Int64.(collect(range(2, 18, 9)))
model_config_ind_idx = Int64.(collect(range(2, 18, 9)))
model_config_avg_ground_truth = quote
    quantity_of_interest::String
    temporal::Bool
    task::String
    activation::Function
    sample_size::Integer
    num_machines::Integer
    num_feats::Integer
    num_neurons::Int64
    causal_effect::Float64
end

model_config_ind_ground_truth = quote
    quantity_of_interest::String
    temporal::Bool
    task::String
    regularized::Bool
    activation::Function
    sample_size::Integer
    num_machines::Integer
    num_feats::Integer
    num_neurons::Int64
    causal_effect::Array{Float64}
end

# Fields for the user supplied data
standard_input_expr = @macroexpand CausalELM.@standard_input_data
standard_input_idx = [2, 4, 6]
standard_input_ground_truth = quote
    X::Array{Float64}
    T::Array{Float64}
    Y::Array{Float64}
end

# Fields for the user supplied data
double_model_input_expr = @macroexpand CausalELM.@standard_input_data
double_model_input_idx = [2, 4, 6]
double_model_input_ground_truth = quote
    X::Array{Float64}
    T::Array{Float64}
    Y::Array{Float64}
    W::Array{Float64}
end

# Generating folds
big_x, big_t, big_y = rand(10000, 8), rand(0:1, 10000), vec(rand(1:100, 10000, 1))
dm = DoubleMachineLearning(big_x, big_t, big_y)
estimate_causal_effect!(dm)
x_fold, t_fold, y_fold = CausalELM.generate_folds(dm.X, dm.T, dm.Y, dm.folds)

@testset "Moments" begin
    @test mean([1, 2, 3]) == 2
    @test CausalELM.var([1, 2, 3]) == 1
end

@testset "One Hot Encoding" begin
    @test CausalELM.one_hot_encode([1, 2, 3]) == [1 0 0; 0 1 0; 0 0 1]
end

@testset "Clipping" begin
    @test CausalELM.clip_if_binary([1.2, -0.02], CausalELM.Binary()) == [1.0, 0.0]
    @test CausalELM.clip_if_binary([1.2, -0.02], CausalELM.Count()) == [1.2, -0.02]
end

@testset "Generating Fields with Macros" begin
    @test model_config_avg_ground_truth.head == model_config_avg_expr.head
    @test model_config_ind_ground_truth.head == model_config_ind_expr.head

    # We only look at even indices because the odd indices have information about what linear
    # of VSCode each variable was defined in, which will differ in both expressions
    @test (
        model_config_avg_ground_truth.args[model_config_avg_idx] ==
        model_config_avg_ground_truth.args[model_config_avg_idx]
    )

    @test (
        model_config_ind_ground_truth.args[model_config_avg_idx] ==
        model_config_ind_ground_truth.args[model_config_avg_idx]
    )

    @test_throws ArgumentError @macroexpand CausalELM.@model_config mean

    @test standard_input_expr.head == standard_input_ground_truth.head

    @test (
        standard_input_expr.args[standard_input_idx] ==
        standard_input_ground_truth.args[standard_input_idx]
    )

    @test double_model_input_expr.head == double_model_input_ground_truth.head

    @test (
        double_model_input_expr.args[double_model_input_idx] ==
        double_model_input_ground_truth.args[double_model_input_idx]
    )
end

@testset "Generating Folds" begin
    @test size(x_fold[1], 2) == size(dm.X, 2)
    @test y_fold isa Vector{Vector{Float64}}
    @test t_fold isa Vector{Vector{Float64}}
    @test length(t_fold) == dm.folds
end
