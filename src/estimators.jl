module Estimators

using CausalELM

mutable struct EventStudy
    X₀::Array{Float64}
    Y₀::Array{Float64}
    X₁::Array{Float64}
    Y₁::Array{Float64}
    task::String
    regularized::Bool
    activation::Function
    validation_metric::Function
    min_neurons::Int64
    max_neurons::Int64
    folds::Int64
    iterations::Int64
    approximator_neurons::Int64
    num_neurons::Int64
    learner::ExtremeLearningMachine
    β::Array{Float64}
    Ŷ::Array{Float64}
    abnormal_returns::Array{Float64}
    performance::Float64
    placebo_test::Tuple{Vector{Float64}, Vector{Float64}}
    function EventStudy(X₀, Y₀, X₁, Y₁; task="regression", regularized=true, 
        activation=relu, validation_metric=mse, min_neurons=1, max_neurons=100, folds=5, 
        iterations=Int(round(size(X₀, 1)/10)), 
        approximator_neurons=Int(round(size(X₀, 1)/10)))

        m1 = "Task must be one of regression or classification"
        @assert task ∈ ("regression", "classification") m1

        new(X₀, Y₀, X₁, Y₁, task, regularized, ctivation, quantity, validation_metric, 
            min_neurons, max_neurons, folds, iterations, approximator_neurons)
    end
end

function estimatetreatmenteffect!(study::EventStudy)
    study.num_neurons = bestsize(study.X₀, study.Y₀, study.validation_metric, study.task, 
        study.activation, study.min_neurons, study.max_neurons, study.regularized, 
        study.folds, true, study.iterations, study.approximator_neurons)

    if regularized
        study.learner = RegularizedExtremeLearner(study.X₀, study.Y₀, study.num_neurons, 
            study.activation)
    else
        study.learner = ExtremeLearner(study.X₀, study.Y₀, study.num_neurons, 
            study.activation)
    end

    study.β = fit!(study.learner)
    study.Ŷ = predict(study.learner, X₁)
    study.abnormal_returns = study.Ŷ - study.Y₁
    study.performance = study.validation_metric(study.Y₁, study.Ŷ)
    study.placebo_test = placebotest(study.learner)
end

function summary(event_study::EventStudy)
    println("Event Study Design\n")
    for fname in fieldnames(event_study)[[5, 6, 7, 8, 12, 13, 14, 15, 16]]
        println("$fname: ???\n")
    end
end

Base.show(io::IO, event_study::EventStudy) = println("Event study design")

end