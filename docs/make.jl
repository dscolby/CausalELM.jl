using CausalELM
using Documenter

DocMeta.setdocmeta!(CausalELM, :DocTestSetup, :(using CausalELM); recursive=true)

makedocs(;
    modules=[CausalELM],
    authors="Darren Colby <dscolby17@gmail.com> and contributors",
    repo="https://github.com/dscolby/CausalELM.jl/blob/{commit}{path}#{line}",
    sitename="CausalELM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dscolby.github.io/CausalELM.jl",
        edit_link="main",
        assets=[],
    ),
    pages=[
        "CausalELM" => "index.md",
        "User Guide" => Any[
            "Interrupted Time Series Estimation" => "guide/its.md",
            "G-computation" => "guide/gcomputation.md",
            "Double Machine Learning" => "guide/doublemachinelearning.md",
            "Metalearners" => "guide/metalearners.md"
        ],
        "API Reference" => Any[
            "CausalELM" => "reference/api.md",
            "Activation Functions" => "reference/activations.md",
            "Cross Validation" => "reference/crossval.md",
            "ATE/ATT/ITE Estimation" => "reference/estimation.md",
            "CATE Estimation" => "reference/metalearners.md",
            "Inference and Summarization" => "reference/inference.md",
            "Model Validation" => "reference/validation.md",
            "Validation Metrics" => "reference/metrics.md",
            "Base Models" => "reference/base.md"
        ],
        "Contributing" => "contributing.md"
    ],
)

deploydocs(;
    repo="github.com/dscolby/CausalELM.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#.#"]
)
