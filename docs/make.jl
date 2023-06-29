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
        "Guide" => Any[
            "Interrupted Time Series Estimation" => "guide/its.md",
            "G-computation" => "guide/gcomputation.md",
            "Doubly Robust Estimation" => "guide/doublyrobust.md",
            "Metalearners" => "guide/metalearners.md"
        ],
        "Reference" => Any[
            "CausalELM" => "reference/api.md",
            "Activation Functions" => "reference/activations.md",
            "Cross Validation" => "reference/crossval.md",
            "Casual Effect Estimation" => "reference/estimation.md",
            "Inference and Summarization" => "reference/inference.md",
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
