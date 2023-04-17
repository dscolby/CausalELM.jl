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
            "Event Study Estimation" => "guide/event_study.md",
            "G-computation" => "guide/gcomputation.md",
            "Doubly Robust Estimation" => "guide/doublyrobust.md",
            "Metalearners" => "guide/metalearners.md"
        ],
        "Reference" => Any[
            "CausalELM" => "reference/api.md",
            "Activation Functions" => "reference/activations.md",
            "Cross Validation" => "reference/crossval.md",
            "ATE/ATT/ITT Estimation" => "reference/average.md",
            "CATE Estimation" => "reference/cate.md",
            "Inference and Summarization" => "reference/inference.md",
            "Validation Metrics" => "reference/metrics.md",
            "Base Models" => "reference/base.md"
        ],
        "Contributing" => Any[
            "contributing/bug.md",
            "contributing/features.md", 
            "contributing/code.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/dscolby/CausalELM.jl",
    versions = [
        "stable" => "v0.2.0", 
        "v0.1", 
        devurl => "https://dscolby.github.io/CausalELM.jl/dev"],
    devbranch="main",
)
