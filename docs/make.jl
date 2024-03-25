using CausalELM
using Documenter

DocMeta.setdocmeta!(CausalELM, :DocTestSetup, :(using CausalELM); recursive=true)

makedocs(;
    modules=[CausalELM],
    warnonly=true,
    authors="Darren Colby <dscolby17@gmail.com> and contributors",
    repo="https://github.com/dscolby/CausalELM.jl/blob/{commit}{path}#{line}",
    sitename="CausalELM",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dscolby.github.io/CausalELM.jl",
        edit_link="main",
        sidebar_sitename=false,
        footer = "© 2024 Darren Colby",
        assets=[],
    ),
    pages=[
        "CausalELM" => "index.md",
        "Getting Started" => [
            "Deciding Which Estimator to Use" => "guide/estimatorselection.md",
            "Interrupted Time Series Estimation" => "guide/its.md",
            "G-computation" => "guide/gcomputation.md",
            "Double Machine Learning" => "guide/doublemachinelearning.md",
            "Metalearners" => "guide/metalearners.md",
            "Doubly Robust Estimation" => "guide/doublyrobust.md"
        ],
        "API" => "api.md",
        "Contributing" => "contributing.md",
        "Release Notes" => "release_notes.md"
    ],
)

deploydocs(;
    repo="github.com/dscolby/CausalELM.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#.#"]
)
