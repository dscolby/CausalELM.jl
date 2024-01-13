using CausalELM
using Documenter

makedocs(;
    modules=[CausalELM],
    warnonly=true,
    authors="Darren Colby <dscolby17@gmail.com> and contributors",
    repo="https://github.com/dscolby/CausalELM.jl/blob/{commit}{path}#{line}",
    sitename="",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dscolby.github.io/CausalELM.jl",
        edit_link="main",
        footer = "© 2024 Darren Colby",
        assets=[],
    ),
    pages=[
        "causalELM" => "index.md",
        "User Guide" => Any[
            "Deciding Which Model to Use" => "guide/estimatorselection.md",
            "Interrupted Time Series Estimation" => "guide/its.md",
            "G-computation" => "guide/gcomputation.md",
            "Double Machine Learning" => "guide/doublemachinelearning.md",
            "Metalearners" => "guide/metalearners.md"
        ],
        "API" => "reference/api.md",
        "Contributing" => "contributing.md"
    ],
)

deploydocs(;
    repo="github.com/dscolby/CausalELM.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#.#"]
)
