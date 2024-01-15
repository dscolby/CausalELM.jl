using CausalELM
using Documenter

makedocs(;
    modules=[CausalELM],
    warnonly=true,
    authors="Darren Colby <dscolby17@gmail.com> and contributors",
    repo="https://github.com/dscolby/CausalELM.jl/blob/{commit}{path}#{line}",
    sitename="causalELM",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dscolby.github.io/CausalELM.jl",
        edit_link="main",
        sidebar_sitename=false,
        footer = "© 2024 Darren Colby",
        assets=[],
    ),
    pages=[
        "causalELM" => "index.md",
        "Getting Started" => "getting_started.md",
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
