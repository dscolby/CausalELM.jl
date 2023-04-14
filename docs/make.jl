using CausalELM
using Documenter
using DocThemeIndigo

DocMeta.setdocmeta!(CausalELM, :DocTestSetup, :(using CausalELM); recursive=true)
indigo = DocThemeIndigo.install(CausalELM)

makedocs(;
    modules=[CausalELM],
    assets=String[indigo],
    authors="Darren Colby <dscolby17@gmail.com> and contributors",
    repo="https://github.com/dscolby/CausalELM.jl/blob/{commit}{path}#{line}",
    sitename="CausalELM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dscolby.github.io/CausalELM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "CausalELM" => "index.md",
        "Examples" => "examples.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/dscolby/CausalELM.jl",
    devbranch="main",
)
