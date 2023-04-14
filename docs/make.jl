using CausalELM
using Documenter
using DocumenterTools: Themes

Themes.compile(joinpath(@__DIR__,"src/assets/dark.scss"), joinpath(@__DIR__,"src/assets/themes/documenter-dark.css"))
DocMeta.setdocmeta!(CausalELM, :DocTestSetup, :(using CausalELM); recursive=true)

makedocs(;
    modules=[CausalELM],
    assets=["assets/favicon.ico", asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css)],
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
