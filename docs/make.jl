using CausalELM
using Documenter

DocMeta.setdocmeta!(CausalELM, :DocTestSetup, :(using CausalELM); recursive=true)

makedocs(;
         sitename = "CausalELM.jl",
         authors = "Darren Colby",
         modules  = [CausalELM],
         format=Documenter.HTML(;
         prettyurls=get(ENV, "CI", "false") == "true",
         canonical="https://dscolby.github.io/CausalELM.jl/index.html",
         assets=String[],
            ),
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
repo="github.com/dscolby/CausalELM.jl", 
devbranch = "main",
devurl="dev",
target = "build",
branch = "gh-pages",
versions = ["stable" => "v^", "v#.#" ]
)