push!(LOAD_PATH,"../src/")
using CausalELM
using Documenter
makedocs(
         sitename = "CausalELM.jl",
         modules  = [CausalELM],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/dscolby/CausalELM.jl",
)