using Documenter, CausalELM

makedocs(
         sitename = "CausalELM",
         author = "Darren Colby",
         modules = [CausalELM.ActivationFunctions, CausalELM.Models],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/dscolby/CausalELM.jl"
)