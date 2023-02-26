using CausalELM.Inference: generatenulldistribution
using CausalELM.Estimators: CausalEstimator
using Test

x, y, t = rand(100, 5), rand(1:100, 100, 1), [rand()<0.4 for i in 1:100]

g_computer = GComputation(x, y, t)
estimatecausaleffect!(g_computer)
g_inference = generatenulldistribution(g_computer)

@testset "Generating Null Distributions" begin
    @test size(g_inference, 1) === 1000
    @test g_inference isa Array{Float64}
end
