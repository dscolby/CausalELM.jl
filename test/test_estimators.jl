using CausalELM.Estimators: EventStudy
using Test

event_study = EventStudy(rand(5, 100), rand(100), rand(5, 10), rand(10))

@testset "Event Study" begin
    @test event_study.X₀ !== Nothing
    @test event_study.Y₀ !== Nothing
    @test event_study.X₁ !== Nothing
    @test event_study.Y₁ !== Nothing
end
