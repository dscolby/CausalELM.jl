"""
    validate(its; kwargs...)

Test the validity of an estimated interrupted time series analysis.

# Arguments
- `its::InterruptedTimeSeries`: an interrupted time seiries estimator.

# Keywords
- `n::Int`: number of times to simulate a confounder.
- `low::Float64`=0.15: minimum proportion of data points to include before or after the 
    tested break in the Wald supremum test.
- `high::Float64=0.85`: maximum proportion of data points to include before or after the 
    tested break in the Wald supremum test.

# Notes
This method coducts a Chow Test, a Wald supremeum test, and tests the model's sensitivity to 
confounders. The Chow Test tests for structural breaks in the covariates between the time 
before and after the event. p-values represent the proportion of times the magnitude of the 
break in a covariate would have been greater due to chance. Lower p-values suggest a higher 
probability the event effected the covariates and they cannot provide unbiased 
counterfactual predictions. The Wald supremum test finds the structural break with the 
highest Wald statistic. If this is not the same as the hypothesized break, it could indicate 
an anticipation effect, a confounding event, or that the intervention or policy took place 
in multiple phases. p-values represent the proportion of times we would see a larger Wald 
statistic if the data points were randomly allocated to pre and post-event periods for the 
predicted structural break. Ideally, the hypothesized break will be the same as the 
predicted break and it will also have a low p-value. The omitted predictors test adds 
normal random variables with uniform noise as predictors. If the included covariates are 
good predictors of the counterfactual outcome, adding irrelevant predictors should not have 
a large effect on the predicted counterfactual outcomes or the estimated effect.

This method does not implement the second test in Baicker and Svoronos because the estimator 
in this package models the relationship between covariates and the outcome and uses an 
extreme learning machine instead of linear regression, so variance in the outcome across 
different bins is not much of an issue.

# References
For more details on the assumptions and validity of interrupted time series designs, see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> X₀, Y₀, X₁, Y₁ = rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
julia> validate(m1)
```
"""
function validate(its::InterruptedTimeSeries; n=1000, low=0.15, high=0.85)
    if all(isnan, its.causal_effect)
        throw(ErrorException("call estimate_causal_effect! before calling validate"))
    end

    return covariate_independence(its; n=n),
    sup_wald(its; low=low, high=high, n=n),
    omitted_predictor(its; n=n)
end

"""
    validate(m; kwargs)

# Arguments
- `m::Union{CausalEstimator, Metalearner}`: model to validate/test the assumptions of.
    
# Keywords
- `devs=::Any`: iterable of deviations from which to generate noise to simulate violations 
    of the counterfactual consistency assumption.
- `num_iterations=10::Int: number of times to simulate a violation of the counterfactual 
    consistency assumption.`
- `min::Float64`=1.0e-6: minimum probability of treatment for the positivity assumption.
- `high::Float64=1-min`: maximum probability of treatment for the positivity assumption.

# Notes
This method tests the counterfactual consistency, exchangeability, and positivity 
assumptions required for causal inference. It should be noted that consistency and 
exchangeability are not directly testable, so instead, these tests do not provide definitive 
evidence of a violation of these assumptions. To probe the counterfactual consistency 
assumption, we simulate counterfactual outcomes that are different from the observed 
outcomes, estimate models with the simulated counterfactual outcomes, and take the averages.
If the outcome is continuous, the noise for the simulated counterfactuals is drawn from 
N(0, dev) for each element in devs, otherwise the default is 0.25, 0.5, 0.75, and 1.0 
standard deviations from the mean outcome. For discrete variables, each outcome is replaced 
with a different value in the range of outcomes with probability ϵ for each ϵ in devs, 
otherwise the default is 0.025, 0.05, 0.075, 0.1. If the average estimate for a given level 
of violation differs greatly from the effect estimated on the actual data, then the model is 
very sensitive to violations of the counterfactual consistency assumption for that level of 
violation. Next, this methods tests the model's sensitivity to a violation of the 
exchangeability assumption by calculating the E-value, which is the minimum strength of 
association, on the risk ratio scale, that an unobserved confounder would need to have with 
the treatment and outcome variable to fully explain away the estimated effect. Thus, higher 
E-values imply the model is more robust to a violation of the exchangeability assumption. 
Finally, this method tests the positivity assumption by estimating propensity scores. Rows 
in the matrix are levels of covariates that have a zero probability of treatment. If the 
matrix is empty, none of the observations have an estimated zero probability of treatment, 
which implies the positivity assumption is satisfied.

# References
For a thorough review of casual inference assumptions see:
    Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
    Francis, 2024. 

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

# Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]), vec(rand(1:100, 100, 1)) 
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> validate(g_computer)
```
"""
function validate(m, devs; iterations=10, min=1.0e-6, max=1.0 - min)
    if all(isnan, m.causal_effect)
        throw(ErrorException("call estimate_causal_effect! before calling validate"))
    end

    return counterfactual_consistency(m, devs, iterations),
    exchangeability(m),
    positivity(m, min, max)
end

function validate(m; iterations=10, min=1.0e-6, max=1.0 - min)
    if var_type(m.Y) isa Continuous
        devs = 0.25, 0.5, 0.75, 1.0
    else
        devs = 0.025, 0.05, 0.075, 0.1
    end
    return validate(m, devs; iterations=iterations, min=min, max=max)
end

"""
    covariate_independence(its; kwargs..)

Test for independence between covariates and the event or intervention.

# Arguments
- `its::InterruptedTImeSeries`: an interrupted time series estimator.

# Keywords
- `n::Int`: number of permutations for assigning observations to the pre and 
        post-treatment periods.

This is a Chow Test for covariates with p-values estimated via randomization inference, 
which does not assume a distribution for the outcome variable. The p-values are the 
proportion of times randomly assigning observations to the pre or post-intervention period 
would have a larger estimated effect on the the slope of the covariates. The lower the 
p-values, the more likely it is that the event/intervention effected the covariates and 
they cannot provide an unbiased prediction of the counterfactual outcomes.

For more information on using a Chow Test to test for structural breaks see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
       randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> covariate_independence(its)
```
"""
function covariate_independence(its::InterruptedTimeSeries; n=1000)
    x₀ = reduce(hcat, (its.X₀[:, 1:(end - 1)], zeros(size(its.X₀, 1))))
    x₁ = reduce(hcat, (its.X₁[:, 1:(end - 1)], ones(size(its.X₁, 1))))
    x = reduce(vcat, (x₀, x₁))
    results = Dict{String, Float64}()

    # Estimate a linear regression with each covariate as a dependent variable and all other
    # covariates and time as independent variables
    for i in axes(x, 2)
        new_x, y = x[:, 1:end .!= i], x[:, i]
        β = last(new_x \ y)
        p = p_val(new_x, y, β; n=n)
        results["Column " * string(i) * " p-value"] = p
    end
    return results
end

"""
    omitted_predictor(its; kwargs...)

See how an omitted predictor/variable could change the results of an interrupted time series 
analysis.

# Arguments
- `its::InterruptedTImeSeries`: interrupted time seiries estimator.

# Keywords
- `n::Int`: number of times to simulate a confounder.

# Notes
This method reestimates interrupted time series models with uniform random variables. If the 
included covariates are good predictors of the counterfactual outcome, adding a random 
variable as a covariate should not have a large effect on the predicted counterfactual 
outcomes and therefore the estimated average effect.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> omitted_predictor(its)
```
"""
function omitted_predictor(its::InterruptedTimeSeries; n=1000)
    if all(isnan, its.causal_effect)
        throw(ErrorException("call estimate_causal_effect! before calling omittedvariable"))
    end

    its_copy = deepcopy(its)
    biased_effects = Vector{Float64}(undef, n)
    results = Dict{String,Float64}()

    for i in 1:n
        its_copy.X₀ = reduce(hcat, (its.X₀, rand(size(its.X₀, 1))))
        its_copy.X₁ = reduce(hcat, (its.X₁, rand(size(its.X₁, 1))))
        biased_effects[i] = mean(estimate_causal_effect!(its_copy))
    end

    biased_effects = sort(biased_effects)
    results["Minimum Biased Effect/Original Effect"] = biased_effects[1]
    results["Mean Biased Effect/Original Effect"] = mean(biased_effects)
    results["Maximum Biased Effect/Original Effect"] = biased_effects[n]
    median = ifelse(
        n % 2 === 1,
        biased_effects[Int(ceil(n / 2))],
        mean([biased_effects[Int(n / 2)], biased_effects[Int(n / 2) + 1]]),
    )
    results["Median Biased Effect/Original Effect"] = median

    return results
end

"""
    sup_wald(its; kwargs)

Check if the predicted structural break is the hypothesized structural break.

# Arguments
- `its::InterruptedTimeSeries`: interrupted time seiries estimator.

# Keywords
- `n::Int`: number of times to simulate a confounder.
- `low::Float64`=0.15: minimum proportion of data points to include before or after the 
        tested break in the Wald supremum test.
- `high::Float64=0.85`: maximum proportion of data points to include before or after the 
        tested break in the Wald supremum test.

# Notes
This method conducts Wald tests and identifies the structural break with the highest Wald 
statistic. If this break is not the same as the hypothesized break, it could indicate an 
anticipation effect, confounding by some other event or intervention, or that the 
intervention or policy took place in multiple phases. p-values are estimated using 
approximate randomization inference and represent the proportion of times we would see a 
larger Wald statistic if the data points were randomly allocated to pre and post-event 
periods for the predicted structural break.

# References
For more information on using a Chow Test to test for structural breaks see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.
    
For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

# Examples
```julia
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
       randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> sup_wald(its)
```
"""
function sup_wald(its::InterruptedTimeSeries; low=0.15, high=0.85, n=1000)
    hypothesized_break, current_break, wald = size(its.X₀, 1), size(its.X₀, 1), 0.0
    high_idx, low_idx = Int(floor(high * size(its.X₀, 1))), Int(ceil(low * size(its.X₀, 1)))
    x, y = reduce(vcat, (its.X₀, its.X₁)), reduce(vcat, (its.Y₀, its.Y₁))
    t = reduce(vcat, (zeros(size(its.X₀, 1)), ones(size(its.X₁, 1))))
    best_x = reduce(hcat, (x, t))
    best_β = last(best_x \ y)

    # Set each time as a potential break and calculate its Wald statistic
    for idx in low_idx:high_idx
        t = reduce(vcat, (zeros(idx), ones(size(x, 1) - idx)))
        new_x = reduce(hcat, (x, t))
        β, ŷ = @fastmath new_x \ y, new_x * (new_x \ y)
        se = @fastmath sqrt(1 / (size(x, 1) - 2)) * (sum(y .- ŷ)^2 / sum(t .- mean(t))^2)
        wald_candidate = last(β) / se

        if wald_candidate > wald
            current_break, wald, best_x, best_β = idx, wald_candidate, new_x, best_β
        end
    end
    p = p_val(best_x, y, best_β; n=n, two_sided=true)
    return Dict(
        "Hypothesized Break Point" => hypothesized_break,
        "Predicted Break Point" => current_break,
        "Wald Statistic" => wald,
        "p-value" => p,
    )
end

"""
    p_val(x, y, β; kwargs...)

Estimate the p-value for the hypothesis that an event had a statistically significant effect 
on the slope of a covariate using randomization inference.

# Arguments
- `x::Array{<:Real}`: covariates.
- `y::Array{<:Real}`: outcome.
- `β::Array{<:Real}`=0.15: fitted weights.

# Keywords
- `two_sided::Bool=false`: whether to conduct a one-sided hypothesis test.

# Examples
```julia
julia> x, y, β = reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 0.5
julia> p_val(x, y, β)
julia> p_val(x, y, β; n=100, two_sided=true)
```
"""
function p_val(x, y, β; n=1000, two_sided=false)
    x_copy, null = deepcopy(x), Vector{Float64}(undef, n)
    min_x, max_x = minimum(x_copy[:, end]), maximum(x_copy[:, end])

    # Run OLS with random treatment vectors to generate a counterfactual distribution
    @simd for i in 1:n
        if var_type(x_copy[:, end]) isa Continuous
            @inbounds x_copy[:, end] =
                (max_x - min_x) * rand(length(x_copy[:, end])) .+ min_x
        else
            @inbounds x_copy[:, end] = float(rand(min_x:max_x, size(x, 1)))
        end

        null[i] = last(x_copy \ y)
    end

    # Wald test is only one sided
    p = two_sided ? length(null[β .< null]) / n : length(null[abs(β) .< abs.(null)]) / n
    return p
end

"""
    counterfactual_consistency(m; kwargs...)

# Arguments
- `m::Union{CausalEstimator, Metalearner}`: model to validate/test the assumptions of.

# Keywords
- `num_devs=(0.25, 0.5, 0.75, 1.0)::Tuple`: number of standard deviations from which to 
    generate noise from a normal distribution to simulate violations of the counterfactual 
    consistency assumption.
- `num_iterations=10::Int: number of times to simulate a violation of the counterfactual 
    consistency assumption.`

# Notes
Examine the counterfactual consistency assumption. First, this function simulates 
counterfactual outcomes that are offset from the outcomes in the dataset by random scalars
drawn from a N(0, num_std_dev). Then, the procedure is repeated num_iterations times and 
averaged. If the model is a metalearner, then the estimated individual treatment effects 
are averaged and the mean CATE is averaged over all the iterations, otherwise the estimated 
treatment effect is averaged over the iterations. The previous steps are repeated for each 
element in num_devs.

# Examples
```julia
julia> x, t = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]
julia> y = vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> counterfactual_consistency(g_computer)
```
"""
function counterfactual_consistency(model, devs, iterations)
    counterfactual_model = deepcopy(model)
    avg_counterfactual_effects = Dict{String,Float64}()

    for dev in devs
        key = string(dev) * " Standard Deviations from Observed Outcomes"
        avg_counterfactual_effects[key] = 0.0

        # Averaging multiple iterations of random violatons for each std dev
        for iteration in 1:iterations
            counterfactual_model.Y = simulate_counterfactual_violations(model.Y, dev)
            estimate_causal_effect!(counterfactual_model)

            if counterfactual_model isa Metalearner
                avg_counterfactual_effects[key] += mean(counterfactual_model.causal_effect)
            else
                avg_counterfactual_effects[key] += counterfactual_model.causal_effect
            end
        end
        avg_counterfactual_effects[key] /= iterations
    end
    return avg_counterfactual_effects
end

"""
    simulate_counterfactual_violations(y, dev)

# Arguments
- `y::Vector{<:Real}`: vector of real-valued outcomes.
- `dev::Float64`: deviation of the observed outcomes from the true counterfactual outcomes.

# Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]), vec(rand(1:100, 100, 1)) 
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> simulate_counterfactual_violations(g_computer)
-0.7748591231872396
```
"""
function simulate_counterfactual_violations(y::Vector{<:Real}, dev::Float64)
    min_y, max_y = minimum(y), maximum(y)

    if var_type(y) isa Continuous
        violations = (sqrt(var(y)) * dev) * randn(length(y))
        counterfactual_Y = y .+ violations
    else
        counterfactual_Y = ifelse.(rand() > dev, Float64(rand(min_y:max_y)), y)
    end
    return counterfactual_Y
end

"""
    exchangeability(model)

Test the sensitivity of a G-computation or doubly robust estimator or metalearner to a 
violation of the exchangeability assumption.

# References
For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

# Examples
```julia
julia> x, t = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]
julia> y = vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> e_value(g_computer)
```
"""
exchangeability(model) = e_value(model)

"""
    e_value(model)
 
Test the sensitivity of an estimator to a violation of the exchangeability assumption.

# References
For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

# Examples
```julia
julia> x, t = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]
julia> y = vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> e_value(g_computer)
```
"""
function e_value(model)
    rr = risk_ratio(model)
    if rr > 1
        return @fastmath rr + sqrt(rr * (rr - 1))
    else
        rr🥰 = @fastmath 1 / rr
        return @fastmath rr🥰 + sqrt(rr🥰 * (rr🥰 - 1))
    end
end

"""
    binarize(x, cutoff)
 
Convert a vector of counts or a continuous vector to a binary vector.

# Arguments
- `x::Any`: interable of numbers to binarize.
- `x::Any`: threshold after which numbers are converted to 1 and befrore which are converted 
    to 0.

# Examples
```jldoctest
julia> CausalELM.binarize([1, 2, 3], 2)
3-element Vector{Int64}:
 0
 0
 1
```
"""
function binarize(x, cutoff)
    if var_type(x) isa Binary
        return x
    else
        x[x .<= cutoff] .= 0
        x[x .> cutoff] .= 1
    end
    return x
end

"""
    risk_ratio(model)
 
Calculate the risk ratio for an estimated model.

# Notes
If the treatment variable is not binary and the outcome variable is not continuous then the 
treatment variable will be binarized.

# References
For more information on how other quantities of interest are converted to risk ratios see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

# Examples
```julia
julia> x, t = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]
julia> y = vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> risk_ratio(g_computer)
```
"""
risk_ratio(mod) = risk_ratio(var_type(mod.T), mod)

# First we dispatch based on whether the treatment variable is binary or not
# If it is binary, we call the risk_ratio method based on the type of outcome variable
risk_ratio(::Binary, mod) = risk_ratio(Binary(), var_type(mod.Y), mod)

# If the treatment variable is not binary this method gets called
function risk_ratio(::Nonbinary, mod)
    # When the outcome variable is continuous, we can treat it the same as if the treatment
    # variable was binary because we don't use the treatment to calculate the risk ratio
    if var_type(mod.Y) isa Continuous
        return risk_ratio(Binary(), Continuous(), mod)

        # Otherwise, we convert the treatment variable to a binary variable and then 
        # dispatch based on the type of outcome variable
    else
        original_T, binary_T = mod.T, binarize(mod.T, mean(mod.T))
        mod.T = binary_T
        rr = risk_ratio(Binary(), mod)

        # Reset to the original treatment variable
        mod.T = original_T

        return rr
    end
end

# This approximates the risk ratio for a binary treatment with a binary outcome
function risk_ratio(::Binary, ::Binary, mod)
    Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
    Xₜ, Xᵤ = reduce(hcat, (Xₜ, ones(size(Xₜ, 1)))), reduce(hcat, (Xᵤ, ones(size(Xᵤ, 1))))

    # For algorithms that use one model to estimate the outcome
    if hasfield(typeof(mod), :ensemble)
        return @fastmath (mean(predict(mod.ensemble, Xₜ)) / mean(predict(mod.ensemble, Xᵤ)))

        # For models that use separate models for outcomes in the treatment and control group
    else
        hasfield(typeof(mod), :μ₀)
        Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
        return @fastmath mean(predict(mod.μ₁, Xₜ)) / mean(predict(mod.μ₀, Xᵤ))
    end
end

# This approximates the risk ratio with a binary treatment and count or categorical outcome
function risk_ratio(::Binary, ::Count, mod)
    Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
    m, n = size(Xₜ, 1), size(Xᵤ, 1) # The number of obeservations in each group
    Xₜ, Xᵤ = reduce(hcat, (Xₜ, ones(m))), reduce(hcat, (Xᵤ, ones(n)))

    # For estimators with a single model of the outcome variable
    if hasfield(typeof(mod), :ensemble)
        return @fastmath (sum(predict(mod.ensemble, Xₜ)) / m) /
            (sum(predict(mod.ensemble, Xᵤ)) / n)

        # For models that use separate models for outcomes in the treatment and control group
    elseif hasfield(typeof(mod), :μ₀)
        Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
        return @fastmath mean(predict(mod.μ₁, Xₜ)) / mean(predict(mod.μ₀, Xᵤ))
    else
        learner = ELMEnsemble(
                reduce(hcat, (mod.X, mod.T)), 
                mod.Y, 
                mod.sample_size, 
                mod.num_machines, 
                mod.num_feats, 
                mod.num_neurons, 
                mod.activation
            )

        fit!(learner)
        @fastmath mean(predict(learner, Xₜ)) / mean(predict(learner, Xᵤ))
    end
end

# This approximates the risk ratio when the outcome variable is continuous
function risk_ratio(::Binary, ::Continuous, mod)
    type = typeof(mod)
    # We use the estimated effect if using DML because it uses linear regression
    d = hasfield(type, :coefficients) ? mod.causal_effect : mean(mod.Y) / sqrt(var(mod.Y))
    return @fastmath exp(0.91 * d)
end

"""
    positivity(model, [,min], [,max])
 
Find likely violations of the positivity assumption.

# Notes
This method uses an extreme learning machine or regularized extreme learning machine to 
estimate probabilities of treatment. The returned matrix, which may be empty, are the 
covariates that have a (near) zero probability of treatment or near zero probability of 
being assigned to the control group, whith their entry in the last column being their 
estimated treatment probability. In other words, they likely violate the positivity 
assumption.

# Arguments
- `model::Union{CausalEstimator, Metalearner}`: a model to validate/test the assumptions of.
- `min::Float64`=1.0e-6: minimum probability of treatment for the positivity assumption.
- `high::Float64=1-min`: the maximum probability of treatment for the positivity assumption.

# Examples
```julia
julia> x, t = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]
julia> y = vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> positivity(g_computer)
```
"""
function positivity(model, min=1.0e-6, max=1 - min)
    ps_mod = ELMEnsemble(
            model.X, 
            model.T, 
            model.sample_size, 
            model.num_machines, 
            model.num_feats, 
            model.num_neurons, 
            model.activation
        )

    fit!(ps_mod)
    propensity_scores = predict(ps_mod, model.X)

    # Observations that have a zero probability of treatment or control assignment
    return reduce(
        hcat,
        (
            model.X[propensity_scores .<= min .|| propensity_scores .>= max, :],
            propensity_scores[propensity_scores .<= min .|| propensity_scores .>= max],
        ),
    )
end

function positivity(model::XLearner, min=1.0e-6, max=1 - min)
    # Observations that have a zero probability of treatment or control assignment
    return reduce(
        hcat,
        (
            model.X[model.ps .<= min .|| model.ps .>= max, :],
            model.ps[model.ps .<= min .|| model.ps .>= max],
        ),
    )
end
