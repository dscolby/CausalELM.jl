"""Abstract type used to dispatch risk_ratio on nonbinary treatments"""
abstract type Nonbinary end

"""Type used to dispatch risk_ratio on binary treatments"""
struct Binary end

"""Type used to dispatch risk_ratio on count and categorical treatments"""
struct Count <: Nonbinary end

"""Type used to dispatch risk_ratio on continuous treatments"""
struct Continuous <: Nonbinary end

"""
    var_type(x)

Determine the type of variable held by a vector.

Examples
```julia
julia> var_type([1, 2, 3, 2, 3, 1, 1, 3, 2])
Binary
```
"""
function var_type(x::Vector{<:Real})
    x_set = Set(x)
    if x_set == Set([0, 1]) || x_set == Set([0]) || x_set == Set([1])
        return Binary()
    elseif x_set == Set(round.(x_set))
        return Count()
    else
        return Continuous()
    end
end

"""
    validate(its; <keyword arguments>)

Test the validity of an estimated interrupted time series analysis.

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

For more details on the assumptions and validity of interrupted time series designs, see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

* Note that this method does not implement the second test in Baicker and Svoronos because 
the estimator in this package models the relationship between covariates and the outcome and 
uses an extreme learning machine instead of linear regression, so variance in the outcome 
across different bins is not much of an issue.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

...
# Arguments
- `its::InterruptedTimeSeries`: an interrupted time seiries estimator.
- `n::Int`: the number of times to simulate a confounder.
- `low::Float64`=0.15: the minimum proportion of data points to include before or after the 
    tested break in the Wald supremum test.
- `high::Float64=0.85`: the maximum proportion of data points to include before or after the 
    tested break in the Wald supremum test.
...

Examples
```julia
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
[0.25714308]
julia> validate(m1)
{"Task" => "Regression", "Regularized" => true, "Activation Function" => relu, 
"Validation Metric" => "mse","Number of Neurons" => 2, 
"Number of Neurons in Approximator" => 10, "β" => [0.25714308], 
"Causal Effect" => -3.9101138, "Standard Error" => 1.903434356, "p-value" = 0.00123356}
```
"""
function validate(its::InterruptedTimeSeries; n=1000, low=0.15, high=0.85)
    if !isdefined(its, :Δ)
        throw(ErrorException("call estimate_causal_effect! before calling validate"))
    end

    return covariate_independence(its; n=n), sup_wald(its; low=low, high=high, n=n), 
        omitted_predictor(its; n=n)
end

"""
    validate(m; <keyword arguments>)

This method tests the counterfactual consistency, exchangeability, and positivity 
assumptions required for causal inference. It should be noted that consistency and 
exchangeability are not directly testable, so instead, these tests do not provide definitive 
evidence of a violation of these assumptions. To probe the counterfactual consistency 
assumption, we assume there were multiple levels of treatments and find them by binning the
dependent vairable for treated observations using Jenks breaks. The optimal number of breaks 
between 2 and num_treatments is found using the elbow method. Using these hypothesized 
treatment assignemnts, this method compares the MSE of linear regressions using the observed 
and hypothesized treatments. If the counterfactual consistency assumption holds then the 
difference between the MSE with hypothesized treatments and the observed treatments should 
be positive because the hypothesized treatments should not provide useful information. If 
it is negative, that indicates there was more useful information provided by the 
hypothesized treatments than the observed treatments or that there is an unobserved 
confounder. Next, this methods tests the model's sensitivity to a violation of the 
exchangeability assumption by calculating the E-value, which is the minimum strength of 
association, on the risk ratio scale, that an unobserved confounder would need to have with 
the treatment and outcome variable to fully explain away the estimated effect. Thus, higher 
E-values imply the model is more robust to a violation of the exchangeability assumption. 
Finally, this method tests the positivity assumption by estimating propensity scores. Rows
in the matrix are levels of covariates that have a zero probability of treatment. If the 
matrix is empty, none of the observations have an estimated zero probability of treatment, 
which implies the positivity assumption is satisfied.


For a thorough review of casual inference assumptions see:
    Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
    Francis, 2024. 

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

...
# Arguments
- `m::Union{CausalEstimator, Metalearner}`: a model to validate/test the assumptions of.
- `num_treatments=5::Int`: the maximum number of treatments to use when testing the 
    plausability of the counterfactual consistency assumption.
- `min::Float64`=1.0e-6: minimum probability of treatment for the positivity assumption.
- `high::Float64=1-min`: the maximum probability of treatment for the positivity assumption.
...

Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100]), 
            vec(rand(1:100, 100, 1)), 
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> validate(g_computer)
 2.7653668647301795
    ```
"""
function validate(m; num_treatments=5, min=1.0e-6, max=1.0-min)
    if !isdefined(m, :causal_effect) || m.causal_effect === NaN
        throw(ErrorException("call estimate_causal_effect! before calling validate"))
    end

    # The causal effect is initialized to zeros in dobuly robust estimation
    if isdefined(m, :causal_effect) && all(m.causal_effect .== 0)
        throw(ErrorException("call estimate_causal_effect! before calling validate"))
    end

    return counterfactual_consistency(m, num_treatments=num_treatments), exchangeability(m), 
        positivity(m, min, max)
end

function validate(R::RLearner; num_treatments=5, min=1.0e-6, max=1.0-min)
    return validate(R.dml, num_treatments=num_treatments, min=min, max=max)
end

function validate(S::SLearner; num_treatments=5, min=1.0e-6, max=1.0-min)
    return validate(S.g, num_treatments=num_treatments, min=min, max=max)
end

"""
    covariate_independence(its; <keyword arguments>)

Test for independence between covariates and the event or intervention.

This is a Chow Test for covariates with p-values estimated via randomization inference. The 
p-values are the proportion of times randomly assigning observations to the pre or 
post-intervention period would have a larger estimated effect on the the slope of the 
covariates. The lower the p-values, the more likely it is that the event/intervention 
effected the covariates and they cannot provide an unbiased prediction of the counterfactual 
outcomes.

For more information on using a Chow Test to test for structural breaks see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

...
# Arguments
- `its::InterruptedTImeSeries`: an interrupted time seiries estimator.
- `n::Int`: the number of permutations for assigning observations to the pre and 
    post-treatment periods.
...

Examples
```julia
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> covariate_independence(its)
 Dict("Column 1 p-value" => 0.421, "Column 5 p-value" => 0.07, "Column 3 p-value" => 0.01, 
 "Column 2 p-value" => 0.713, "Column 4 p-value" => 0.043)
```
"""
function covariate_independence(its::InterruptedTimeSeries; n=1000)
    y₀ = reduce(hcat, (its.X₀[:, 1:end-1], zeros(size(its.X₀, 1))))
    y₁ = reduce(hcat, (its.X₁[:, 1:end-1], ones(size(its.X₁, 1))))
    all_vars = [reduce(vcat, (y₀, y₁)) ones(size(y₀, 1) + size(y₁, 1))]
    x = all_vars[:, end-1:end]
    results = Dict{String, Float64}()

    # Estimate a linear regression with each covariate as a dependent variable and all other
    # covariates and time as independent variables
    for i in 1:size(all_vars, 2)-2
        y = all_vars[:, i] 
        β = first(x\y)
        p = p_val(x, y, β, n=n)
        results["Column " * string(i) * " p-value"] = p
    end
    return results
end

"""
    omitted_predictor(its; <keyword arguments>)

See how an omitted predictor/variable could change the results of an interrupted time series 
analysis.

This method reestimates interrupted time series models with uniform random variables. If the 
included covariates are good predictors of the counterfactual outcome, adding a random 
variable as a covariate should not have a large effect on the predicted counterfactual 
outcomes and therefore the estimated average effect.

For more information on using a Chow Test to test for structural breaks see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

...
# Arguments
- `its::InterruptedTImeSeries`: an interrupted time seiries estimator.
- `n::Int`: the number of times to simulate a confounder.
...

Examples
```julia
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> omitted_predictor(its)
 Dict("Mean Biased Effect/Original Effect" => -0.1943184744720332, "Median Biased 
 Effect/Original Effect" => -0.1881814122689084, "Minimum Biased Effect/Original Effect" => 
 -0.2725194360603799, "Maximum Biased Effect/Original Effect" => -0.1419197976977072)
```
"""
function omitted_predictor(its::InterruptedTimeSeries; n=1000)
    if !isdefined(its, :Δ)
        throw(ErrorException("call estimatecausaleffect! before calling omittedvariable"))
    end

    its_copy = deepcopy(its)
    biased_effects = Vector{Float64}(undef, n)
    results = Dict{String, Float64}()

    for i in 1:n
        its_copy.X₀ = reduce(hcat, (its.X₀, rand(size(its.X₀, 1))))
        its_copy.X₁ = reduce(hcat, (its.X₁, rand(size(its.X₁, 1))))
        biased_effects[i] = mean(estimate_causal_effect!(its_copy))
    end
    
    biased_effects = sort(biased_effects)
    results["Minimum Biased Effect/Original Effect"] = biased_effects[1]
    results["Mean Biased Effect/Original Effect"] = mean(biased_effects)
    results["Maximum Biased Effect/Original Effect"] = biased_effects[n]
    median = ifelse(n%2 === 1, biased_effects[Int(ceil(n/2))], 
        mean([biased_effects[Int(n/2)], biased_effects[Int(n/2)+1]]))
    results["Median Biased Effect/Original Effect"] = median

    return results
end

"""
    sup_wald(its; <keyword arguments>)

Check if the predicted structural break is the hypothesized structural break.

This method conducts Wald tests and identifies the structural break with the highest Wald 
statistic. If this break is not the same as the hypothesized break, it could indicate an 
anticipation effect, confounding by some other event or intervention, or that the 
intervention or policy took place in multiple phases. p-values are estimated using 
approximate randomization inference and represent the proportion of times we would see a 
larger Wald statistic if the data points were randomly allocated to pre and post-event 
periods for the predicted structural break.

For more information on using a Chow Test to test for structural breaks see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.
    
For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

...
# Arguments
- `its::InterruptedTimeSeries`: an interrupted time seiries estimator.
- `n::Int`: the number of times to simulate a confounder.
- `low::Float64`=0.15: the minimum proportion of data points to include before or after the 
    tested break in the Wald supremum test.
- `high::Float64=0.85`: the maximum proportion of data points to include before or after the 
    tested break in the Wald supremum test.
...

Examples
```julia
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> sup_wald(its)
 Dict{String, Real}("Wald Statistic" => 58.16649796321913, "p-value" => 0.005, "Predicted 
 Break Point" => 39, "Hypothesized Break Point" => 100)
```
"""
function sup_wald(its::InterruptedTimeSeries; low=0.15, high=0.85, n=1000)
    hypothesized_break, current_break, wald = size(its.X₀, 1), size(its.X₀, 1), 0.0
    high_idx, low_idx = Int(floor(high * size(its.X₀, 1))), Int(ceil(low * size(its.X₀, 1)))
    x, y = reduce(vcat, (its.X₀, its.X₁))[:, 1:end-1], reduce(vcat, (its.Y₀, its.Y₁))
    t = reduce(vcat, (zeros(size(its.X₀, 1)), ones(size(its.X₁, 1))))
    best_x = reduce(hcat, (t, x, ones(length(t))))
    best_β = first(best_x\y)
    
    # Set each time as a potential break and calculate its Wald statistic
    for idx in low_idx:high_idx
        t = reduce(vcat, (zeros(idx), ones(size(x, 1)-idx)))
        new_x = reduce(hcat, (t, x, ones(size(x, 1))))
        β, ŷ = @fastmath new_x\y, new_x*(new_x\y)
        se = @fastmath sqrt(1/(size(x, 1)-2))*(sum(y .- ŷ)^2/sum(t .- mean(t))^2)
        wald_candidate = first(β)/se

        if wald_candidate > wald
            current_break, wald, best_x, best_β = idx, wald_candidate, new_x, best_β
        end
    end
    p = p_val(best_x, y, best_β; n=n, two_sided=true)
    return Dict("Hypothesized Break Point" => hypothesized_break, 
        "Predicted Break Point" => current_break, "Wald Statistic" => wald, "p-value" => p)
end

"""
    p_val(x, y, β; <keyword arguments>)

Estimate the p-value for the hypothesis that an event had a statistically significant effect 
on the slope of a covariate using randomization inference.

...
# Arguments
- `x::Array{<:Real}`: covariates.
- `y::Array{<:Real}`: the outcome.
- `β::Array{<:Real}`=0.15: the fitted weights.
- `two_sided::Bool=false`: whether to conduct a one-sided hypothesis test.
...

Examples
```julia
julia> x, y, β = reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 0.5
julia> p_val(x, y, β)
 0.98
julia> p_val(x, y, β; n=100, two_sided=true)
 0.08534054
```
"""
function p_val(x, y, β; n=1000, two_sided=false)
    m2 = "the first column of x should be a treatment vector of 0s and 1s"
    if sort(union(x[:, 1], [0, 1])) != [0, 1]
        throw(ArgumentError(m2))
    end

    l = unique(x[:, end])

    if length(l) !== 1 | (length(l) === 1 && l[1] !== 1.0)
        throw(ArgumentError("the second column in x should be an intercept with all 1s"))
    end

    x_copy, null = deepcopy(x), Vector{Float64}(undef, n)

    # Run OLS with random treatment vectors to generate a counterfactual distribution
    @simd for i in 1:n
        @inbounds x_copy[:, 1] = float(rand(0:1, size(x, 1)))  # Random treatment vector
        null[i] = first(x_copy\y)
    end

    # Wald test is only one sided
    p = two_sided ? length(null[β.<null])/n : length(null[abs(β).<abs.(null)])/n
    return p
end

"""
    counterfactual_consistency(m; <keyword arguments>)

Examine the counterfactual consistency assumption. First, this function generates Jenks 
breaks based on outcome values for the treatment group. Then, it replaces treatment statuses 
with the numbers corresponding to each group. Next, it runs two linear regressions, one for 
the treatment group, one with and one without the fake treatment assignemnts generated by 
the Jenks breaks. Finally, it subtracts the mean squared error from the regression with real 
data from the mean squared error from the regression with the fake treatment statuses. If 
this number is negative, it might indicate a violation of the counterfactual consistency 
assumption or omitted variable bias.

For a primer on G-computation and its assumptions see:
    Naimi, Ashley I., Stephen R. Cole, and Edward H. Kennedy. "An introduction to g 
    methods." International journal of epidemiology 46, no. 2 (2017): 756-762.

...
# Arguments
- `m::Union{CausalEstimator, Metalearner}`: a model to validate/test the assumptions of.
- `num_treatments=5::Int`: the maximum number of treatments to use when testing the 
    plausability of the counterfactual consistency assumption.
...

Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100], 
            vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> counterfactual_consistency(g_computer)
 2.7653668647301795
```
"""
function counterfactual_consistency(m; num_treatments=5)
    treatment_covariates, treatment_outcomes = m.X[m.T .== 1, :], m.Y[m.T .== 1]
    fake_treat = best_splits(treatment_outcomes, num_treatments)
    β_real = treatment_covariates\treatment_outcomes
    β_fake = Real.(reduce(hcat, (treatment_covariates, fake_treat))\treatment_outcomes)
    ŷ_real = treatment_covariates*β_real
    ŷ_fake = Real.(reduce(hcat, (treatment_covariates, fake_treat))*β_fake)
    mse_real_treat = mse(treatment_outcomes, ŷ_real)
    mse_fake_treat = mse(treatment_outcomes, ŷ_fake)

    return mse_fake_treat - mse_real_treat
end

"""
    exchangeability(model)

Test the sensitivity of a G-computation or doubly robust estimator or metalearner to a 
violation of the exchangeability assumption.

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100], 
            vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> e_value(g_computer)
 1.13729886008143832
```
"""
exchangeability(model) = e_value(model)

"""
    e_value(model)
 
Test the sensitivity of an estimator to a violation of the exchangeability assumption.

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100], 
            vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> e_value(g_computer)
 2.2555405766985125
```
"""
function e_value(model)
    rr = risk_ratio(model)
    if rr > 1
        return @fastmath rr + sqrt(rr*(rr-1))
    else
        rr🥰 = @fastmath 1/rr
        return @fastmath rr🥰 + sqrt(rr🥰*(rr🥰-1))
    end
end

"""
    binarize(x, cutoff)
 
Convert a vector of counts or a continuous vector to a binary vector.

Examples
```julia
julia> binarize([1, 2, 3], 2)
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

If the treatment variable is not binary and the outcome variable is not continuous then the 
treatment variable will be binarized.

For more information on how other quantities of interest are converted to risk ratios see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100], 
            vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> risk_ratio(g_computer)
 2.5320694766985125
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
        original_T, binary_T = mod.T, binarize(mod.T, mean(mod.Y))
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
    if hasfield(typeof(mod), :learner)
        return @fastmath mean(predict(mod.learner, Xₜ))/mean(predict(mod.learner, Xᵤ))

    # For models that use separate models for outcomes in the treatment and control group
    else hasfield(typeof(mod), :μ₀)
        Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
        return @fastmath mean(predict(mod.μ₁, Xₜ))/mean(predict(mod.μ₀, Xᵤ))
    end
end

# This approximates the risk ratio with a binary treatment and count or categorical outcome
function risk_ratio(::Binary, ::Count, mod)
    Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
    m, n = size(Xₜ, 1), size(Xᵤ, 1) # The number of obeservations in each group
    Xₜ, Xᵤ = reduce(hcat, (Xₜ, ones(m))), reduce(hcat, (Xᵤ, ones(n)))

    # For estimators with a single model of the outcome variable
    if hasfield(typeof(mod), :learner)
        return @fastmath (sum(predict(mod.learner, Xₜ))/m)/(sum(predict(mod.learner, Xᵤ))/n)

    # For models that use separate models for outcomes in the treatment and control group
    elseif hasfield(typeof(mod), :μ₀)
        Xₜ, Xᵤ = mod.X[mod.T .== 1, :], mod.X[mod.T .== 0, :]
        return @fastmath mean(predict(mod.μ₁, Xₜ))/mean(predict(mod.μ₀, Xᵤ))
    else
        if mod.regularized
            learner = RegularizedExtremeLearner(reduce(hcat, (mod.X, mod.T)), mod.Y, 
                                                mod.num_neurons, mod.activation)
        else
            learner = ExtremeLearner(reduce(hcat, (mod.X, mod.T)), mod.Y, mod.num_neurons, 
                                     mod.activation)
        end
        fit!(learner)
        @fastmath (sum(predict(learner, Xₜ))/m)/(sum(predict(learner, Xᵤ))/n)
    end
end

# This approximates the risk ratio when the outcome variable is continuous
function risk_ratio(::Binary, ::Continuous, mod)
    type = typeof(mod)
    # We use the estimated effect if using DML because it uses linear regression
    d = hasfield(type, :coefficients) ? mod.causal_effect : mean(mod.Y)/sqrt(var(mod.Y))
    return @fastmath exp(0.91 * d)
end

"""
    positivity(model[,min][,max])
 
Find likely violations of the positivity assumption.

This method uses an extreme learning machine or regularized extreme learning machine to 
estimate probabilities of treatment. The returned matrix, which may be empty, are the 
covariates that have a (near) zero probability of treatment or near zero probability of 
being assigned to the control group, whith their entry in the last column being their 
estimated treatment probability. In other words, they likely violate the positivity 
assumption.

...
# Arguments
- `model::Union{CausalEstimator, Metalearner}`: a model to validate/test the assumptions of.
- `min::Float64`=1.0e-6: minimum probability of treatment for the positivity assumption.
- `high::Float64=1-min`: the maximum probability of treatment for the positivity assumption.
...

Examples
```julia
julia> x, t, y = rand(100, 5), Float64.([rand()<0.4 for i in 1:100], 
            vec(rand(1:100, 100, 1)))
julia> g_computer = GComputation(x, t, y, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> positivity(g_computer)
0×5 Matrix{Float64}
```
"""
positivity(model, min=1.0e-6, max=1-min) = positivity(model, min, max)

function positivity(mod::XLearner, min=1.0e-6, max=1-min)
    # Observations that have a zero probability of treatment or control assignment
    return reduce(hcat, (mod.X[mod.ps .<= min .|| mod.ps .>= max, :], 
                  mod.ps[mod.ps .<= min .|| mod.ps .>= max]))
end

function positivity(mod::DoubleMachineLearning, min=1.0e-6, max=1-min)
    task = mod.t_cat || var_type(mod.T) == Binary() ? "classification" : "regression"
    T = mod.t_cat ? one_hot_encode(mod.T) : mod.T

    num_neurons = best_size(mod.X, T, mod.validation_metric, task, mod.activation, 
                            mod.min_neurons, mod.max_neurons, mod.regularized, mod.folds, 
                            false,  mod.iterations, mod.approximator_neurons)

    if mod.regularized
        ps_mod = RegularizedExtremeLearner(mod.X, mod.T, num_neurons, mod.activation)
    else
        ps_mod = ExtremeLearner(mod.X, mod.T, num_neurons, mod.activation)
    end

    fit!(ps_mod)
    propensity_scores = predict(ps_mod, mod.X)

    # Observations that have a zero probability of treatment or control assignment
    return reduce(hcat, (mod.X[propensity_scores .<= min .|| propensity_scores .>= max, :], 
                  propensity_scores[propensity_scores .<= min .|| 
                  propensity_scores .>= max]))
end

function positivity(mod, min=1.0e-6, max=1-min)
    if mod.regularized
        ps_mod = RegularizedExtremeLearner(mod.X, mod.T, mod.num_neurons, mod.activation)
    else
        ps_mod = ExtremeLearner(mod.X, mod.T, mod.num_neurons, mod.activation)
    end

    fit!(ps_mod)
    propensity_scores = predict(ps_mod, mod.X)

    # Observations that have a zero probability of treatment or control assignment
    return reduce(hcat, (mod.X[propensity_scores .<= min .|| propensity_scores .>= max, :], 
                  propensity_scores[propensity_scores .<= min .|| 
                  propensity_scores .>= max]))
end

"""
    sums_of_squares(data, num_classes)

Calculate the minimum sum of squares for each data point and class for the Jenks breaks 
    algorithm.

This should not be called by the user.

Examples
```julia
julia> sums_of_squares([1, 2, 3, 4, 5], 2)
5×2 Matrix{Real}:
 0.0       0.0
 0.25      0.25
 0.666667  0.666667
 1.25      1.16667
 2.0       1.75
```
"""
function sums_of_squares(data, num_classes=5)
    n = length(data)
    sums_of_squares = zeros(Float64, n, num_classes)
    
    @inbounds for (k, i) in Iterators.product(1:num_classes, 1:n)
        if k == 1
            @inbounds sums_of_squares[i, k] = variance(data[1:i])

        # Calculates the sums of squares for each potential class and break point
        else
            sums = Vector{Float64}(undef, i)
            @simd for j in 1:i
                @inbounds sums[j] = sums_of_squares[j, k-1] + (i-j+1) * variance(data[j:i])
            end
            @inbounds sums_of_squares[i, k] = minimum(sums)
            end
    end
    return sums_of_squares
end

"""
    class_pointers(data, num_classes, sums_of_sqs)

Compute class pointers that minimize the sum of squares for Jenks breaks.

This should not be callled by the user.

Examples
```julia
julia> sums_squares = sums_of_sqs::Matrix{Float64}
5×2 Matrix{Float64}:
 0.0       0.0
 0.25      0.25
 0.666667  0.666667
 1.25      1.16667
 2.0       1.75
julia> class_pointers([1, 2, 3, 4, 5], 2, sums_squares)
5×2 Matrix{Int64}:
 1  0
 1  1
 1  1
 1  1
 1  1
```
"""
function class_pointers(data, num_classes, sums_of_sqs)
    n = length(data)
    class_pointers = Matrix{Int}(undef, n, num_classes)

    # Initialize the first column of class pointers
    for i in 1:n
        @inbounds class_pointers[i, 1] = 1
    end

    # Update class pointers based on their sums of squares
    @simd for k in 2:num_classes
        for i in 2:n
            @inbounds map(1:i) do j
                class_pointers[i, k] = argmin([sums_of_sqs[j, k-1]+class_pointers[j, k-1]])
            end
        end
    end
    return class_pointers
end

"""
    backtrack_to_find_breaks(data, num_classes, sums_of_sqs)

Determine break points from class assignments.

This should not be called by the user.

Examples
```julia
julia> data = [1, 2, 3, 4, 5]
[1, 2, 3, 4, 5]
5-element Vector{Int64}:
 1
 2
 3
 4
 5
julia> ptr = class_pointers([1, 2, 3, 4, 5], 2, sums_of_squares([1, 2, 3, 4, 5], 2))
5×2 Matrix{Int64}:
 1  28
 1   1
 1   1
 1   1
 1   1
julia> backtrack_to_find_breaks([1, 2, 3, 4, 5], ptr)
2-element Vector{Int64}:
 1
 4
```
"""
function backtrack_to_find_breaks(data, class_pointers)
    n, num_classes = size(class_pointers)
    breaks = Vector{eltype(data)}(undef, num_classes)
    current_class, current_break = num_classes, n
    
    @simd for i in (n-1):-1:1
        if class_pointers[i, current_class] != current_break
            @inbounds current_break = class_pointers[i, current_class]
            @inbounds breaks[current_class] = data[i+1]
            current_class -= 1
        end
    end
    
    # Assigns breaks at the smallest value in the data if a class doesn't have a break
    @simd for j in current_class:-1:1
        @inbounds breaks[j] = data[1]
    end

    return breaks
end

"""
    variance(data)

Calculate the variance of some numbers.

Note this function does not use Besel's correction.

Examples
```julia
julia> variance([1, 2, 3, 4, 5])
2.0
```
"""
function variance(data)
    mean_val = mean(data)
    sum_squares = sum((x - mean_val)^2 for x in data)

    return sum_squares/length(data)
end

"""
    best_splits(data, num_classes)

Find the best number of splits for Jenks breaks.

This function finds the best number of splits by finding the number of splits that results 
    in the greatest decrease in the slope of the line between itself and its GVF and the 
    next higher number of splits and its GVF. This is the same thing as the elbow method.

This should nto be called by the user.

Examples
```julia
julia> best_splits(collect(1:10), 5)
10-element Vector{Int64}:
 1
 3
 3
 ⋮
 3
 4
```
"""
function best_splits(data, num_classes)
    candidate_classes = [fake_treatments(data, i) for i in 2:num_classes]
    grouped_candidate_breaks = [group_by_class(data, class) for class in candidate_classes]
    gvfs = [gvf(breaks) for breaks in grouped_candidate_breaks]

    # Find the set of splits with the largest decrease in slope
    rise, run = consecutive(gvfs), consecutive(collect(2:num_classes))
    δ_slope = consecutive(rise ./ run)
    _, δ_idx = findmax(δ_slope)

    return candidate_classes[δ_idx+1]
end

"""
    group_by_class(data, classes)

Group data points into vectors such that data points assigned to the same class are in the 
same vector.

This should nto be called by the user.

Examples
```julia
julia> group_by_class([1, 2, 3, 4, 5], [1, 1, 1, 2, 3])
3-element Vector{Vector{Real}}:
 [1, 2, 3]
 [4]
 [5]
```
"""
function group_by_class(data, classes)
    # Create a dictionary to store elements by class
    class_dict = Dict{Int, Vector{Real}}()
    
    # Iterate through numbers and classes and group elements
    for (point, cls) in zip(data, classes)
        if haskey(class_dict, cls)
            push!(class_dict[cls], point)
        else
            class_dict[cls] = [point]
        end
    end
    # Convert the dictionary to a vector of vectors
    class_vectors = [class_dict[cls] for cls in unique(classes)]
    return class_vectors
end

"""
    jenks_breaks(data, num_classes)

Generate Jenks breaks for a vector of real numbers.

Examples
```julia
julia> jenks_breaks([1, 2, 3, 4, 5], 3)
3-element Vector{Int64}:
 1
 3
 4
```
"""
function jenks_breaks(data, num_classes)
    sorted_data = sort(data)                 # The data needs to be sorted for this to work
    sums_squares = sums_of_squares(sorted_data, num_classes)
    cls_pointers = class_pointers(sorted_data, num_classes, sums_squares)
    breaks = backtrack_to_find_breaks(sorted_data, cls_pointers)
    
    return breaks
end

"""
    fake_treatments(data, num_classes)

Generate fake treatment statuses corresponding to the classes assigned by the Jenks breaks 
algorithm.

Examples
```julia
julia> fake_treatments([1, 2, 3, 4, 5], 4)
5-element Vector{Int64}:
 1
 2
 3
 4
 4
```
"""
function fake_treatments(data, num_classes)
    breaks = jenks_breaks(data, num_classes)

    # Finds the class of a single data point
    function find_class(number)
        if number <= breaks[1]
            return 1
        elseif number >= breaks[end]
            return num_classes
        else
            for i in 2:(length(breaks))
                if number < breaks[i]
                    return i-1
                elseif number == breaks[i]
                    return i
                end
            end
        end
    end
    return find_class.(data)
end

"""
    sdam(x)

Calculate the sum of squared deviations for array mean for a set of sub arrays.

Examples
```julia
julia> sdam([5, 4, 9, 10]) 
26.0
```
"""
function sdam(x::Vector{T}) where T <: Real
    x̄ = mean(x)

    return @fastmath sum((x .- x̄).^2)
end

"""
    sdcm(x)

Calculate the sum of squared deviations for class means for a set of sub arrays.

Examples
```julia
julia> scdm([[4], [5, 9, 10]]) 
14.0
```
"""
scdm(x::Vector{Vector{T}}) where T <: Real = @fastmath sum(sdam.(x))

"""
    gvf(x)

Calculate the goodness of variance fit for a set of sub vectors.

Examples
```julia
julia> gvf([[4, 5], [9, 10]])
0.96153846153
```
"""
function gvf(x::Vector{Vector{T}}) where T <: Real
    return (sdam(collect(Iterators.flatten(x)))-scdm(x))/sdam(collect(Iterators.flatten(x)))
end
