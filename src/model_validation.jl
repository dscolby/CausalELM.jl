module ModelValidation

using ..Estimators: InterruptedTimeSeries, estimate_causal_effect!, GComputation
using ..Metrics: mse
using CausalELM: mean, consecutive

"""
    validate(its; n, low, high)

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

Examples
```julia-repl
julia> X₀, Y₀, X₁, Y₁ =  rand(100, 5), rand(100), rand(10, 5), rand(10)
julia> m1 = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
julia> estimate_causal_effect!(m1)
[0.25714308]
julia> testassumptions(m1)
{"Task" => "Regression", "Regularized" => true, "Activation Function" => relu, 
"Validation Metric" => "mse","Number of Neurons" => 2, 
"Number of Neurons in Approximator" => 10, "β" => [0.25714308], 
"Causal Effect" => -3.9101138, "Standard Error" => 1.903434356, "p-value" = 0.00123356}
```
"""
function validate(its::InterruptedTimeSeries; n::Int=1000, low::Float64=0.15, 
    high::Float64=0.85)
    if !isdefined(its, :Δ)
        throw(ErrorException("call estimatecausaleffect! before calling testassumptions"))
    end

    return testcovariateindependence(its; n=n), supwald(its; low=low, high=high, n=n), 
        testomittedpredictor(its; n=n)
end

"""
    testcovariateindependence(its; n)

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

Examples
```julia-repl
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> testcovariateindependence(its)
Dict("Column 1 p-value" => 0.421, "Column 5 p-value" => 0.07, "Column 3 p-value" => 0.01, 
"Column 2 p-value" => 0.713, "Column 4 p-value" => 0.043)
```
"""
function testcovariateindependence(its::InterruptedTimeSeries; n::Int=1000)
    y₀ = reduce(hcat, (its.X₀[:, 1:end-1], zeros(size(its.X₀, 1))))
    y₁ = reduce(hcat, (its.X₁[:, 1:end-1], ones(size(its.X₁, 1))))
    all_vars = [reduce(vcat, (y₀, y₁)) ones(size(y₀, 1) + size(y₁, 1))]
    x = all_vars[:, end-1:end]
    results = Dict{String, Float64}()

    for i in 1:size(all_vars, 2)-2
        y = all_vars[:, i] 
        β = first(x\y)
        p = pval(x, y, β, n=n)
        results["Column " * string(i) * " p-value"] = p
    end
    return results
end

"""
    testomittedpredictor(its; n)

See how an omitted predictor/variable could change the results of an interrupted time series 
analysis.

This method reestimates interrupted time series models with normal random variables and 
uniform noise. If the included covariates are good predictors of the counterfactual outcome, 
adding a random variable as a covariate should not have a large effect on the predicted 
counterfactual outcomes and therefore the estimated average effect.

For more information on using a Chow Test to test for structural breaks see:
    Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
    interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

For a primer on randomization inference see: 
    https://www.mattblackwell.org/files/teaching/s05-fisher.pdf

Examples
```julia-repl
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> testomittedpredictor(its)
Dict("Mean Biased Effect/Original Effect" => -0.1943184744720332, "Median Biased 
Effect/Original Effect" => -0.1881814122689084, "Minimum Biased Effect/Original Effect" => 
-0.2725194360603799, "Maximum Biased Effect/Original Effect" => -0.1419197976977072)
```
"""
function testomittedpredictor(its::InterruptedTimeSeries; n::Int=1000)
    if !isdefined(its, :Δ)
        throw(ErrorException("call estimatecausaleffect! before calling omittedvariable"))
    end

    its_copy = deepcopy(its)
    biased_effects = Vector{Float64}(undef, n)
    results = Dict{String, Float64}()

    for i in 1:n
        its_copy.X₀ = reduce(hcat, (its.X₀, randn(size(its.X₀, 1)).+rand(size(its.X₀, 1))))
        its_copy.X₁ = reduce(hcat, (its.X₁, randn(size(its.X₁, 1)).+rand(size(its.X₁, 1))))
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
    supwald(its; low, high, n)

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

Examples
```julia-repl
julia> x₀, y₀, x₁, y₁ = (Float64.(rand(1:5, 100, 5)), randn(100), rand(1:5, (10, 5)), 
           randn(10))
julia> its = InterruptedTimeSeries(x₀, y₀, x₁, y₁)
julia> estimate_causal_effect!(its)
julia> supwald(its)
Dict{String, Real}("Wald Statistic" => 58.16649796321913, "p-value" => 0.005, "Predicted 
Break Point" => 39, "Hypothesized Break Point" => 100)
```
"""
function supwald(its::InterruptedTimeSeries; low::Float64=0.15, high::Float64=0.85, 
    n::Int=1000)
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
        β = new_x\y
        ŷ = new_x*β
        se = sqrt(1/(size(x, 1)-2))*(sum(y .- ŷ)^2/sum(t .- mean(t))^2)
        wald_candidate = first(β)/se

        if wald_candidate > wald
            current_break, wald, best_x, best_β = idx, wald_candidate, new_x, best_β
        end
    end
    p = pval(best_x, y, best_β; n=n, wald=true)
    return Dict("Hypothesized Break Point" => hypothesized_break, 
        "Predicted Break Point" => current_break, "Wald Statistic" => wald, "p-value" => p)
end

"""
    pval(x, y, β; n, wald)

Estimate the p-value for the hypothesis that an event had a statistically significant effect 
on the slope of a covariate using randomization inference.

Examples
```julia-repl
julia> x, y, β = reduce(hcat, (float(rand(0:1, 10)), ones(10))), rand(10), 0.5
julia> pval(x, y, β)
0.98
julia> pval(x, y, β; n=100, wald=true)
0.08534054
```
"""
function pval(x::Array{Float64}, y::Array{Float64}, β::Float64; n::Int=1000, 
    wald::Bool=false)
    m2 = "the first column of x should be a treatment vector of 0s and 1s"
    if sort(union(x[:, 1], [0, 1])) != [0, 1]
        throw(ArgumentError(m2))
    end

    l = unique(x[:, end])

    if length(l) !== 1 | (length(l) === 1 && l[1] !== 1.0)
        throw(ArgumentError("the second column in x should be an intercept with all 1s"))
    end

    x_copy = deepcopy(x)
    null = Vector{Float64}(undef, n)
    for i in 1:n
        x_copy[:, 1] = float(rand(0:1, size(x, 1)))  # Random treatment vector
        null[i] = first(x_copy\y)
    end

    # Wald test is only one sided
    p = ifelse(wald === true, length(null[β.<null])/n, length(null[abs(β).<abs.(null)])/n)
    return p
end

"""
    counterfactualconsistency(g; num_treatments)

Examine the counterfactual consistency assumption. First, this function generates for Jenks 
breaks based on outcome values for the treatment group. Then, it replaces treatment statuses 
with the numbers corresponding to each group. Next, it runs two linear regressions, one 
with and one without the fake treatment assignemnts generated by the Jenks breaks. Finally, 
it subtracts the mean squared error from the regression with real data from the mean squared 
error from the regression with the fake treatment statuses. If this number is negative, it 
might indicate a violation of the counterfactual consistency assumption or omitted variable
bias.

Examples
```julia-repl
julia> x, y, t = rand(100, 5), vec(rand(1:100, 100, 1)), 
    Float64.([rand()<0.4 for i in 1:100])
julia> g_computer = GComputation(x, y, t, temporal=false)
julia> estimate_causal_effect!(g_computer)
julia> counterfactualconsistency(g_computer)
2.7653668647301795
```
"""
function counterfactualconsistency(g::GComputation; num_treatments::Int=5)
    treatment_covariates, treatment_outcomes = g.X[g.T .== 1, :], g.Y[g.T .== 1]
    fake_treat = bestsplits(treatment_outcomes, num_treatments)
    β_real = treatment_covariates\treatment_outcomes
    β_fake = Real.(reduce(hcat, (treatment_covariates, fake_treat))\treatment_outcomes)
    ŷ_real = treatment_covariates*β_real
    ŷ_fake = Real.(reduce(hcat, (treatment_covariates, fake_treat))*β_fake)
    mse_real_treat = mse(treatment_outcomes, ŷ_real)
    mse_fake_treat = mse(treatment_outcomes, ŷ_fake)

    return mse_fake_treat - mse_real_treat
end

"""
    sumsofsquares(data, num_classes)

Calculate the minimum sum of squares for each data point and class for the Jenks breaks 
    algorithm.

Examples
```julia-repl
julia> sumsofsquares([1, 2, 3, 4, 5], 2)
5×2 Matrix{Real}:
 0.0       0.0
 0.25      0.25
 0.666667  0.666667
 1.25      1.16667
 2.0       1.75
```
"""
function sumsofsquares(data::Vector{<:Real}, num_classes::Int=5)
    n = length(data)
    sums_of_squares = zeros(Float64, n, num_classes)
    
    @inbounds for (k, i) in Iterators.product(1:num_classes, 1:n)
        if k == 1
            sums_of_squares[i, k] = variance(data[1:i])

        # Calculates the sums of squares for each potential class and break point
        else
            sums = Vector{Float64}(undef, i)
            for j in 1:i
                sums[j] = sums_of_squares[j, k-1] + (i-j+1) * variance(data[j:i])
            end
            sums_of_squares[i, k] = minimum(sums)
            end
    end
    return sums_of_squares
end

"""
    classpointers(data, num_classes, sums_of_sqs)

Compute class pointers that minimize the sum of squares for Jenks breaks.

Examples
```julia-repl
julia> sums_squares = sums_of_sqs::Matrix{Float64}
5×2 Matrix{Float64}:
 0.0       0.0
 0.25      0.25
 0.666667  0.666667
 1.25      1.16667
 2.0       1.75
julia> classpointers([1, 2, 3, 4, 5], 2, sums_squares)
5×2 Matrix{Int64}:
 1  0
 1  1
 1  1
 1  1
 1  1
```
"""
function classpointers(data::Vector{<:Real}, num_classes::Int, sums_of_sqs::Matrix{Float64})
    n = length(data)
    class_pointers = Matrix{Int}(undef, n, num_classes)

    # Initialize the first column of class pointers
    for i in 1:n
        class_pointers[i, 1] = 1
    end

    # Update class pointers based on their sums of squares
    for k in 2:num_classes
        for i in 2:n
            map(1:i) do j
                class_pointers[i, k] = argmin([sums_of_sqs[j, k-1]+class_pointers[j, k-1]])
            end
        end
    end
    return class_pointers
end

"""
    backtrack_to_find_breaks(data, num_classes, sums_of_sqs)

Determine break points from class assignments.

Examples
```julia-repl
julia> data = [1, 2, 3, 4, 5]
[1, 2, 3, 4, 5]
5-element Vector{Int64}:
 1
 2
 3
 4
 5
julia> ptr = classpointers([1, 2, 3, 4, 5], 2, sumsofsquares([1, 2, 3, 4, 5], 2))
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
function backtrack_to_find_breaks(data::Vector{<:Real}, class_pointers::Matrix{<:Real})
    n, num_classes = size(class_pointers)
    breaks = Vector{eltype(data)}(undef, num_classes)
    current_class, current_break = num_classes, n
    
    @inbounds for i in (n-1):-1:1
        if class_pointers[i, current_class] != current_break
            current_break = class_pointers[i, current_class]
            breaks[current_class] = data[i+1]
            current_class -= 1
        end
    end
    
    # Assigns breaks at the smallest value in the data if a class doesn't have a break
    @inbounds for j in current_class:-1:1
        breaks[j] = data[1]
    end

    return breaks
end

"""
    variance(data)

Calculate the variance of some numbers.

Note this function does not use Besel's correction.

Examples
```julia-repl
julia> variance([1, 2, 3, 4, 5])
2.0
```
"""
function variance(data::Vector{<:Real})
    mean_val = mean(data)
    sum_squares = sum((x - mean_val)^2 for x in data)

    return sum_squares/length(data)
end

"""
    bestsplits(data, num_classes)

Find the best number of splits for Jenks breaks.

This function finds the best number of splits by finding the number of splits that results 
    in the greatest decrease in the slope of the line between itself and its GVF and the 
    next higher number of splits and its GVF. This is the same thing as the elbow method.

Examples
```julia-repl
julia> bestsplits(collect(1:10), 5)
10-element Vector{Int64}:
 1
 3
 3
 ⋮
 3
 4
```
"""
function bestsplits(data::Vector{<:Real}, num_classes::Int)
    candidate_classes = [faketreatments(data, i) for i in 2:num_classes]
    grouped_candidate_breaks = [groupbyclass(data, class) for class in candidate_classes]
    gvfs = [gvf(breaks) for breaks in grouped_candidate_breaks]

    # Find the set of splits with the largest decrease in slope
    rise, run = consecutive(gvfs), consecutive(collect(2:num_classes))
    δ_slope = consecutive(rise ./ run)
    _, δ_idx = findmax(δ_slope)

    return candidate_classes[δ_idx+1]
end

"""
    groupbyclass(data, classes)

Group data points into vectors such that data points assigned to the same class are in the 
same vector.

Examples
```julia-repl
julia> groupbyclass([1, 2, 3, 4, 5], [1, 1, 1, 2, 3])
3-element Vector{Vector{Real}}:
 [1, 2, 3]
 [4]
 [5]
```
"""
function groupbyclass(data::Vector{<:Real}, classes::Vector{<:Real})
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
    jenksbreaks(data, num_classes)

Generate Jenks breaks for a vector of real numbers.

Examples
```julia-repl
julia> jenksbreaks([1, 2, 3, 4, 5], 3)
3-element Vector{Int64}:
 1
 3
 4
```
"""
function jenksbreaks(data::Vector{<:Real}, num_classes::Int)
    sorted_data = sort(data)                 # The data needs to be sorted for this to work
    sums_of_squares = sumsofsquares(sorted_data, num_classes)
    class_pointers = classpointers(sorted_data, num_classes, sums_of_squares)
    breaks = backtrack_to_find_breaks(sorted_data, class_pointers)
    
    return breaks
end

"""
    faketreatments(data, num_classes)

Generate fake treatment statuses corresponding to the classes assigned by the Jenks breaks 
algorithm.

Examples
```julia-repl
julia> faketreatments([1, 2, 3, 4, 5], 4)
5-element Vector{Int64}:
 1
 2
 3
 4
 4
```
"""
function faketreatments(data::Vector{<:Real}, num_classes::Int)
    breaks = jenksbreaks(data, num_classes)

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
```julia-repl
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
```julia-repl
julia> scdm([[4], [5, 9, 10]]) 
14.0
```
"""
scdm(x::Vector{Vector{T}}) where T <: Real = @fastmath sum(sdam.(x))

"""
    gvf(x)

Calculate the goodness of variance fit for a set of sub vectors.

Examples
```julia-repl
julia> gvf([[4, 5], [9, 10]])
0.96153846153
```
"""
function gvf(x::Vector{Vector{T}}) where T <: Real
    return (sdam(collect(Iterators.flatten(x)))-scdm(x))/sdam(collect(Iterators.flatten(x)))
end

end
