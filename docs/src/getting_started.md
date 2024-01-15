# Deciding Which Estimator to Use
Which model you should use depends on what you are trying to model and the type of data you 
have. The table below can serve as a useful reference when deciding which model to use for a 
given dataset and causal question.

| Model                            | Struct                | Causal Estimands                 | Supported Treatment Types              | Supported Outcome Types   |
|----------------------------------|-----------------------|----------------------------------|----------------------------------------|---------------------------|
| Interrupted Time Series Analysis | InterruptedTimeSeries | ATE, Cumulative Treatment Effect | Binary                                 | Binary, Continuous        |
| G-computation                    | GComputation          | ATE, ATT, ITT                    | Binary                                 | Binary, Continuous, Time to Event        |
| Double Machine Learning          | DoubleMachineLearning | ATE                              | Binary, Count, Categorical, Continuous | Continuous                |
| S-learning                       | SLearner              | CATE                             | Binary                                 | Binary, Continuous, Count |
| T-learning                       | TLearner              | CATE                             | Binary                                 | Binary, Continuous        |
| X-learning                       | XLearner              | CATE                             | Binary                                 | Binary, Continuous, Count |
| R-learning                       | RLearner              | CATE                             | Binary, Count, Categorical, Continuous | Continuous                |


# Interrupted Time Series Analysis
Sometimes we want to know how an outcome variable for a single unit changed after an event 
or intervention. For example, if regulators announce sanctions against company A, we might 
want to know how the price of stock A changed after the announcement. Since we do not know
what the price of Company A's stock would have been if the santions were not announced, we
need some way to predict those values. An interrupted time series analysis does this by 
using some covariates that are related to the oucome variable but not related to whether the 
event happened to predict what would have happened. The estimated effects are the 
differences between the predicted post-event counterfactual outcomes and the observed 
post-event outcomes, which can also be aggregated to mean or cumulative effects. 
Estimating an interrupted time series design in CausalELM consists of three steps.

!!! note
    For a deeper dive see:
    
        Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini. "Interrupted time series 
        regression for the evaluation of public health interventions: a tutorial." International 
        journal of epidemiology 46, no. 1 (2017): 348-355.


## Step 1: Initialize an interrupted time series estimator
The InterruptedTimeSeries method takes at least four agruments: an array of pre-event 
covariates, a vector of pre-event outcomes, an array of post-event covariates, and a vector 
of post-event outcomes. 

You can also specify whether or not to use L2 regularization, which activation function to 
use, the metric to use when using cross validation to find the best number of neurons, the 
minimum number of neurons to consider, the maximum number of neurons to consider, the number 
of folds to use during cross caidation, the number of neurons to use in the ELM that learns 
a mapping from number of neurons to validation loss, and whether to include a rolling 
average autoregressive term. These options can be specified using the keyword arguments 
regularized, activation, validation_metric, min_neurons, max_neurons, folds, iterations, 
approximator_neurons, and autoregression.

```julia
# Generate some data to use
X₀, Y₀, X₁, Y₁ =  rand(1000, 5), rand(1000), rand(100, 5), rand(100)

# We could also use DataFrames
# using DataFrames
# X₀ = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# X₁ = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# Y₀, Y₁ = DataFrame(y=rand(1000)), DataFrame(y=rand(1000))

its = InterruptedTimeSeries(X₀, Y₀, X₁, Y₁)
```

## Step 2: Estimate the Treatment Effect
Estimating the treatment effect only requires one argument: an InterruptedTimeSeries struct.

```julia
estimate_causal_effect!(its)
```

## Step 3: Get a Summary
We can get a summary of the model, including a p-value and statndard via asymptotic 
randomization inference, by pasing the model to the summarize method.

Calling the summarize methodd returns a dictionary with the estimator's task (always 
regression for interrupted time series analysis), whether the model uses an L2 penalty, 
the activation function used in the model's outcome predictors, the validation metric used 
for cross validation to find the best number of neurons, the number of neurons used in the 
ELMs used by the estimator, the number of neurons used in the ELM used to learn a mapping 
from number of neurons to validation loss during cross validation, the causal effect, 
standard error, and p-value.
```julia
summarize(its)
```

## Step 4: Validate the Model
For an interrupted time series design to work well we need to be able to get an unbiased 
prediction of the counterfactual outcomes. If the event or intervention effected the 
covariates we are using to predict the counterfactual outcomes, then we will not be able to 
get unbiased predictions. We can verify this by conducting a Chow Test on the covariates. An
ITS design also assumes that any observed effect is due to the hypothesized intervention, 
rather than any simultaneous interventions, anticipation of the intervention, or any 
intervention that ocurred after the hypothesized intervention. We can use a Wald supremum 
test to see if the hypothesized intervention ocurred where there is the largest structural 
break in the outcome or if there was a larger, statistically significant break in the 
outcome that could confound an ITS analysis. The covariates in an ITS analysis should be 
good predictors of the outcome. If this is the case, then adding irrelevant predictors 
should not have much of a change on the results of the analysis. We can conduct all these 
tests in one line of code.

One can also specify the number of simulated confounders to generate to test the sensitivity 
of the model to confounding and the minimum and maximum proportion of data to use in the 
Wald supremum test by including the n, low, and high keyword arguments.

!!! danger
    Obtaining correct estimates is dependent on meeting the assumptions for interrupted time 
    series estimation. If the assumptions are not met then any estimates may be biased and 
    lead to incorrect conclusions.

!!! note
    For a review of interrupted time series identifying assumptions and robustness checks, see:

        Baicker, Katherine, and Theodore Svoronos. Testing the validity of the single 
        interrupted time series design. No. w26080. National Bureau of Economic Research, 2019.

```julia
validate(its)
```


# G-Computation
In some cases, we may want to know the causal effect of a treatment that varies and is 
confounded over time. For example, a doctor might want to know the effect of a treatment 
given at multiple times whose status depends on the health of the patient at a given time. 
One way to get an unbiased estimate of the causal effect is to use G-computation. The basic 
steps for using G-computation in CausalELM are below.

!!! note 
    For a good overview of G-Computation see:
    
        Chatton, Arthur, Florent Le Borgne, Clémence Leyrat, Florence Gillaizeau, Chloé 
        Rousseau, Laetitia Barbin, David Laplaud, Maxime Léger, Bruno Giraudeau, and Yohann 
        Foucher. "G-computation, propensity score-based methods, and targeted maximum likelihood 
        estimator for causal inference with different covariates sets: a comparative simulation 
        study." Scientific reports 10, no. 1 (2020): 9219.

## Step 1: Initialize a Model
The GComputation method takes at least three arguments: an array of covariates, a vector of 
treatment statuses, and an outcome vector. 

You can also specify the causal estimand, whether to employ L2 regularization, which 
activation function to use, whether the data is of a temporal nature, the metric to use when 
using cross validation to find the best number of neurons, the minimum number of neurons to 
consider, the maximum number of neurons to consider, the number of folds to use during cross 
caidation, and the number of neurons to use in the ELM that learns a mapping from number of 
neurons to validation loss. These are options are specified with the following keyword 
arguments: quantity_of_interest, regularized, activation, temporal, validation_metric, 
min_neurons, max_neurons, folds, iterations, and approximator_neurons.
```julia
# Create some data with a binary treatment
X, T, Y =  rand(1000, 5), [rand()<0.4 for i in 1:1000], rand(1000)

# We could also use DataFrames
# using DataFrames
# X = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:1000]), DataFrame(y=rand(1000))

g_computer = GComputation(X, T, Y)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we pass the model above to estimatecausaleffect!.
```julia
# Note that we could also estimate the ATT by setting quantity_of_interest="ATT"
estimate_causal_effect!(g_computer)
```

## Step 3: Get a Summary
We get a summary of the model that includes a p-value and standard error estimated via 
asymptotic randomization inference by passing our model to the summarize method.

Calling the summarize methodd returns a dictionary with the estimator's task (regression or 
classification), the quantity of interest being estimated (ATE or ATT), whether the model 
uses an L2 penalty, the activation function used in the model's outcome predictors, whether 
the data is temporal, the validation metric used for cross validation to find the best 
number of neurons, the number of neurons used in the ELMs used by the estimator, the number 
of neurons used in the ELM used to learn a mapping from number of neurons to validation 
loss during cross validation, the causal effect, standard error, and p-value.
```julia
summarize(g_computer)
```

## Step 4: Validate the Model
We can validate the model by examining the plausibility that the main assumptions of causal 
inference, counterfactual consistency, exchangeability, and positivity, hold. It should be 
noted that consistency and exchangeability are not directly testable, so instead, these 
tests do not provide definitive evidence of a violation of these assumptions. To probe the 
counterfactual consistency assumption, we assume there were multiple levels of treatments 
and find them by binning the dependent vairable for treated observations using Jenks breaks. 
The optimal number of breaks between 2 and num_treatments is found using the elbow method. 
Using these hypothesized treatment assignemnts, this method compares the MSE of linear 
regressions using the observed and hypothesized treatments. If the counterfactual 
consistency assumption holds then the difference between the MSE with hypothesized 
treatments and the observed treatments should be positive because the hypothesized 
treatments should not provide useful information. If it is negative, that indicates there 
was more useful information provided by the hypothesized treatments than the observed 
treatments or that there is an unobserved confounder. Next, this methods tests the model's 
sensitivity to a violation of the exchangeability assumption by calculating the E-value, 
which is the minimum strength of association, on the risk ratio scale, that an unobserved 
confounder would need to have with the treatment and outcome variable to fully explain away 
the estimated effect. Thus, higher E-values imply the model is more robust to a violation of 
the exchangeability assumption. Finally, this method tests the positivity assumption by 
estimating propensity scores. Rows in the matrix are levels of covariates that have a zero 
probability of treatment. If the matrix is empty, none of the observations have an estimated 
zero probability of treatment, which implies the positivity assumption is satisfied.

One can also specify the maxium number of possible treatments to consider for the causal 
consistency assumption and the minimum and maximum probabilities of treatment for the 
positivity assumption with the num_treatments, min, and max keyword arguments.

!!! danger
    Obtaining correct estimates is dependent on meeting the assumptions for G-computation. 
    If the assumptions are not met then any estimates may be biased and lead to incorrect 
    conclusions.

!!! note
    For a thorough review of casual inference assumptions see:

        Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
        Francis, 2024. 

    For more information on the E-value test see:
    
        VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
        introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.


```julia
validate(g_computer)
```


# Double Machine Learning
Double machine learning, also called debiased or orthogonalized machine learning, enables
estimating causal effects when the dimensionality of the covariates is too high for linear 
regression or the model does not assume a parametric form. In other words, when the 
relathionship between the treatment or covariates and outcome is nonlinear and we do not 
know the functional form. 

!!! note
    For more information see:

    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.


## Step 1: Initialize a Model
The DoubleMachineLearning constructor takes at least three arguments, an array of 
covariates, a treatment vector, and an outcome vector. 

You can also specify the following options: whether the treatment vector is categorical ie 
not continuous and containing more than two classes, whether to use L2 regularization, the 
activation function, the validation metric to use when searching for the best number of 
neurons, the minimum and maximum number of neurons to consider, the number of folds to use 
for cross validation, the number of iterations to perform cross validation, and the number 
of neurons to use in the ELM used to learn the function from number of neurons to validation 
loss. These arguments are specified with the following keyword arguments: t_cat, 
regularized, activation, validation_metric, min_neurons, max_neurons, folds, iterations, and 
approximator_neurons.
```julia
# Create some data with a binary treatment
X, T, Y =  rand(100, 5), [rand()<0.4 for i in 1:100], rand(100)

# We could also use DataFrames
# using DataFrames
# X = DataFrame(x1=rand(100), x2=rand(100), x3=rand(100), x4=rand(100), x5=rand(100))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:100]), DataFrame(y=rand(100))

dml = DoubleMachineLearning(X, T, Y)
```

## Step 2: Estimate the Causal Effect
To estimate the causal effect, we call estimatecausaleffect! on the model above.
```julia
# we could also estimate the ATT by passing quantity_of_interest="ATT"
estimate_causal_effect!(dml)
```

# Get a Summary
We can get a summary that includes a p-value and standard error estimated via asymptotic 
randomization inference by passing our model to the summarize method.

Calling the summarize methodd returns a dictionary with the estimator's task (regression or 
classification), the quantity of interest being estimated (ATE), whether the model uses an 
L2 penalty (always true for DML), the activation function used in the model's outcome 
predictors, whether the data is temporal (always false for DML), the validation metric used 
for cross validation to find the best number of neurons, the number of neurons used in the 
ELMs used by the estimator, the number of neurons used in the ELM used to learn a mapping 
from number of neurons to validation loss during cross validation, the causal effect, 
standard error, and p-value.
```julia
# Can also use the British spelling
# summarise(dml)

summarize(dml)
```

## Step 4: Validate the Model
We can validate the model by examining the plausibility that the main assumptions of causal 
inference, counterfactual consistency, exchangeability, and positivity, hold. It should be 
noted that consistency and exchangeability are not directly testable, so instead, these 
tests do not provide definitive evidence of a violation of these assumptions. To probe the 
counterfactual consistency assumption, we assume there were multiple levels of treatments 
and find them by binning the dependent vairable for treated observations using Jenks breaks. 
The optimal number of breaks between 2 and num_treatments is found using the elbow method. 
Using these hypothesized treatment assignemnts, this method compares the MSE of linear 
regressions using the observed and hypothesized treatments. If the counterfactual 
consistency assumption holds then the difference between the MSE with hypothesized 
treatments and the observed treatments should be positive because the hypothesized 
treatments should not provide useful information. If it is negative, that indicates there 
was more useful information provided by the hypothesized treatments than the observed 
treatments or that there is an unobserved confounder. Next, this methods tests the model's 
sensitivity to a violation of the exchangeability assumption by calculating the E-value, 
which is the minimum strength of association, on the risk ratio scale, that an unobserved 
confounder would need to have with the treatment and outcome variable to fully explain away 
the estimated effect. Thus, higher E-values imply the model is more robust to a violation of 
the exchangeability assumption. Finally, this method tests the positivity assumption by 
estimating propensity scores. Rows in the matrix are levels of covariates that have a zero 
or near zero probability of treatment. If the matrix is empty, none of the observations have 
an estimated zero probability of treatment, which implies the positivity assumption is 
satisfied.

One can also specify the maxium number of possible treatments to consider for the causal 
consistency assumption and the minimum and maximum probabilities of treatment for the 
positivity assumption with the num_treatments, min, and max keyword arguments.

!!! danger
    Obtaining correct estimates is dependent on meeting the assumptions for double machine 
    learning. If the assumptions are not met then any estimates may be biased and lead to 
    incorrect conclusions.

!!! note
    For a thorough review of casual inference assumptions see:

        Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
        Francis, 2024. 

    For more information on the E-value test see:
  
        VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
        introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.


```julia
validate(g_computer)
```


# Metalearners
Instead of knowing the average cuasal effect, we might want to know which units benefit and 
which units lose by being exposed to a treatment. For example, a cash transfer program might 
motivate some people to work harder and incentivize others to work less. Thus, we might want 
to know how the cash transfer program affects individuals instead of it average affect on 
the population. To do so, we can use metalearners. Depending on the scenario, we may want to 
use an S-learner, a T-learner, an X-learner, or an R-learner. The basic steps to use all 
three metalearners are below. The difference between the metalearners is how they estimate 
the CATE and what types of variables they can handle. In the case of S, T, and X learners, 
they can only handle binary treatments. On the other hand, R-learners can handle binary, 
categorical, count, or continuous treatments but only supports continuous outcomes.

!!! note
    For a deeper dive on S-learning, T-learning, and X-learning see:
    
        Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for 
        estimating heterogeneous treatment effects using machine learning." Proceedings of the 
        national academy of sciences 116, no. 10 (2019): 4156-4165.


    To learn more about R-learning see:
    
        Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment 
        effects." Biometrika 108, no. 2 (2021): 299-319.

# Initialize a Metalearner
S-learners, T-learners, and X-learners all take at least three arguments: an array of 
covariates, a vector of outcomes, and a vector of treatment statuses. Additional options can 
be specified for each type of metalearner using its keyword arguments.
```julia
# Generate data to use
X, Y, T =  rand(1000, 5), rand(1000), [rand()<0.4 for i in 1:1000]

# We could also use DataFrames
# using DataFrames
# X = DataFrame(x1=rand(1000), x2=rand(1000), x3=rand(1000), x4=rand(1000), x5=rand(1000))
# T, Y = DataFrame(t=[rand()<0.4 for i in 1:1000]), DataFrame(y=rand(1000))

s_learner = SLearner(X, Y, T)
t_learner = TLearner(X, Y, T)
x_learner = XLearner(X, Y, T)
r_learner = RLearner(X, Y, T)
```

# Estimate the CATE
We can estimate the CATE for all the models by passing them to estimate_causal_effect!.
```julia
estimate_causal_effect!(s_learner)
estimate_causal_effect!(t_learner)
estimate_causal_effect!(x_learner)
estimate_causal_effect!(r_learner)
```

# Get a Summary
We can get a summary of the models that includes p0values and standard errors for the 
average treatment effect by passing the models to the summarize method.

Calling the summarize methodd returns a dictionary with the estimator's task (regression or 
classification), the quantity of interest being estimated (CATE), whether the model 
uses an L2 penalty, the activation function used in the model's outcome predictors, whether 
the data is temporal, the validation metric used for cross validation to find the best 
number of neurons, the number of neurons used in the ELMs used by the estimator, the number 
of neurons used in the ELM used to learn a mapping from number of neurons to validation 
loss during cross validation, the causal effect, standard error, and p-value for the ATE.
```julia
summarize(s_learner)
summarize(t_learner)
summarize(x_learner)
summarize(r_learner)
```

## Step 4: Validate the Model
We can validate the model by examining the plausibility that the main assumptions of causal 
inference, counterfactual consistency, exchangeability, and positivity, hold. It should be 
noted that consistency and exchangeability are not directly testable, so instead, these 
tests do not provide definitive evidence of a violation of these assumptions. To probe the 
counterfactual consistency assumption, we assume there were multiple levels of treatments 
and find them by binning the dependent vairable for treated observations using Jenks breaks. 
The optimal number of breaks between 2 and num_treatments is found using the elbow method. 
Using these hypothesized treatment assignemnts, this method compares the MSE of linear 
regressions using the observed and hypothesized treatments. If the counterfactual 
consistency assumption holds then the difference between the MSE with hypothesized 
treatments and the observed treatments should be positive because the hypothesized 
treatments should not provide useful information. If it is negative, that indicates there 
was more useful information provided by the hypothesized treatments than the observed 
treatments or that there is an unobserved confounder. Next, this methods tests the model's 
sensitivity to a violation of the exchangeability assumption by calculating the E-value, 
which is the minimum strength of association, on the risk ratio scale, that an unobserved 
confounder would need to have with the treatment and outcome variable to fully explain away 
the estimated effect. Thus, higher E-values imply the model is more robust to a violation of 
the exchangeability assumption. Finally, this method tests the positivity assumption by 
estimating propensity scores. Rows in the matrix are levels of covariates that have a zero 
probability of treatment. If the matrix is empty, none of the observations have an estimated 
zero probability of treatment, which implies the positivity assumption is satisfied.

One can also specify the maxium number of possible treatments to consider for the causal 
consistency assumption and the minimum and maximum probabilities of treatment for the 
positivity assumption with the num_treatments, min, and max keyword arguments.

!!! danger
    Obtaining correct estimates is dependent on meeting the assumptions for interrupted time 
    series estimation. If the assumptions are not met then any estimates may be biased and 
    lead to incorrect conclusions.

!!! note
    For a thorough review of casual inference assumptions see:

        Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
        Francis, 2024. 

    For more information on the E-value test see:

        VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
        introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.

```julia
validate(s_learner)
validate(t_learner)
validate(x_learner)
validate(r_learner)
```