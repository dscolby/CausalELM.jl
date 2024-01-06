# Double Machine Learning
Double machine learning, also called debiased or orthogonalized machine learning, enables
estimating causal effects when the dimensionality of the covariates is too high for linear 
regression or the model does not assume a parametric form. In other words, when the 
relathionship between the treatment or covariates and outcome is nonlinear and we do not 
know the functional form. 

For more information see:
    Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, 
    Whitney Newey, and James Robins. "Double/debiased machine learning for treatment and 
    structural parameters." (2018): C1-C68.

## # Step 1: Initialize a Model
The DoubleMachineLearning constructor takes at least three arguments, an array of 
covariates, an outcome vector, and a vector of treatment statuses. You can also specify the 
following options: whether the treatment vector is categorical ie not continuous and 
containing more than two classes, whether to use L2 regularization, the activation function, 
the validation metric to use when searching for the best number of neurons, the minimum and 
maximum number of neurons to consider, the number of folds to use for cross validation, the 
number of iterations to perform cross validation, and the number of neurons to use in the 
ELM used to learn the function from number of neurons to validation loss. These arguments 
are specified with the following keyword arguments: t_cat, regularized, activation, 
validation_metric, min_neurons, max_neurons, folds, iterations, and approximator_neurons.
```julia
# Create some data with a binary treatment
X, Y, T =  rand(100, 5), rand(100), [rand()<0.4 for i in 1:100]

dml = DoubleMachineLearning(X, Xₚ, Y, T)
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
```julia
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
probability of treatment. If the matrix is empty, none of the observations have an estimated 
zero probability of treatment, which implies the positivity assumption is satisfied.


For a thorough review of casual inference assumptions see:
    Hernan, Miguel A., and James M. Robins. Causal inference what if. Boca Raton: Taylor and 
    Francis, 2024. 

For more information on the E-value test see:
    VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: 
    introducing the E-value." Annals of internal medicine 167, no. 4 (2017): 268-274.
```julia
validate(g_computer)
```