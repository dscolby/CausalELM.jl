```@raw html
<div style="width:100%; height:15px;;
        border-radius:6px;text-align:center;
        color:#1e1e20">
    <a class="github-button" href="https://github.com/dscolby/CausalELM.jl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star dscolby/CausalELM.jl on GitHub" style="margin:auto">Star</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
</div>
```

```@meta
CurrentModule = CausalELM
```

# Overview

causalELM enables Estimation of causal quantities of interest in research designs where a 
counterfactual must be predicted and compared to the observed outcomes. More specifically, 
CausalELM provides a simple API to execute interupted time series analysis, G-Computation, 
and double machine learning as well as estimation of the CATE via S-Learning, T-Learning, 
X-Learning, and R-learning. Once a causal model has beeen estimated, causalELM's summarize 
method provides basic information about the model as well as a p-value and standard error 
estimated with approximate randomization inference. One can then validate causal modeling 
assumptions for any model with a single call to the validate method. In all of these 
implementations, causalELM predicts the counterfactuals using an Extreme Learning Machine 
that includes an L2 penalty by default. In this context, ELMs strike a good balance between 
prediction accuracy, generalization, ease of implementation, speed, and interpretability. 

### Features
*   Simple interface enables estimating causal effects in only a few lines of code
*   Analytically derived L2 penalty reduces cross validation time and multicollinearity
*   Fast automatic cross validation works with longitudinal, panel, and time series data
*   Includes 13 activation functions and allows user-defined activation functions
*   Single interface for continous, binary, and categorical outcome variables
*   Estimation of p-values and standard errors via asymptotic randomization inference
*   No dependencies outside of the Julia standard library
*   Validate causal modeling assumptions with one line of code
*   Non-parametric randomization (permutation) inference-based p-values for all models

### What's New?
*   Added support for dataframes
*   Permutation of continuous treatments draws from a continuous, instead of discrete uniform distribution
    during randomization inference
*   Estimators can handle any array whose values are <:Real
*   Estimator constructors are now called with model(X, T, Y) instead of model(X, Y, T)
*   Improved documentation
*   causalELM has a new logo

### What makes causalELM different?
Other packages, mainly EconML, DoWhy, CausalAI, and CausalML, have similar funcitonality. 
Beides being written in Julia rather than Python, the main differences between CausalELM and 
these libraries are:
*   Simplicity is core to casualELM's design philosophy. causalELM only uses one type of
    machine learning model, extreme learning machines (with optional L2 regularization) and 
    does not require you to import any other packages or initialize machine learning models, 
    pass machine learning structs to causalELM's estimators, convert dataframes or arrays to 
    a special type, or one hot encode categorical treatments. By trading a little bit of 
    flexibility for a simple API, all of causalELM's functionality can be used with just 
    four lines of code.
*   As part of this design principle, causalELM's estimators handle all of the work in 
    finding the best number of neurons during estimation. They create folds or rolling 
    rolling for time series data and use an extreme learning machine interpolator to find 
    the best number of neurons.
*   causalELM's validate method, which is specific to each estimator, allows you to validate 
    or test the sentitivity of an estimator to possible violations of identifying assumptions.
*   Unlike packages that do not allow you to estimate p-values and standard errors, use 
    bootstrapping to estimate them, or use incorrect hypothesis tests, all of causalELM's 
    estimators provide p-values and standard errors generated via approximate randomization 
    inference. 
*   causalELM strives to be lightweight while still being powerful and therefore does not 
    have external dependencies: all the functions it uses are in the Julia standard library.

### Installation
causalELM requires Julia version 1.7 or greater and can be installed from the REPL as shown 
below. 
```julia
using Pkg 
Pkg.add("CausalELM")
```
