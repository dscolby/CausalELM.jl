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

CausalELM leverages new techniques in machine learning and statistics to estimate individual 
and aggregate treatment effects in situations where traditional methods are unsatisfactory 
or infeasible. To enable this, CausalELM provides a simple API to initialize a model, 
estimate a causal effect, get a summary from the model, and test the robustness of the 
model. CausalELM includes estimators for interupted time series analysis, G-Computation, 
double machine learning, S-Learning, T-Learning, X-Learning, R-learning, and doubly robust 
estimation. Underlying all these estimators are bagged extreme learning machines. Extreme 
learning machines are a single layer feedfoward neural network that relies on randomized 
weights and least squares optimization, making them expressive, simple, and computationally 
efficient. Combining them with bagging reduces the variance due to their randomized weights 
and provides a form of regularization that does not have to be tuned through cross 
validation.These attributes make CausalELM a very simple and powerful package for estimating 
treatment effects.

### Features
*   Estimate a causal effect, get a summary, and validate assumptions in just four lines of code
*   Bagging improves performance and reduces variance without the need to tune a regularization parameter
*   Enables using the same structs for regression and classification
*   Includes 13 activation functions and allows user-defined activation functions
*   Most inference and validation tests do not assume functional or distributional forms
*   Implements the latest techniques form statistics, econometrics, and biostatistics
*   Works out of the box with arrays or any data structure that implements the Tables.jl interface
*   Codebase is high-quality, well tested, and regularly updated

### What's New?
*   Now includes doubly robust estimator for CATE estimation
*   Uses generalized cross validation with successive halving to find the best ridge penalty
*   Double machine learning, R-learning, and doubly robust estimators suppot specifying confounders and covariates of interest separately
*   Counterfactual consistency validation simulates outcomes that violate the assumption rather than the previous binning approach
*   Standardized and improved docstrings and added doctests
*   CausalELM talk has been accepted to JuliaCon 2024!

### What makes CausalELM different?
Other packages, mainly EconML, DoWhy, CausalAI, and CausalML, have similar funcitonality. 
Beides being written in Julia rather than Python, the main differences between CausalELM and 
these libraries are:
*   Simplicity is core to casualELM's design philosophy. CausalELM only uses one type of
    machine learning model, extreme learning machines (with optional L2 regularization) and 
    does not require you to import any other packages or initialize machine learning models, 
    pass machine learning structs to CausalELM's estimators, convert dataframes or arrays to 
    a special type, or one hot encode categorical treatments. By trading a little bit of 
    flexibility for a simpler API, all of CausalELM's functionality can be used with just 
    four lines of code.
*   As part of this design principle, CausalELM's estimators handle all of the work in 
    finding the best number of neurons during estimation. They use a simple log heuristic 
    for determining the number of neurons to use and automatically select the best ridge 
    penalty via generalized cross validation.
*   CausalELM's validate method, which is specific to each estimator, allows you to validate 
    or test the sentitivity of an estimator to possible violations of identifying assumptions.
*   Unlike packages that do not allow you to estimate p-values and standard errors, use 
    bootstrapping to estimate them, or use incorrect hypothesis tests, all of CausalELM's 
    estimators provide p-values and standard errors generated via approximate randomization 
    inference. 
*   CausalELM strives to be lightweight while still being powerful and therefore does not 
    have external dependencies: all the functions it uses are in the Julia standard library.
*   The other packages and many others mostly use techniques from one field. Instead, 
    CausalELM incorporates a hodgepodge of ideas from statistics, machine learning, 
    econometrics, and biostatistics.

### Installation
CausalELM requires Julia version 1.7 or greater and can be installed from the REPL as shown 
below. 
```julia
using Pkg 
Pkg.add("CausalELM")
```
