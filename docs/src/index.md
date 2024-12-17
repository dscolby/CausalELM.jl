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

CausalELM provides easy-to-use implementations of modern causal inference methods. While
CausalELM implements a variety of estimators, they all have one thing in common—the use of 
machine learning models to flexibly estimate causal effects. This is where the ELM in 
CausalELM comes from—the machine learning model underlying all the estimators is an extreme 
learning machine (ELM). ELMs are a simple neural network that use randomized weights and 
offer a good tradeoff between learning non-linear dependencies and simplicity. Furthermore, 
CausalELM implements bagged ensembles of ELMs to reduce the variance resulting from 
randomized weights.

## Estimators
CausalELM implements estimators for aggreate e.g. average treatment effect (ATE) and 
individualized e.g. conditional average treatment effect (CATE) quantities of interest.

### Estimators for Aggregate Effects
*   Interrupted Time Series Estimator
*   G-computation
*   Double machine Learning

### Individualized Treatment Effect (CATE) Estimators
*   S-learner
*   T-learner
*   X-learner
*   R-learner
*   Doubly Robust Estimator

## Features
*   Estimate a causal effect, get a summary, and validate assumptions in just four lines of code
*   Enables using the same structs for regression and classification
*   Includes 13 activation functions and allows user-defined activation functions
*   Most inference and validation tests do not assume functional or distributional forms
*   Implements the latest techniques from statistics, econometrics, and biostatistics
*   Works out of the box with AbstractArrays or any data structure that implements the Tables.jl interface
*   Works with CuArrays, ROCArrays, and any other GPU-specific arrays that are AbstractArrays
*   CausalELM is lightweight—its only dependency is Tables.jl
*   Codebase is high-quality, well tested, and regularly updated

## What's New?
*   Includes support for GPU-specific arrays and data structures that implement the Tables.jl API
*   Only performs randomization inference when the inference argument is set to true in summarize methods
*   Summaries support calculating marginal effects and confidence intervals
*   Randomization inference now uses multithreading
*   CausalELM was presented at JuliaCon 2024 in Eindhoven
*   Refactored code to be easier to extend and understand

## What makes CausalELM different?
Other packages, mainly EconML, DoWhy, CausalAI, and CausalML, have similar funcitonality. 
Beides being written in Julia rather than Python, the main differences between CausalELM and 
these libraries are:
*   Simplicity is core to casualELM's design philosophy. CausalELM only uses one type of
    machine learning model, extreme learning machines (with bagging) and does not require 
    you to import any other packages or initialize machine learning models, pass machine 
    learning structs to CausalELM's estimators, convert dataframes or arrays to a special 
    type, or one hot encode categorical treatments. By trading a little bit of flexibility 
    for a simpler API, all of CausalELM's functionality can be used with just four lines of 
    code.
*   As part of this design principle, CausalELM's estimators decide whether to use regression 
    or classification based on the type of outcome variable. This is in contrast to most 
    machine learning packages, which have separate classes or structs fro regressors and 
    classifiers of the same model.
*   CausalELM's validate method, which is specific to each estimator, allows you to validate 
    or test the sentitivity of an estimator to possible violations of identifying assumptions.
*   Unlike packages that do not allow you to estimate p-values and standard errors, use 
    bootstrapping to estimate them, or use incorrect hypothesis tests, all of CausalELM's 
    estimators provide p-values and standard errors generated via approximate randomization 
    inference. 
*   CausalELM strives to be lightweight while still being powerful and therefore does not 
    have external dependencies: all the functions it uses are in the Julia standard library
    with the exception of model constructors, which use Tables.matrix to ensure integration 
    with a wide variety of data structures.
*   The other packages and many others mostly use techniques from one field. Instead, 
    CausalELM incorporates a hodgepodge of ideas from statistics, machine learning, 
    econometrics, and biostatistics.
*   CausalELM doesn't use any unnecessary abstractions. The only structs are the actual 
    models. Estimated effects are returned as arrays, summaries are returned in a dictionary, 
    and the results of validating an estimator are returned as tuples. This is in contrast 
    to other packages that utilize separate structs (classes) for summaries and inference 
    results.

## Installation
CausalELM requires Julia version 1.8 or greater and can be installed from the REPL as shown 
below. 
```julia
using Pkg 
Pkg.add("CausalELM")
```
## More Information
For a more interactive overview, see our JuliaCon 2024 talk[here](https://www.youtube.com/watch?v=hh_cyj8feu8&t=26s)
