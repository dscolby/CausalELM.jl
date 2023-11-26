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

CausalELM enables Estimation of causal quantities of interest in research designs where a 
counterfactual must be predicted and compared to the observed outcomes. More specifically, 
CausalELM provides a simple API to execute interupted time series analysis, G-Computation, 
and double machine learning as well as estimation of the CATE via S-Learning, T-Learning, 
and X-Learning. Once a causal model has beeen estimated, CausalELM's summarize method 
provides basic information about the model as well as a p-value and standard error estimated 
with approximate randomization inference. One can then validate causal modeling assumptions 
for any model with a single call to the validate method. In all of these implementations, 
CausalELM predicts the counterfactuals using an Extreme Learning Machine that includes an L2 
penalty by default. In this context, ELMs strike a good balance between prediction accuracy, 
generalization, ease of implementation, speed, and interpretability. 

### Features
*   Simple interface enables estimating causal effects in only a few lines of code
*   Analytically derived L2 penalty reduces cross validation time and multicollinearity
*   Fast automatic cross validation works with longitudinal, panel, and time series data
*   Includes 13 activation functions and allows user-defined activation functions
*   Single interface for continous, binary, and categorical outcome variables
*   Estimation of p-values and standard errors via asymptotic randomization inference
*   No dependencies outside of the Julia standard library
*   Validate causal modeling assumptiions with one line of code

### What's New?
*   All functions and methods converted to snake case
*   Randomization inference for interrupted time series randomizes all indices
*   Implemented validate method to probe assumptions for all estimators and metalearners
*   Reimplemented cross validation for temporal data
*   Fixed issue related to recoding variables to calculate validation metrics for cross validation

### Comparison with Other Packages
Other packages, mainly EconML, DoWhy, and CausalML, have similar funcitonality. Beides being 
written in Julia rather than Python, the main differences between CausalELM and these 
libraries are:

*   CausalELM uses extreme learning machines rather than tree-based or deep learners
*   CausalELM performs cross validation during training
*   CausalELM performs inference via asymptotic randomization inference rather than 
    bootstrapping
*   CausalELM does not require you to instantiate a model and pass it into a separate class 
    or struct for training
*   CausalELM creates train/test splits automatically
*   CausalELM does not have external dependencies: all the functions it uses are in the 
    Julia standard library

### Installation
CausalELM requires Julia version 1.7 or greater and can be installed from the REPL as shown 
below. 
```julia
using Pkg 
Pkg.add("CausalELM")
```
