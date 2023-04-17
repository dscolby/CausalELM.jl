```@meta
CurrentModule = CausalELM
```

# Overview

CausalELM enables Estimation of causal quantities of interest in research designs where a 
counterfactual must be predicted and compared to the observed outcomes. More specifically, 
CausalELM provides structs and methods to execute event study designs (interupted time 
series analysis), G-Computation, and doubly robust estimation as well as estimation of the 
CATE via S-Learning, T-Learning, and X-Learning. Once a causal model has beeen estimated, 
CausalELM's summarize method provides basic information about the model as well as a p-value 
and standard error estimated with approximate randomization inference. In all of these 
implementations, CausalELM predicts the counterfactuals using an Extreme Learning Machine 
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
CausalELM reuires Julia version 1.7 or greater and can be installed from the REPL as shown 
below.
```julia
using Pkg
Pkg.add("CausalELM")
```
