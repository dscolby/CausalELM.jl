![CausalELM logo] (https://github.com/dscolby/dscolby.github.io/blob/main/causalelm-high-resolution-logo-black-on-transparent-background.png)


[![Build Status](https://github.com/dscolby/CausalELM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dscolby/CausalELM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dscolby/CausalELM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dscolby/CausalELM.jl)

CausalELM is a project to learn about how machine learning methods can be used for causal inference, gain a better understanding of Extreme Learning Machines, and build proficiency with Julia. The intutition for using ELM for causal inference is that in event studies and similar research designs, one must make a tradeoff between accuracy vs complexity and computing power when deciding on a model to estimate the counterfactual for the post-event/post-treatment period. ELMs represent a good balance because they have been shown to be very accurate while not having to rely on backpropagation for training. Furthermore, since ELMs solve the least squares problem, their estimates (with the design matrix that has been transformed by an activation function) represent the best linear unbiased estimate and are asymptotically consistent. While this is a research project, it is functional and has passed a large number of unit tests, so it should be mostly reliable. As the goal is to be able to use this package for causal inference I have added a couple structs and methods for that goal. First, the predictcounterfactual method uses a trained ExtremeLearningMachine struct to estimate the counterfactual, or post-event observations, using data that can be passed in by the user. Second, there is a method to perform a placebo test whereby counterfactuals are predicted using both pre-event and post-event covariates. If these predictions are significantly different there is likely a confounder or some other issue with the study design. Finally, to address the issue of multicollinearity the RegularizedExtremeLearner struct and associated methods fit and predict ELMs with L2 regularization, as proposed by Li and Niu (2013). My intent for this package is for it to be lightweight, so I wrote it such that it does not have any dependencies.

## Suggestions
If you have any suggestions, feature requests, or find a bug feel free to report it on the [issue tracker].

[issue tracker]: https://github.com/dscolby/CausalELM.jl/issues


