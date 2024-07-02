# Release Notes
These release notes adhere to the [keep a changelog](https://keepachangelog.com/en/1.0.0/) format. Below is a list of changes since CausalELM was first released.

## Version [v0.7.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.6.1) - 2024-06-22
### Added
*   Implemented bagged ensemble of extreme learning machines to use with estimators [#67](https://github.com/dscolby/CausalELM.jl/issues/67)
### Changed
*   Compute the number of neurons to use with log heuristic instead of cross validation [#62](https://github.com/dscolby/CausalELM.jl/issues/62)
*   Calculate probabilities as the average label predicted by the ensemble instead of clipping [#71](https://github.com/dscolby/CausalELM.jl/issues/71)
*   Made calculation of p-values and standard errors optional and not executed by default in summarize methods [#65](https://github.com/dscolby/CausalELM.jl/issues/65)
*   Removed redundant W argument for double machine learning, R-learning, and doubly robust estimation [#68](https://github.com/dscolby/CausalELM.jl/issues/68)
### Fixed
*   Applying the weight trick for R-learning [#70](https://github.com/dscolby/CausalELM.jl/issues/70)

## Version [v0.6.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.6.0) - 2024-06-15
### Added
*   Implemented doubly robust learner for CATE estimation [#31](https://github.com/dscolby/CausalELM.jl/issues/31)
*   Provided better explanations of supported treatment and outcome variable types in the docs [#41](https://github.com/dscolby/CausalELM.jl/issues/41)
*   Added support for specifying confounders, W, separate from covariates of interest, X, for double machine 
learning and doubly robust estimation [39](https://github.com/dscolby/CausalELM.jl/issues/39)
### Changed
*   Removed the estimate_causal_effect! call in the model constructor docstrings [#35](https://github.com/dscolby/CausalELM.jl/issues/35)
*   Standardized and improved docstrings and added doctests [#44](https://github.com/dscolby/CausalELM.jl/issues/44)
*   Counterfactual consistency now simulates outcomes that violate the counterfactual consistency assumption rather than 
binning of treatments and works with discrete or continuous treatments [#33](https://github.com/dscolby/CausalELM.jl/issues/33)
*   Refactored estimator and metalearner structs, constructors, and estimate_causal_effect! methods [#45](https://github.com/dscolby/CausalELM.jl/issues/45)
### Fixed
*   Clipped probabilities between 0 and 1 for estimators that use predictions of binary variables [#36](https://github.com/dscolby/CausalELM.jl/issues/36)
*   Fixed sample splitting and cross fitting procedure for doubly robust estimation [#42](https://github.com/dscolby/CausalELM.jl/issues/42)
*   Addressed numerical instability when finding the ridge penalty by replacing the previous ridge formula with 
generalized cross validation [#43](https://github.com/dscolby/CausalELM.jl/issues/43)
*   Uses the correct variable in the ommited predictor test for interrupted time series.
*   Uses correct range for p-values in interrupted time series validation tests.
*   Correctly subsets the data for ATT estimation in G-computation [#52](https://github.com/dscolby/CausalELM.jl/issues/52)

## Version [v0.5.1](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.5.1) - 2024-01-15
### Added
* More descriptive docstrings [#21](https://github.com/dscolby/CausalELM.jl/issues/21)
### Fixed
* Permutation of continuous treatments draws from a continuous, instead of discrete uniform distribution
  during randomization inference

## Version [v0.5.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.5.0) - 2024-01-13
### Added
*   Constructors for estimators taht accept dataframes from DataFrames.jl [#25](https://github.com/dscolby/CausalELM.jl/issues/25)
### Changed
*   Estimators can handle any array whose values are <:Real [#23](https://github.com/dscolby/CausalELM.jl/issues/23)
*   Estimator constructors are now called with model(X, T, Y) instead of model(X, Y, T)
*   Removed excess type constraints for many methods [#23](https://github.com/dscolby/CausalELM.jl/issues/23)
*   Vectorized a few for loops
*   Increased test coverage

## Version [v0.4.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.4.0) - 2024-01-06
### Added
*   R-learning
*   Softmax function for arrays
### Changed
*   Moved all types and methods under the main module
*   Decreased size of function definitions [#22](https://github.com/dscolby/CausalELM.jl/issues/15)
*   SLearner has a G-computation field that does the heavy lifting for S-learning
*   Removed excess fields from estimator structs
### Fixed
*   Changed the incorrect name of DoublyRobustEstimation struct to DoubleMachineLearning
*   Caclulation of risk ratios and E-values
*   Calculation of validation metrics for multiclass classification
*   Calculation of output weights for L2 regularized extreme learning machines

## Version [v0.3.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.3.0) - 2023-11-25
### Added
*   Splitting of temporal data for cross validation [18](https://github.com/dscolby/CausalELM.jl/issues/18)
*   Methods to validate/test senstivity to violations of identifying assumptions [#16](https://github.com/dscolby/CausalELM.jl/issues/16)
### Changed
*   Converted all functions and methods to snake case [#17](https://github.com/dscolby/CausalELM.jl/issues/17)
*   Randomization inference for interrupted time series randomizes all the indices [#15](https://github.com/dscolby/CausalELM.jl/issues/15)
### Fixed
*   Issue related to recoding variables to calculate validation metrics for cross validation

## Version [v0.2.1](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.2.1) - 2023-06-07
### Added
*   Cross fitting to the doubly robust estimator

## Version [v0.2.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.2.0) - 2023-04-16
### Added
*   Calculation of p-values and standard errors via randomization inference
### Changed
*   Divided package into modules

## Version [v0.1.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.1.0) - 2023-02-14
### Added
*   Event study, g-computation, and doubly robust estimators
*   S-learning, T-learning, and X-learning
*   Model summarization methods