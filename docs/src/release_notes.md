# Release Notes
These release notes adhere to the [keep a changelog](https://keepachangelog.com/en/1.0.0/) format. Below is a list of changes since CausalELM was first released.

## Version [v0.6.0](https://github.com/dscolby/CausalELM.jl/releases/tag/v0.6.0) - 2024-03-23
### Added
*   Double machine learning and R-learning now support binary and categorical treatments and outcomes

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