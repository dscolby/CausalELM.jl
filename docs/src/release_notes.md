# Release Notes
Below is a list of changes since causalELM was first released.

## v0.5.0
### Added
    *   Constructors for estimators taht accept dataframes from DataFrames.jl
### Changed
    *   Estimators can handle any array whose values are <:Real
    *   Estimator constructors are now called with model(X, T, Y) instead of model(X, Y, T)
    *   Removed excess type constraints for many methods
    *   Vectorized a few for loops
    *   Increased test coverage

## v0.4.0
### Added
    *   R-learning
    *   Softmax function for arrays
### Changed
    *   Moved all types and methods under the main module
    *   Decreased size of function definitions
    *   SLearner has a G-computation field that does the heavy lifting for S-learning
    *   Removed excess fields from estimator structs
### Fixed
    *   Changed the incorrect name of DoublyRobustEstimation struct to DoubleMachineLearning
    *   Caclulation of risk ratios and E-values
    *   Calculation of validation metrics for multiclass classification
    *   Calculation of output weights for L2 regularized extreme learning machines

## v0.3.0
### Added
    *   Splitting of temporal data for cross validation
    *   Methods to validate/test senstivity to violations of identifying assumptions
### Changed
    *   Converted all functions and methods to snake case
    *   Randomization inference for interrupted time series randomizes all the indices
### Fixed
    *   Issue related to recoding variables to calculate validation metrics for cross validation

## v0.2.1
### Added
    *   Cross fitting to the doubly robust estimator

## v0.2.0
### Added
    *   Calculation of p-values and standard errors via randomization inference
### Changed
    *   Divided package into modules

## v0.1.0
### Added
    *   Event study, g-computation, and doubly robust estimators
    *   S-learning, T-learning, and X-learning
    *   Model summarization methods