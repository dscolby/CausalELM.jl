# Deciding Which Estimator to Use
Which model you should use depends on what you are trying to model and the type of data you 
have. The table below can serve as a useful reference when deciding which model to use for a 
given dataset and causal question.

| Model                            | Struct                | Causal Estimands                 | Supported Treatment Types | Supported Outcome Types                  |
|----------------------------------|-----------------------|----------------------------------|---------------------------|------------------------------------------|
| Interrupted Time Series Analysis | InterruptedTimeSeries | ATE, Cumulative Treatment Effect | Binary                   | Continuous, Count[^2], Time to Event         |
| G-computation                    | GComputation          | ATE, ATT, ITT                    | Binary                   | Binary[^1],Continuous, Time to Event, Count[^2] |
| Double Machine Learning          | DoubleMachineLearning | ATE                              | Binary[^1], Count[^2], Continuous | Binary[^1], Count[^2], Continuous, Time to Event |
| S-learning                       | SLearner              | CATE                             | Binary                    | Binary[^1], Continuous, Time to Event, Count[^2] |
| T-learning                       | TLearner              | CATE                             | Binary                    | Binary[^1], Continuous, Count[^2], Time to Event |
| X-learning                       | XLearner              | CATE                             | Binary[^1]                    | Binary[^1], Continuous, Count[^2], Time to Event |
| R-learning                       | RLearner              | CATE                             | Binary[^1], Count[^2], Continuous | Binary[^1], Count[^2], Continuous, Time to Event |
| Doubly Robust Estimation         | DoublyRobustLearner   | CATE                             | Binary                    | Binary[^1], Continuous, Count[^2], Time to Event |

[^1]: Models that use propensity scores or predict binary treatment assignment may, on very rare occasions, return values outside of [0, 1]. In that case, values are clipped to be between 0.0000001 and 0.9999999.

[^2]: Similar to other packages, predictions of count variables is treated as a continuous regression task.