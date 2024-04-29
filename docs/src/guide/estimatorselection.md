# Deciding Which Estimator to Use
Which model you should use depends on what you are trying to model and the type of data you 
have. The table below can serve as a useful reference when deciding which model to use for a 
given dataset and causal question.

| Model                            | Struct                | Causal Estimands                 | Supported Treatment Types | Supported Outcome Types                  |
|----------------------------------|-----------------------|----------------------------------|---------------------------|------------------------------------------|
| Interrupted Time Series Analysis | InterruptedTimeSeries | ATE, Cumulative Treatment Effect | Binary                   | Continuous, Count**, Time to Event         |
| G-computation                    | GComputation          | ATE, ATT, ITT                    | Binary                   | Binary*,Continuous, Time to Event, Count** |
| Double Machine Learning          | DoubleMachineLearning | ATE                              | Binary*, Count**, Continuous | Binary*, Count**, Continuous, Time to Event |
| S-learning                       | SLearner              | CATE                             | Binary                    | Binary*, Continuous, Time to Event, Count** |
| T-learning                       | TLearner              | CATE                             | Binary                    | Binary*, Continuous, Count**, Time to Event |
| X-learning                       | XLearner              | CATE                             | Binary*                    | Binary*, Continuous, Count**, Time to Event |
| R-learning                       | RLearner              | CATE                             | Binary*, Count**, Continuous | Binary*, Count**, Continuous, Time to Event |
| Doubly Robust Estimation         | DoublyRobustLearner   | CATE                             | Binary                    | Binary*, Continuous, Count**, Time to Event |

*Models that use propensity scores or predict binary treatment assignment may, on very rare occasions, return values outside of [0, 1]. In that case, values are clipped to be between 0.0000001 and 0.9999999.

**Similar to other packages, predictions of count variables is treated as a continuous regression task.