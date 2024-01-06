# Deciding Which Model to Use
Which model you should use depends on what you are trying to model and the type of data you 
have. The table below can serve as a useful reference when deciding which model to use for a 
given dataset and causal question.

| Model                            | Struct                | Causal Estimands                 | Supported Treatment Types              | Supported Outcome Types   |
|----------------------------------|-----------------------|----------------------------------|----------------------------------------|---------------------------|
| Interrupted Time Series Analysis | InterruptedTimeSeries | ATE, Cumulative Treatment Effect | Binary                                 | Binary, Continuous        |
| G-computation                    | GComputation          | ATE, ATT, ITT                    | Binary                                 | Binary, Continuous        |
| Double Machine Learning          | DoubleMachineLearning | ATE                              | Binary, Count, Categorical, Continuous | Continuous                |
| S-learning                       | SLearner              | CATE                             | Binary                                 | Binary, Continuous, Count |
| T-learning                       | TLearner              | CATE                             | Binary                                 | Binary, Continuous        |
| X-learning                       | XLearner              | CATE                             | Binary                                 | Binary, Continuous, Count |
| R-learning                       | RLearner              | CATE                             | Binary, Count, Categorical, Continuous | Continuous                |