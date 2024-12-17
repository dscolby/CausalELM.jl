# CausalELM
```@docs
CausalELM.CausalELM
```

## Types
```@docs
InterruptedTimeSeries
GComputation
DoubleMachineLearning
SLearner
TLearner
XLearner
RLearner
DoublyRobustLearner
CausalELM.CausalEstimator
CausalELM.Metalearner
CausalELM.ExtremeLearner
CausalELM.ELMEnsemble
CausalELM.Nonbinary
CausalELM.Binary
CausalELM.Count
CausalELM.Continuous
```

## Activation Functions
```@docs
binary_step
σ
CausalELM.tanh
relu
leaky_relu
swish
softmax
softplus
gelu
gaussian
hard_tanh
elish
fourier
```

## Average Causal Effect Estimators
```@docs
CausalELM.g_formula!
CausalELM.predict_residuals
CausalELM.moving_average
```

## Metalearners
```@docs
CausalELM.doubly_robust_formula!
CausalELM.stage1!
CausalELM.stage2!
CausalELM.weight_trick
```

## Common Methods
```@docs
estimate_causal_effect!
```

## Inference
```@docs
summarize
CausalELM.generate_null_distribution
CausalELM.quantities_of_interest
CausalELM.confidence_interval
CausalELM.p_value_and_std_err
```

## Model Validation
```@docs
validate
CausalELM.covariate_independence
CausalELM.omitted_predictor
CausalELM.sup_wald
CausalELM.p_val
CausalELM.counterfactual_consistency
CausalELM.simulate_counterfactual_violations
CausalELM.exchangeability
CausalELM.e_value
CausalELM.binarize
CausalELM.risk_ratio
CausalELM.positivity
```

## Validation Metrics
```@docs
mse
mae
accuracy
CausalELM.precision
recall
F1
CausalELM.confusion_matrix
```

## Extreme Learning Machines
```@docs
CausalELM.fit!
CausalELM.predict
CausalELM.predict_counterfactual!
CausalELM.placebo_test
CausalELM.set_weights_biases
```

## Utility Functions
```@docs
CausalELM.var_type
CausalELM.mean
CausalELM.var
CausalELM.one_hot_encode
CausalELM.clip_if_binary
CausalELM.@model_config
CausalELM.@standard_input_data
CausalELM.generate_folds
CausalELM.convert_if_table
```
