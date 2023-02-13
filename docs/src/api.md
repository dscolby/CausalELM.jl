# API
This is a reference for advanced users or those who wish to contribute.

```@index
```

## CausalELM
```@docs
CausalELM
```

## Activation Functions
```@docs
CausalELM.ActivationFunctions
CausalELM.ActivationFunctions.binarystep
CausalELM.ActivationFunctions.σ
CausalELM.ActivationFunctions.tanh
CausalELM.ActivationFunctions.relu
CausalELM.ActivationFunctions.leakyrelu
CausalELM.ActivationFunctions.swish
CausalELM.ActivationFunctions.softmax
CausalELM.ActivationFunctions.softplus
CausalELM.ActivationFunctions.gelu
CausalELM.ActivationFunctions.gaussian
CausalELM.ActivationFunctions.hardtanh
CausalELM.ActivationFunctions.elish
CausalELM.ActivationFunctions.fourier
```

## Cross Valdiation
```@docs
CausalELM.CrossValidation
CausalELM.CrossValidation.recode
CausalELM.CrossValidation.traintest
CausalELM.CrossValidation.validate
CausalELM.CrossValidation.crossvalidate
CausalELM.CrossValidation.bestsize
```

## ATE/ATE/ITT Estimation
```@docs
CausalELM.Estimators
CausalELM.Estimators.EventStudy
CausalELM.Estimators.GComputation
CausalELM.Estimators.DoublyRobust
CausalELM.Estimators.estimatecausaleffect!
CausalELM.Estimators.summarize
```

## CATE Estimation
```@docs
CausalELM.Metalearners
CausalELM.Metalearners.SLearner
CausalELM.Metalearners.TLearner
CausalELM.Metalearners.XLearner
```

## Validation Metrics
```@docs
CausalELM.Metrics
CausalELM.Metrics.mse
CausalELM.Metrics.mae
CausalELM.Metrics.accuracy
CausalELM.Metrics.precision
CausalELM.Metrics.recall
CausalELM.Metrics.F1
```

## Base Models
```@docs
CausalELM.Models
CausalELM.Models.ExtremeLearningMachine
CausalELM.Models.ExtremeLearner
CausalELM.Models.RegularizedExtremeLearner
CausalELM.Models.fit!
CausalELM.Models.predict
CausalELM.Models.predictcounterfactual!
CausalELM.Models.placebotest
```