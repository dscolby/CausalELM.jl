# CausalELM.jl
*Lightweight implementation of extreme learning machines for event studies and other time series based causal inference tasks.*
## Package Features
- Predict the counterfactuals in event studies using Extreme ELarning Machines
- Reduce multicollinearity via L2 Regularization
- Conduct placebo tests
- 13 activation functions and user defined activation functions possible
## Function Documentation
```@docs
binarystep
```
```@docs
σ
```
```@docs
tanh
```
```@docs
relu
```
```@docs
leakyrelu
```
```@docs
swish
```
```@docs
softmax
```
```@docs
softplus
```
```@docs
gelu
```
```@docs
gaussian
```
```@docs
hardtanh
```
```@docs
elish
```
```@docs
fourier
```
```@docs
ExtremeLearner
```
```@docs
RegularizedExtremeLearner
```
```@docs
fit!
```
```@docs
predict
```
```@docs
predictcounterfactual!
```
```@docs
placebotest
```
