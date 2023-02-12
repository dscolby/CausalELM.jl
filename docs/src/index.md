```@meta
CurrentModule = CausalELM
```

# CausalELM

Documentation for [CausalELM](https://github.com/dscolby/CausalELM.jl).

# Overview

CausalELM enables Estimation of causal quantities of interest in research designs where a 
counterfactual must be predicted and compared to the observed outcomes. More specifically, 
CausalELM provides structs and methods to execute event study designs (interupted time 
series analysis), G-Computation, and doubly robust estimation as well as estimation of the 
CATE via S-Learning, T-Learning, and X-Learning. In all of these implementations, CausalELM 
predicts the counterfactuals using an Extreme Learning Machine. In this context, ELMs strike
a good balance between prediction accuracy, generalization, ease of implementation, speed, 
and interpretability. In addition, CausalELM provides the ability to incorporate an L2 
penalty.

## Getting Started

### Installation
```julia
Pkg.add("CausalELM")
```

### Estimating Causal Effects
```julia

using CausalELM

# 1000 data points with 5 features in pre-event period
x0 = rand(1000, 5)

# Pre-event outcome
y0 = rand(1000)

# 200 data points in the post-event period
x1 = rand(200, 5)

# Pose-event outcome
y1 = rand(200)

# Instantiate an EventStudy struct
event_study = EventStudy(x0, y0, x1, y1)

estimatecausaleffect!(event_study)

summarize(event_study)
```

```@index
```

```@autodocs
Modules = [CausalELM]
```