<div align="center">
    <img src="https://github.com/dscolby/dscolby.github.io/blob/main/causalelm-high-resolution-logo-black-on-transparent-background.png">
</div>
<br>

<p align="center">
    <a href="https://github.com/dscolby/CausalELM.jl/actions">
        <img src="https://github.com/dscolby/CausalELM.jl/actions/workflows/CI.yml/badge.svg?branch=main"
            alt="Build Status">
    </a>
    <a href="https://app.codecov.io/gh/dscolby/CausalELM.jl/tree/main/src">
        <img src="https://codecov.io/gh/dscolby/CausalELM.jl/branch/main/graph/badge.svg"
         alt="Code Coverage">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yelllow"
            alt="License">
    </a>
    <a href="https://dscolby.github.io/CausalELM.jl/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="Documentation">
  </a>
</p>
<h2>TL;DR</h2>
<p>
CausalELM enables Estimation of causal quantities of interest in research designs where a 
counterfactual must be predicted and compared to the observed outcomes. More specifically, 
CausalELM provides structs and methods to execute event study designs (interupted time 
series analysis), G-Computation, and doubly robust estimation as well as estimation of the 
CATE via S-Learning, T-Learning, and X-Learning. In all of these implementations, CausalELM 
predicts the counterfactuals using an Extreme Learning Machine. In this context, ELMs strike
a good balance between prediction accuracy, generalization, ease of implementation, speed, 
and interpretability. In addition, CausalELM provides the ability to incorporate an L2 
penalty.
</p>

<h2>Extreme Learning Machines and Causal Inference</h2>
<p>
In some cases we would like to know the causal effect of some intervention but we do not 
have the counterfactual, making conventional methods of statistical analysis infeasible. 
However, it may still be possible to get an unbiased estimate of the causal effect (ATE, 
ATE, or ITT) by predicting the counterfactual and comparing it to the observed outcomes. 
This is the approach CausalELM takes to estimate event study designs (interrupted time 
series analysis), G-Computation, doubly robust estimation (DRE), and meatlearning via 
S-Learners, T-Learners, and X-Learners. In event study designs, we want to estimate the 
effect of some intervention on the outcome of a single unit that we observe during multiple 
time periods. For example, we might want to know how the announcement of a merger affected 
the price of Stock A. To do this, we need to know what the price of stock A would have been 
if the merger had not been announced, which we can predict with machine learning methods. 
Then, we can compare this predicted counterfactual to the observed price data to estimate 
the effect of the merger announcement. In another case, we might want to know the effect of 
medicine X on disease Y but the administration of X was not random and it might have also 
been administered at mulitiple time periods, which would produce biased estimates. To 
overcome this, G-computation models the observed data, uses the model to predict the 
outcomes if all patients recieved the treatment, and compares it to the predictions of the 
outcomes if none of the patients recieved the treatment. Doubly robust estimation (DRE) 
takes a similar approach but also models the treatment mechanism and uses it to adjust the 
initial estimates. The advantage of DRE is that only the model of the outcome OR the model 
of the treatment mechanism has to be correctly specified to yield unbiased estimates. 
Furthermore, we might be more interested in how much an individual can benefit from a 
treatment, as opposed to the average treatment effect. Depending on the characteristics of 
our data, we can use metalearning methods such as S-Learning, T-Learning, or X-Learning to 
do so. In all of these scenarios, how well we estimate the treatment effect depends on how 
well we can predict the counterfactual. The most common approaches to getting accurate 
predictions of the counterfactual are to use a super learner, which combines multiple 
machine learning methods and requires extensive tuning, or tree-based methods, which also 
have large hyperparameter spaces. In these cases hyperparameter tuning can be 
computationally expensive and requires researchers to make arbitrary decisions about how 
many and what models to use, how much regularization to apply, the depth of trees, 
interaction effects, etc. On the other hands, ELMs are able to achieve good accuracy on a 
variety of regression and classification tasks and generalize well. Moreover, they have a 
much smaller hyperparameter space to tune and are fast to train becasue they do not use 
backpropagation to update their weights like conventional neural networks.
</p>

<h2>CausalELM Features</h2>
<ul>
  <li>Simple interface enables estimating causal effects in only a few lines of code</li>
  <li>L2 penalty derived analytically to reduce training time</li>
  <li>Fast cross validation to calculate optimal number of neurons by using an ELM to 
  approximate the validation loss function
  </li>
  <li>Cross validation procedure is automatic and works with longitudinal, panel, and time 
  series data
  </li>
  <li>Use one of 13 activation functions or provide a custom activation function</li>
  <li>The same interface can be used for continous, binary, and categorical outcome 
  variables
  </li>
  <li>Does not require dependencies outside of the Julia standard library</li>
</ul>

<h2>Using CausalELM</h2>

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

<h2>Next Steps</h2>
<p>
The next release of CausalELM will incorporate p-value calculations, confidence intervals, 
and robustness tests for the models.
</p>

<h2>Suggestions</h2>
<p>
If you have any suggestions, feature requests, or find a bug feel free to report it on the 
<a href="https://github.com/dscolby/CausalELM.jl/issues">issue tracker</a>
</p>

<h2>Contributing</h2>
<p>
All contributions are welcome. Feel free to submit a pull request 
<a href="https://github.com/dscolby/CausalELM.jl/pulls">here.
</p>