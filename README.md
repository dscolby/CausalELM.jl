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
    <a href="https://dscolby.github.io/CausalELM.jl/stable">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="Documentation">
    </a>
    <a href="https://dscolby.github.io/CausalELM.jl/dev/">
        <img src="https://img.shields.io/badge/docs-dev-blue.svg"
             alt="Develmopmental Documentation">
    </a>
</p>
<h2>TL;DR</h2>
<p>
CausalELM enables estimation of causal effects in settings where a randomized control trial 
would be impossible or infeasible. Estimation of the average treatment effect (ATE), intent
to treat effect (ITE), and average treatment effect on the treated (ATT) can be estimated 
via G-computation or double machine learning (DML) while temporal changes can be estimated 
from an interrupted time series analysis. CausalELM also supports estimation of individual 
treatment effects or conditional average treatment effects (CATE) via S-learning, 
T-learning, and X-learning. The underlying machine learning model for all these estimators 
is an extreme learning machine or L2 regularized extreme learning machine.
</p>

<h2>Extreme Learning Machines and Causal Inference</h2>
<p>
In some cases we would like to know the causal effect of some intervention but we do not 
have the counterfactual, making conventional methods of statistical analysis infeasible. 
However, it may still be possible to get an unbiased estimate of the causal effect (ATE, 
ATE, or ITT) by predicting the counterfactual and comparing it to the observed outcomes. 
This is the approach CausalELM takes to conduct interrupted time series analysis, 
G-Computation, DML, and meatlearning via S-Learners, T-Learners, and X-Learners. In 
interrupted time series analysis, we want to estimate the effect of some intervention on 
the outcome of a single unit that we observe during multiple time periods. For example, we 
might want to know how the announcement of a merger affected the price of Stock A. To do 
this, we need to know what the price of stock A would have been if the merger had not been 
announced, which we can predict with machine learning methods. Then, we can compare this 
predicted counterfactual to the observed price data to estimate the effect of the merger 
announcement. In another case, we might want to know the effect of medicine X on disease Y 
but the administration of X was not random and it might have also been administered at 
mulitiple time periods, which would produce biased estimates. To overcome this, 
G-computation models the observed data, uses the model to predict the outcomes if all 
patients recieved the treatment, and compares it to the predictions of the outcomes if none 
of the patients recieved the treatment. Double machine learning (DML) takes a similar 
approach but also models the treatment mechanism and uses it to adjust the initial 
estimates. This approach has three advantages. First, it is more efficient with high 
dimensional data than conventional methods. Second, it allows one to model complex, 
nonlinear relationships between the treatment and the outcome. Finally, it is a doubly 
robust estimator, meaning that only the model of the outcome OR the model of the 
treatment mechanism has to be correctly specified to yield unbiased estimates. The DML 
implementation in CausalELM. also overcomes bias from regularization by employing cross 
fitting. Furthermore, we might be more interested in how much an individual can benefit from 
a treatment, as opposed to the average treatment effect. Depending on the characteristics of 
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
  <li>Analytically derived L2 penalty reduces cross validation time and multicollinearity</li>
  <li>Fast automatic cross validation works with longitudinal, panel, and time series data</li>
  <li>Includes 13 activation functions and allows user-defined activation functions</li>
  <li>Single interface for continous, binary, and categorical outcome variables</li>
  <li>Estimation of p-values and standard errors via asymptotic randomization inference</li>
  <li>No dependencies outside of the Julia standard library</li>
</ul>

<h2>What's New?</h2>
<ul>
  <li>All functions and methods converted to snake case</li>
  <li>Randomization inference for interrupted time series randomizes all indices</li>
  <li>Implemented validate method to probe assumptions for all estimators and metalearners</li>
  <li>Reimplemented cross validation for temporal data</li>
  <li>Fixed issue related to recoding variables to calculate validation metrics for cross validation</li>
</ul>

<h2>Next Steps</h2>
<p>
The next version/s will focus on optimizing the codebase for maintainability and modularity. 
More specifically, all functions will be generalized to accepts any real numbers, long 
functions and methods will be broken up into smaller functions or methods, and design 
patterns will be used where applicable.
</p>

<h2>Disclaimer</h2>
CausalELM is extensively tested and almost every function or method has multiple tests. That
being said, CausalELM is still in the early stages of development and may have some bugs. 
Also, expect breaking releases for now.

<h2>Contributing</h2>
<p>
All contributions are welcome. Before submitting a pull request please read the  
<a href="https://dscolby.github.io/CausalELM.jl/stable/contributing/">contribution guidlines.
</p>
