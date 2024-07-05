<div align="center">
    <img src="https://github.com/dscolby/dscolby.github.io/blob/main/github_logo.jpg">
</div>

<p align="center">
    <a href="https://github.com/dscolby/CausalELM.jl/actions">
        <img src="https://github.com/dscolby/CausalELM.jl/actions/workflows/CI.yml/badge.svg?branch=main"
            alt="Build Status">
    </a>
    <a href="https://app.codecov.io/gh/dscolby/CausalELM.jl/tree/main/src">
        <img src="https://codecov.io/gh/dscolby/CausalELM.jl/graph/badge.svg"
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
    <a href="https://github.com/JuliaTesting/Aqua.jl">
        <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg"
             alt="Aqua QA">
    </a>
    <a href="https://github.com/JuliaDiff/BlueStyle">
        <img src="https://img.shields.io/badge/code%20style-blue-4495d1.svg"
             alt="Code Style: Blue">
    </a>
</p>

<p>
CausalELM enables estimation of causal effects in settings where a randomized control trial 
or traditional statistical models would be infeasible or unacceptable. It enables estimation 
of the average treatment effect (ATE)/intent to treat effect (ITE) with interrupted time 
series analysis, G-computation, and double machine learning; average treatment effect on the 
treated (ATT) with G-computation; cumulative treatment effect with interrupted time series 
analysis; and the conditional average treatment effect (CATE) via S-learning, T-learning, 
X-learning, R-learning, and doubly robust estimation. Underlying all of these estimators are 
ensembles of extreme learning machines, a simple neural network that uses randomized weights 
and least squares optimization instead of gradient descent. Once a model has been estimated, 
CausalELM can summarize the model and conduct sensitivity analysis to validate the 
plausibility of modeling assumptions. Furthermore, all of this can be done in four lines of 
code.
</p>

<h2>Extreme Learning Machines and Causal Inference</h2>
<p>
In some cases we would like to know the causal effect of some intervention but we do not 
have the counterfactual, making conventional methods of statistical analysis infeasible. 
However, it may still be possible to get an unbiased estimate of the causal effect (ATE, 
ATE, or ITT) by predicting the counterfactual and comparing it to the observed outcomes. 
This is the approach CausalELM takes to conduct interrupted time series analysis, 
G-Computation, double machine learning, and metalearning via S-Learners, T-Learners, 
X-Learners, R-learners, and doubly robust estimation. In interrupted time series analysis, 
we want to estimate the effect of some intervention on the outcome of a single unit that we 
observe during multiple time periods. For example, we might want to know how the 
announcement of a merger affected the price of Stock A. To do this, we need to know what the 
price of stock A would have been if the merger had not been announced, which we can predict 
with machine learning methods. Then, we can compare this predicted counterfactual to the 
observed price data to estimate the effect of the merger announcement. In another case, we 
might want to know the effect of medicine X on disease Y but the administration of X was not 
random and it might have also been administered at mulitiple time periods, which would 
produce biased estimates. To overcome this, G-computation models the observed data, uses the 
model to predict the outcomes if all patients recieved the treatment, and compares it to the 
predictions of the outcomes if none of the patients recieved the treatment. Double machine 
learning (DML) takes a similar approach but also models the treatment mechanism and uses it 
to adjust the initial estimates. This approach has three advantages. First, it is more 
efficient with high dimensional data than conventional methods. Metalearners take a similar 
approach to estimate the CATE. While all of these models are different, they have one thing 
in common: how well they perform depends on the underlying model they fit to the data. To 
that end, CausalELMs use bagged ensembles of extreme learning machines because they are 
simple yet flexible enough to be universal function approximators with lower varaince than 
single extreme learning machines.
</p>

<h2>CausalELM Features</h2>
<ul>
  <li>Estimate a causal effect, get a summary, and validate assumptions in just four lines of code</li>
  <li>Bagging improves performance and reduces variance without the need to tune a regularization parameter</li>
  <li>Enables using the same structs for regression and classification</li>
  <li>Includes 13 activation functions and allows user-defined activation functions</li>
  <li>Most inference and validation tests do not assume functional or distributional forms</li>
  <li>Implements the latest techniques form statistics, econometrics, and biostatistics</li>
  <li>Works out of the box with arrays or any data structure that implements the Tables.jl interface</li>
  <li>Codebase is high-quality, well tested, and regularly updated</li>
</ul>

<h2>What's New?</h2>
<ul>
  <li>Now includes doubly robust estimator for CATE estimation</li>
  <li>All estimators now implement bagging to reduce predictive performance and reduce variance</li>
  <li>Counterfactual consistency validation simulates more realistic violations of the counterfactual consistency assumption</li>
  <li>Uses a simple heuristic to choose the number of neurons, which reduces training time and still works well in practice</li>
  <li>Probability clipping for classifier predictions and residuals is no longer necessary due to the bagging procedure</li>
  <li>CausalELM talk has been accepted to JuliaCon 2024!</li> 
</ul>

<h2>What's Next?</h2>
<p>
Newer versions of CausalELM will hopefully support using GPUs and provide interpretations of 
the results of calling validate on a model that has been estimated. In addition, some 
estimators will also support using instrumental variables. However, these priorities could 
also change depending on feedback recieved at JuliaCon.
</p>

<h2>Disclaimer</h2>
CausalELM is extensively tested and almost every function or method has multiple tests. That
being said, CausalELM is still in the early/ish stages of development and may have some 
bugs. Also, expect breaking releases for now.

<h2>Contributing</h2>
<p>
All contributions are welcome. Before submitting a pull request please read the  
<a href="https://dscolby.github.io/CausalELM.jl/stable/contributing/">contribution guidlines.
</p>
