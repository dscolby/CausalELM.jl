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
CausalELM provides easy-to-use implementations of modern causal inference methods. While
CausalELM implements a variety of estimators, they all have one thing in common—the use of 
machine learning models to flexibly estimate causal effects. This is where the ELM in 
CausalELM comes from—the machine learning model underlying all the estimators is an extreme 
learning machine (ELM). ELMs are a simple neural network that use randomized weights and 
offer a good tradeoff between learning non-linear dependencies and simplicity. Furthermore, 
CausalELM implements bagged ensembles of ELMs to reduce the variance resulting from 
randomized weights.
</p>

<h2>Estimators</h2>
<p>
CausalELM implements estimators for aggreate e.g. average treatment effect (ATE) and 
individualized e.g. conditional average treatment effect (CATE) quantities of interest.
</p>

<h3>Estimators for Aggregate Effects</h3>
<ul>
    <li>Interrupted Time Series Estimator</li>
    <li>G-computation</li>
    <li>Double machine Learning</li>
</ul>

<h3>Individualized Treatment Effect (CATE) Estimators</h3>
<ul>
    <li>S-learner</li>
    <li>T-learner</li>
    <li>X-learner</li>
    <li>R-learner</li>
    <li>Doubly Robust Estimator</li>
</ul>

<h2>Features</h2>
<ul>
  <li>Estimate a causal effect, get a summary, and validate assumptions in just four lines of code</li>
  <li>Bagging improves performance and reduces variance without the need to tune a regularization parameter</li>
  <li>Enables using the same structs for regression and classification</li>
  <li>Includes 13 activation functions and allows user-defined activation functions</li>
  <li>Most inference and validation tests do not assume functional or distributional forms</li>
  <li>Implements the latest techniques form statistics, econometrics, and biostatistics</li>
  <li>Works out of the box with arrays or any data structure that implements the Tables.jl interface</li>
  <li>Works out of the box with AbstractArrays or any data structure that implements the Tables.jl interface</li>
  <li>Works with CuArrays, ROCArrays, and any other GPU-specific arrays that are AbstractArrays</li>
  <li>CausalELM is lightweight—its only dependency is Tables.jl</li>
  <li>Codebase is high-quality, well tested, and regularly updated</li>
</ul>

<h2>What's New?</h2>
<ul>
  <li>See the JuliaCon 2024 CausalELM demonstration <a href="https://www.youtube.com/watch?v=hh_cyj8feu8&t=26s">here.
  <li>Includes support for GPU-specific arrays and data structures that implement the Tables.jl API<li>
  <li>Only performs randomization inference when the inference argument is set to true in summarize methods</li>
  <li>Summaries support calculating marginal effects and confidence intervals</li>
  <li>Randomization inference now uses multithreading</li>
  <li>Refactored code to be easier to extend and understand</li>
  <li>Uses a simple heuristic to choose the number of neurons, which reduces training time and still works well in practice</li>
  <li>Probability clipping for classifier predictions and residuals is no longer necessary due to the bagging procedure</li>
</ul>

<h2>What's Next?</h2>
<p>
Efforts for the next version of CausalELM will focus on providing interpreteations for the results of callin validate as well
as fixing any bugs and eliciting feedback.
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
