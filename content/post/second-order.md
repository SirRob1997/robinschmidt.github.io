---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Distributed Second-Order Optimization Using Kronecker-Factored Approximate Curvature"
subtitle: ""
summary: "A recent publication tried to parallelize K-FAC for multiple processes to speed up convergence. In this blog post I want to summarize the main contribution and give a little more insights."
authors: ["Robin M. Schmidt"]
tags: ["Optimization", "Second-Order", "K-FAC", "Parallelization"]
categories: ["Optimization", "Second-Order", "K-FAC", "Parallelization"]
date: 2019-11-03T00:04:59+01:00
lastmod: 2019-11-03T00:04:59+01:00
featured: true
draft: false
markup: mmark

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# Introduction

This blog post ommits some citations due to visualisation purposes, for a full list of references please refer to the {{% staticref "files/paper_large_scale.pdf" "newtab" %}}whitepaper version{{% /staticref %}} which was written for a seminar during my graduate studies.


# Notation
In our notation the training data $$\mathcal{T}=\left\{\left(\mathbf{x}_{1}, \mathbf{y}_{1}\right),\left(\mathbf{x}_{2}, \mathbf{y}_{2}\right), \cdots,\left(\mathbf{x}_{n}, \mathbf{y}_{n}\right)\right\}$$ consists of $$n$$ feature-label pairs. Here, each $$\mathbf{x}_{i} \in \mathbb{R}^{d_{x}}$$ is the feature vector and $$\mathbf{y}_{i} \in \mathbb{R}^{d_{y}}$$ is the label vector with their respective sizes $$d_{x}$$ and $$d_{y}$$. The deep learning model is described as a mapping $$F(\cdot ; \boldsymbol{\theta}): \mathcal{X} \rightarrow \mathcal{Y}$$ from the feature space $$\mathcal{X}$$ to the label space $$\mathcal{Y}$$ where $$\boldsymbol{\theta}$$ are the parameters of the model. This leaves us with a model notation which when presented with an input instance $$\mathbf{x}_{i} \in \mathcal{X}$$ yields a predicted output denoted as $$\hat{\mathbf{y}}_{i}=F\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right)$$. The difference of the true label $$\mathbf{y}_{i} \in \mathcal{Y}$$ to this predicted label $$\hat{\mathbf{y}}_{i}$$ is then described as the loss term $$\ell\left(\hat{\mathbf{y}}_{i}, \mathbf{y}_{i}\right)$$ which can have different definitions based on the specific problem (e.g. Mean Squared Error, Hinge Loss, Cross Entropy Loss, etc.). By summing up over all data points in the training data we get the total loss term defined as:

$$\mathcal{L}(\boldsymbol{\theta} ; \mathcal{T})=\sum_{\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right) \in \mathcal{T}} \ell\left(\hat{\mathbf{y}}_{i}, \mathbf{y}_{i}\right)$$

During training we try to optimize the model parameters $$\boldsymbol{\theta} \in \mathbb{R}^{d_{\theta}}$$, which are part of the variable domain $$\Theta$$, by minimizing the total loss on the training set. This can be described as:

$$\min _{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta} ; \mathcal{T})$$

For this process of finding a global or local minimum for convex and non-convex loss surfaces, a variety of different optimization algorithms are available. Most of these algorithms use the learning rate $$\eta$$ to determine the step sizes taken for the parameters $$\boldsymbol{\theta}$$ at each update step $$\tau$$ in the opposite direction of the gradient of the loss function $$\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(\tau-1)} ; \cdot)$$. Here, $$\boldsymbol{\theta}^{(\tau)}$$ are the parameters of the model at the update step $$\tau$$ with $$\tau \geq 1$$. The initialized hyperparameter values are denoted as $$\boldsymbol{\theta}^{(0)}$$.

# Related Work

Related Work in the realm of Deep Learning Optimizers can broadly be classified in First- and Second-Order Optimization Algorithms. Here we want to give a quick overview over those two areas.

## First-Order Optimization Algorithms

There are various First-Order Optimization Algorithms which are widely used in Deep Learning. One of the most popular choices due to its simplicity is still *Stochastic Gradient Descent* (SGD) with the update rule:

$$\boldsymbol{\theta}^{(\tau)}=\boldsymbol{\theta}^{(\tau-1)}-\eta \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}\left(\boldsymbol{\theta}^{(\tau-1)} ;\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)\right)$$

However, there have been recent advances yielding new and improved First-Order Optimizers such as *Adam*, *AdamW*, *AMSGrad*, *AdaBound*, *AMSBound*, *RAdam*, *LookAhead*, *Ranger* and many more which offer time-convergence improvements based on Adaptive Gradient methods and Momentum Terms.
## Second-Order Optimization Algorithms

The generalized *Gauss-Newton-Method*  and *Natural Gradient Descent* (NGD)  set the groundwork for improvements on Second-Order Optimization Algorithms. Such work yielded the *Kronecker-factored Approximate Curvature* (K-FAC) which effeciently approximates the empirical Fisher information matrix (FIM) $$\mathbf{F}_{\boldsymbol{\theta}}$$ given in the next Equation through block-diagonalization and Kronecker factorization of these blocks:

$$\mathbf{F}_{\boldsymbol{\theta}} = \mathop{\mathbb{E}}_{p(\mathbf{x},\mathbf{y})}\left[\nabla \log p(\mathbf{y}|\mathbf{x};\boldsymbol{\theta}) \nabla \log p(\mathbf{y}|\mathbf{x};\boldsymbol{\theta})^T\right]$$

For a neural network with $$L$$ Layers K-FAC approximates $$\mathbf{F}_{\boldsymbol{\theta}}$$ as displayed in the next Equation with $$\mathbf{F}_\ell$$ being the block matrix for the FIM of the $$\ell$$ th layer:

$$\mathbf{F}_{\boldsymbol{\theta}} \approx \operatorname{diag}\left(\mathbf{F}_1, \mathbf{F}_2, \dots, \mathbf{F}_\ell, \dots \mathbf{F}_L \right)$$

Each block is then approximated using the Kronecker-factorization:

$$\mathbf{F}_\ell \approx \mathbf{G}_\ell \otimes \mathbf{A}_{\ell-1}$$

With the properties of the Kronecker-factorization we can write the blocks as:

$$\mathcal{G}_\ell^{(\tau-1)} = \left({\mathbf{G}_\ell^{(\tau-1)}}^{-1} \otimes {\mathbf{A}_{\ell-1}^{(\tau-1)}}^{-1}\right)$$

Now using the NGD update rule we get the update rule for the parameters $$\boldsymbol{\theta}_\ell^{(\tau)}$$:

$$\boldsymbol{\theta}_\ell^{(\tau)} = \boldsymbol{\theta}_\ell^{(\tau-1)} - \eta \cdot \mathcal{G}_\ell^{(\tau-1)} \cdot \nabla \mathcal{L}_\ell\left(\boldsymbol{\theta}_\ell^{(\tau-1)};\cdot\right)$$

Besides the problem of inverting infeasible large matrices such as the FIM or the Hessian, which K-FAC tries to solve, a common drawback for Second-order optimizers is the complexity to optimize them for distributed computing. This is where [^1] tries to contribute a method which will improve the state-of-the-art.
# Parallelized K-FAC



# Results

# Conclusion & Outlook




[^1]: Kazuki  Osawa,  Yohei  Tsuji,  Yuichiro Ueno,  Akira Naruse,  Rio Yokota,  and Satoshi  Matsuoka. Large-scale  distributed second-order optimization using kronecker-factored approximate curvature for deep convolutional neural networks, 2018
