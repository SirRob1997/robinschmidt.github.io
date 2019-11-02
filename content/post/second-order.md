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
draft: true
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

I have more [^1] to say.


# Notation
In our notation the training data $$\mathcal{T}=\left\{\left(\mathbf{x}_{1}, \mathbf{y}_{1}\right),\left(\mathbf{x}_{2}, \mathbf{y}_{2}\right), \cdots,\left(\mathbf{x}_{n}, \mathbf{y}_{n}\right)\right\}$$ consists of $$n$$ feature-label pairs. Here, each $$\mathbf{x}_{i} \in \mathbb{R}^{d_{x}}$$ is the feature vector and $$\mathbf{y}_{i} \in \mathbb{R}^{d_{y}}$$ is the label vector with their respective sizes $$d_{x}$$ and $$d_{y}$$. The deep learning model is described as a mapping $$F(\cdot ; \boldsymbol{\theta}): \mathcal{X} \rightarrow \mathcal{Y}$$ from the feature space $$\mathcal{X}$$ to the label space $$\mathcal{Y}$$ where $$\boldsymbol{\theta}$$ are the parameters of the model. This leaves us with a model notation which when presented with an input instance $$\mathbf{x}_{i} \in \mathcal{X}$$ yields a predicted output denoted as $$\hat{\mathbf{y}}_{i}=F\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right)$$. The difference of the true label $$\mathbf{y}_{i} \in \mathcal{Y}$$ to this predicted label $$\hat{\mathbf{y}}_{i}$$ is then described as the loss term $$\ell\left(\hat{\mathbf{y}}_{i}, \mathbf{y}_{i}\right)$$ which can have different definitions based on the specific problem (e.g. Mean Squared Error, Hinge Loss, Cross Entropy Loss, etc.). By summing up over all data points in the training data we get the total loss term defined as:

$$\mathcal{L}(\boldsymbol{\theta} ; \mathcal{T})=\sum_{\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right) \in \mathcal{T}} \ell\left(\hat{\mathbf{y}}_{i}, \mathbf{y}_{i}\right)$$

During training we try to optimize the model parameters $$\boldsymbol{\theta} \in \mathbb{R}^{d_{\theta}}$$, which are part of the variable domain $$\Theta$$, by minimizing the total loss on the training set. This can be described as:

$$\min _{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta} ; \mathcal{T})$$

For this process of finding a global or local minimum for convex and non-convex loss surfaces, a variety of different optimization algorithms are available. Most of these algorithms use the learning rate $$\eta$$ to determine the step sizes taken for the parameters $$\boldsymbol{\theta}$$ at each update step $$\tau$$ in the opposite direction of the gradient of the loss function $$\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(\tau-1)} ; \cdot)$$. Here, $$\boldsymbol{\theta}^{(\tau)}$$ are the parameters of the model at the update step $$\tau$$ with $$\tau \geq 1$$. The initialized hyperparameter values are denoted as $$\boldsymbol{\theta}^{(0)}$$.

# Related Work

## First-Order Optimization Algorithms

## Second-Order Optimization Algorithms

# Parallelized K-FAC

# Results

# Conclusion & Outlook




[^1]: Footnote example.
