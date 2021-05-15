+++
title = "Explainability-aided Domain Generalization for Image Classification"
date = 2021-03-01T00:00:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Robin M. Schmidt"]

# Publication type.
# Legend:
# 0 = Uncategorized
# 1 = Conference paper
# 2 = Journal article
# 3 = Manuscript
# 4 = Report
# 5 = Book
# 6 = Book section
publication_types = ["0"]

# Publication name and optional abbreviated version.
#publication = "In *International Conference on Machine Learning, ICML 2021*."
#publication_short = "In *ICML*"

# Abstract and optional shortened version.
abstract = "Traditionally, for most machine learning settings, gaining some degree of explainability that tries to give users more insights into how and why the network arrives at its predictions, restricts the underlying model and hinders performance to a certain degree. For example, decision trees are thought of as being more explainable than deep neural networks but they lack performance on visual tasks. In this work, we empirically demonstrate that applying methods and architectures from the explainability literature can, in fact, achieve state-of-the-art performance for the challenging task of domain generalization while offering a framework for more insights into the prediction and training process. For that, we develop a set of novel algorithms including DivCAM, an approach where the network receives guidance during training via gradient based class activation maps to focus on a diverse set of discriminative features, as well as ProDrop and D-Transformers which apply prototypical networks to the domain generalization task, either with self-challenging or attention alignment. Since these methods offer competitive performance on top of explainability, we argue that the proposed methods can be used as a tool to improve the robustness of deep neural network architectures."
abstract_short = "We present multiple methods for Domain Generalization based on some recent trends in the explainability literature."

# Is this a featured publication? (true/false)
featured = true

# Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Tags (optional).
#   Set `tags = []` for no tags, or use the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []

# Links (optional).
url_pdf = "https://arxiv.org/pdf/2104.01742.pdf"
url_preprint = "https://arxiv.org/abs/2104.01742"
#url_code = ""
#url_dataset = "#"
#url_project = "#"
#url_slides = "#"
#url_video = "#"
#url_poster = "#"
#url_source = "#"

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# url_custom = [{name = "OpenReview", url = "https://openreview.net/forum?id=rJg6ssC5Y7"}, {name = #"Documentation", url = "https://deepobs.readthedocs.io/"}]

# Digital Object Identifier (DOI)
doi = ""

# Does this page contain LaTeX math? (true/false)
math = true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
[image]
  # Caption (optional)
  # caption = "Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)"

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = "TopLeft"
+++