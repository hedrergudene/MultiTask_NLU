# MultiTask NLU

---
## Table of contents
- [1. Introduction](#introduction)
- [2. Implementation features](#implementation-features)
- [3. Monitoring integration](#monitoring-integration)
- [4. Quickstart code](#quickstart-code)
- [5. License](#license)
---

## Introduction
Text and token classification are two of the most popular downstream tasks in Natural Language Processing (NLP), enabling semantical and lexical analysis of utterances, respectively. Both problems are intrinsecally linked, even though there has always been some disparity between them, therefore it makes sense to ask ourselves if there is any procedure to combine them in a network to help one task solve the other, and *vice versa*. That work was carried out in [this paper](https://www.researchgate.net/publication/355862206_Unified_Transformer_Multi-Task_Learning_for_Intent_Classification_With_Entity_Recognition), and in that model we will inspire our work.

We will make use of the recently released [MASSIVE dataset](https://github.com/alexa/massive):

> MASSIVE is a parallel dataset of > 1M utterances across 51 languages with annotations for the Natural Language Understanding tasks of intent prediction and slot annotation. Utterances span 60 intents and include 55 slot types. MASSIVE was created by localizing the SLURP dataset, composed of general Intelligent Voice Assistant single-shot interactions.

Information about intent and entities within utterances is contained in the dataset.

## Implementation features

As it has already been mentioned, the architecture we use combines both text and token classification from a single feature extractor. To that end, we have to provide utterance intent and entity labelling. Some of the most remarkable components are:

* Text tokenisation and entity labels are included in dataset collator for memory efficiency purposes.
* Initial unification of hidden size to enable individualised processing of each problem.
* Relational modules to combine IC and NER information in both directions.
* Use of categorical crossentropy loss function for the IC branch, and focal loss function with $\gamma=2$ for the NER one. Final loss function is the average of both losses. Label smoothing for the first loss component, and gamma  parameter for the second, can be customised in the `training_config.json` file.
* Simple linear decay learning rate scheduler with warm start.

A visual description of the implementation is shown now:

![MTImage](input/MultiTask_image.PNG)



## Monitoring integration
This experiment has been integrated with Weights and Biases to track all metrics, hyperparameters, callbacks and GPU performance. You only need to adapt the parameters in the `wandb_config.json` configuration file to keep track of the model training and evaluation. An example is shown [here](https://wandb.ai/azm630/MultiTask_NLU).


## Quickstart code
You can start by using this notebook [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](/Quickstart.ipynb) in which you can easily get up-to-speed with your own data and customise parameters.


## License
Released under [MIT](/LICENSE) by [@hedrergudene](https://github.com/hedrergudene).
