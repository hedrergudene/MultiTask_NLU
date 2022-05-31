# MultiTask NLU

---
## Table of contents
- [1. Introduction](#introduction)
- [2. Model architecture](#model-architecture)
- [3. Monitoring integration](#monitoring-integration)
- [4. Quickstart code](#quickstart-code)
- [5. License](#license)
---

## Introduction
Text and token classification are two of the most popular downstream tasks in Natural Language Processing (NLP), enabling semantical and lexical analysis of utterances, respectively. Both problems are intrinsecally linked, even though there has always been some disparity between them, therefore it makes sense to ask ourselves if there is any procedure to combine them in a network to help one task solve the other, and *vice versa*. That work was carried out in [this paper](https://www.researchgate.net/publication/355862206_Unified_Transformer_Multi-Task_Learning_for_Intent_Classification_With_Entity_Recognition), and in that model we will inspire our work.

We will make use of the recently released [MASSIVE dataset](https://github.com/alexa/massive):

> MASSIVE is a parallel dataset of > 1M utterances across 51 languages with annotations for the Natural Language Understanding tasks of intent prediction and slot annotation. Utterances span 60 intents and include 55 slot types. MASSIVE was created by localizing the SLURP dataset, composed of general Intelligent Voice Assistant single-shot interactions.

Information about intent and entities within utterances is contained in the dataset.

## Model architecture
![MTImage](input/MultiTask_image.PNG)



## Monitoring integration
This experiment has been integrated with Weights and Biases to track all metrics, hyperparameters, callbacks and GPU performance. You only need to adapt the parameters in the `wandb_config.json` configuration file to keep track of the model training and evaluation. An example is shown [here](https://wandb.ai/azm630/MultiTask_NLU).


## Quickstart code
WIP

## License
Released under [MIT](/LICENSE) by [@hedrergudene](https://github.com/hedrergudene).
