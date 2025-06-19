# Predictive Coding for MNIST Classification

## Overview

Predictive coding is a theoretical framework used in neuroscience and computational modeling to describe how the brain processes sensory information and learns from it. It proposes that the brain continuously generates predictions about incoming sensory data based on an internal model of the world and updates this model by comparing predictions to actual sensory inputs. The difference between predictions and reality, known as the prediction error, drives perception and learning to make future predictions more accurate.

This project implements a predictive coding neural network with Hebbian learning to classify MNIST digits (0-9). Predictive coding is a biologically inspired framework where the network generates top-down predictions, computes prediction errors, and updates activities and weights to minimize errors. The model uses a hierarchical structure with iterative inference and batch processing for efficiency.

# Features
---
**Architecture** Four layers (input: 784, hidden1: 128, hidden2: 64, output: 10).
**Learning** Hebbian updates for weights (W1, W2: unsupervised; W3: supervised) based on activity correlations and output errors.
**Inference** Iterative activity updates (up to 20 steps) to minimize layer-wise prediction errors (e1, e2, e3, e4).
**Layer-by-layer error computation.**
**State updates with ReLU.**
**Normalized Hebbian weight updates.**
**Lateral connections**
**MNIST data processing, training, and visualization.**

---

##  Core Concepts

- **Top-Down Prediction**: Higher layers generate expectations for lower layers.
- **Bottom-Up Error**: Mismatch between prediction and reality is computed as an error signal.
- **State Update**: Internal beliefs are refined iteratively.
- **Weight Learning**: Weights are updated after convergence to reduce future errors.
- **Biological Plausibility**: Mimics cortical circuits and avoids backpropagation.


