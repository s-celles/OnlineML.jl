# OnlineML.jl

Online/incremental machine learning for Julia.

## Overview

OnlineML.jl provides streaming machine learning algorithms that update incrementally,
one observation at a time. This enables:

- **Memory efficiency**: No need to store all training data
- **Real-time learning**: Models update as new data arrives
- **Concept drift handling**: Detect and adapt to changing data distributions

## Quick Start

```julia
using OnlineML
using OnlineML.Linear: LogisticRegression
using OnlineML.Metrics: Accuracy

# Create a classifier
model = LogisticRegression()

# Train incrementally
for (x, y) in data_stream
    fit!(model, x, y)
end

# Make predictions
y_pred = predict(model, x_new)
```

## Features

- **Linear Models**: Regression, LogisticRegression, Perceptron, PassiveAggressive
- **Trees**: HoeffdingTree (VFDT algorithm)
- **Ensembles**: Bagging, AdaptiveRandomForest, LeveragingBagging
- **Instance-Based**: KNN with sliding window
- **Naive Bayes**: GaussianNB, MultinomialNB, BernoulliNB
- **Clustering**: KMeans, DenStream
- **Anomaly Detection**: HalfSpaceTrees, LODA
- **Transformers**: StandardScaler, MinMaxScaler, OneHotEncoder, TargetEncoder
- **Drift Detection**: ADWIN, DDM, EDDM, PageHinkley
- **Evaluation**: progressive_val_score, holdout_score
- **Pipelines**: Composable with `|>` operator

## Installation

```julia
using Pkg
Pkg.add("OnlineML")
```

## Index

```@index
```
