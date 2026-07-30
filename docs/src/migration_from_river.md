# Migration from River (Python)

This guide helps you translate your River (Python) code to OnlineML.jl.

## Core Concepts

| River (Python) | OnlineML.jl | Notes |
|----------------|-------------|-------|
| `model.learn_one(x, y)` | `fit!(model, (x, y))` | Tuple syntax for supervised |
| `model.learn_one(x)` | `fit!(model, x)` | Direct for unsupervised |
| `model.predict_one(x)` | `predict(model, x)` | Same concept |
| `model.predict_proba_one(x)` | `predict_proba(model, x)` | Returns Dict |

## Linear Models

### River
```python
from river import linear_model

# Logistic Regression
model = linear_model.LogisticRegression(optimizer=optim.SGD(0.01))
model.learn_one(x, y)
y_pred = model.predict_one(x)

# Perceptron
model = linear_model.Perceptron()

# Passive-Aggressive
model = linear_model.PAClassifier()
```

### OnlineML.jl
```julia
using OnlineML.Linear
using Optimisers

# Logistic Regression
model = LogisticRegression(optimizer=Descent(0.01))
fit!(model, (x, y))
y_pred = predict(model, x)

# Perceptron
model = Perceptron()

# Passive-Aggressive
model = PassiveAggressiveClassifier()
```

## Decision Trees

### River
```python
from river import tree

model = tree.HoeffdingTreeClassifier(
    grace_period=200,
    split_criterion='gini',
    delta=1e-7
)
model.learn_one(x, y)
```

### OnlineML.jl
```julia
using OnlineML.Trees

model = HoeffdingTree(
    grace_period=200,
    split_criterion=:gini,
    delta=1e-7
)
fit!(model, (x, y))
```

## Naive Bayes

### River
```python
from river import naive_bayes

model = naive_bayes.GaussianNB()
model.learn_one(x, y)
probs = model.predict_proba_one(x)
```

### OnlineML.jl
```julia
using OnlineML.Bayes

model = GaussianNB()
fit!(model, (x, y))
probs = predict_proba(model, x)  # Returns Dict{class => prob}
```

## Ensemble Methods

### River
```python
from river import ensemble

# Bagging
model = ensemble.BaggingClassifier(
    model=linear_model.LogisticRegression(),
    n_models=10
)

# Adaptive Random Forest
model = ensemble.AdaptiveRandomForestClassifier(n_models=10)
```

### OnlineML.jl
```julia
using OnlineML.Ensemble
using OnlineML.Linear: LogisticRegression

# Bagging
model = Bagging(() -> LogisticRegression(), n_estimators=10)

# Adaptive Random Forest
model = AdaptiveRandomForest(n_estimators=10)
```

## Drift Detection

### River
```python
from river import drift

# ADWIN
detector = drift.ADWIN()
detector.update(error_value)
if detector.drift_detected:
    print("Drift!")

# DDM
detector = drift.DDM()
```

### OnlineML.jl
```julia
using OnlineML.Drift

# ADWIN
detector = ADWIN()
status = update!(detector, error_value)
if status == DriftDetected
    println("Drift!")
end

# DDM
detector = DDM()
```

## Transformers

### River
```python
from river import preprocessing

# Standard Scaler
scaler = preprocessing.StandardScaler()
x_scaled = scaler.learn_one(x).transform_one(x)

# MinMax Scaler
scaler = preprocessing.MinMaxScaler()

# One-Hot Encoder
encoder = preprocessing.OneHotEncoder()
```

### OnlineML.jl
```julia
using OnlineML.Transform

# Standard Scaler
scaler = StandardScaler()
x_scaled = fit_transform!(scaler, x)

# MinMax Scaler
scaler = MinMaxScaler()

# One-Hot Encoder
encoder = OneHotEncoder()
```

## Pipelines

### River
```python
from river import compose

pipeline = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression()
)

pipeline.learn_one(x, y)
y_pred = pipeline.predict_one(x)
```

### OnlineML.jl
```julia
using OnlineML.Pipeline
using OnlineML.Transform: StandardScaler
using OnlineML.Linear: LogisticRegression

pipeline = StandardScaler() |> LogisticRegression()

fit!(pipeline, (x, y))
y_pred = predict(pipeline, x)
```

## Metrics

### River
```python
from river import metrics

metric = metrics.Accuracy()
metric.update(y_true, y_pred)
score = metric.get()
```

### OnlineML.jl
```julia
using OnlineML.Metrics

metric = Accuracy()
fit!(metric, (y_pred, y_true))
score = value(metric)
```

## Evaluation

### River
```python
from river import evaluate

# Progressive validation
evaluate.progressive_val_score(
    dataset,
    model,
    metric
)
```

### OnlineML.jl
```julia
using OnlineML.Metrics: progressive_val_score

score, trajectory = progressive_val_score(
    model,
    generator,
    metric,
    n_samples=1000
)
```

## Synthetic Data

### River
```python
from river import synth

# SEA
stream = synth.SEA(seed=42, variant=0)

# Agrawal
stream = synth.Agrawal(seed=42, classification_function=0)

for x, y in stream.take(1000):
    model.learn_one(x, y)
```

### OnlineML.jl
```julia
using OnlineML.Streams
using Random

# SEA
gen = SEAGenerator(function_id=1, rng=MersenneTwister(42))

# Agrawal
gen = AgrawalGenerator(function_id=1, rng=MersenneTwister(42))

for _ in 1:1000
    x, y = generate(gen)
    fit!(model, (x, y))
end
```

## Anomaly Detection

### River
```python
from river import anomaly

model = anomaly.HalfSpaceTrees(
    n_trees=10,
    height=8,
    window_size=250
)
score = model.score_one(x)
model.learn_one(x)
```

### OnlineML.jl
```julia
using OnlineML.Anomaly

model = HalfSpaceTrees(
    n_trees=10,
    height=8,
    window_size=250
)
score = score_one(model, x)
fit!(model, x)
```

## Key Differences

1. **Tuple Syntax**: OnlineML uses `fit!(model, (x, y))` tuple syntax for supervised learning to match OnlineStatsBase conventions.

2. **Immutable Types**: Julia prefers immutable types where possible. OnlineML models are mutable structs.

3. **Type Parameters**: Julia's parametric types provide compile-time optimization. OnlineML leverages this for performance.

4. **Generator Functions**: Instead of iterators, OnlineML uses `generate(gen)` functions for synthetic data.

5. **Optimizer Integration**: OnlineML uses `Optimisers.jl` for gradient-based optimizers, providing access to Adam, Momentum, etc.

6. **Drift Status Enum**: OnlineML uses `DriftStatus` enum (`NoDrift`, `Warning`, `DriftDetected`) instead of boolean flags.

## Performance Notes

OnlineML.jl is designed for high performance:
- Linear models: >1M observations/second
- Tree models: >100K observations/second
- Typically 2-10x faster than River for equivalent algorithms

This is achieved through:
- Julia's JIT compilation
- Type stability
- Minimal allocations in hot paths
- Wrapping optimized Julia packages (OnlineStats, NearestNeighbors)
