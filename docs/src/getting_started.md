# Getting Started with OnlineML.jl

This guide will help you get started with online machine learning in Julia.

## Installation

```julia
using Pkg
Pkg.add("OnlineML")
```

## Basic Concepts

Online learning algorithms update incrementally, one observation at a time. The core interface follows the OnlineStatsBase.jl conventions:

- `fit!(model, observation)` - Update model with one observation
- `predict(model, x)` - Generate a prediction
- `value(model)` - Get current model state
- `nobs(model)` - Get observation count
- `reset!(model)` - Reinitialize model

## Your First Online Model

### Classification Example

```julia
using OnlineML
using OnlineML.Linear: LogisticRegression
using OnlineML.Metrics: Accuracy
using OnlineML.Streams: SEAGenerator, generate

# Create a classifier
model = LogisticRegression()
metric = Accuracy()

# Create a synthetic data stream
gen = SEAGenerator(function_id=1, noise=0.1)

# Train incrementally with test-then-train evaluation
for i in 1:1000
    x, y = generate(gen)

    # Predict before training (prequential evaluation)
    if nobs(model) > 0
        y_pred = predict(model, x)
        fit!(metric, (y_pred, y))
    end

    # Update model
    fit!(model, (x, y))
end

println("Accuracy: ", value(metric))
```

### Regression Example

```julia
using OnlineML
using OnlineML.Linear: Regression
using OnlineML.Metrics: MAE

# Create regressor
model = Regression()
metric = MAE()

# Simulate streaming data
for i in 1:500
    x = randn(5)
    y = sum(x) + 0.1 * randn()  # Linear target with noise

    if nobs(model) > 0
        y_pred = predict(model, x)
        fit!(metric, (y_pred, y))
    end

    fit!(model, (x, y))
end

println("MAE: ", value(metric))
```

## Using Transformers

Transform your features before modeling with online scalers:

```julia
using OnlineML
using OnlineML.Transform: StandardScaler
using OnlineML.Linear: LogisticRegression

# Create transformer and model
scaler = StandardScaler()
model = LogisticRegression()

# Process data
for (x, y) in data_stream
    # Fit and transform features
    x_scaled = fit_transform!(scaler, x)

    # Train model on scaled features
    fit!(model, (x_scaled, y))
end

# For prediction, use transform (not fit_transform!)
x_new_scaled = transform(scaler, x_new)
y_pred = predict(model, x_new_scaled)
```

## Building Pipelines

Chain transformers and models with the `|>` operator:

```julia
using OnlineML
using OnlineML.Transform: StandardScaler, MinMaxScaler
using OnlineML.Linear: LogisticRegression
using OnlineML.Pipeline: OnlinePipeline

# Create a pipeline
pipeline = StandardScaler() |> LogisticRegression()

# Train the entire pipeline
for (x, y) in data_stream
    fit!(pipeline, (x, y))
end

# Predict through the pipeline
y_pred = predict(pipeline, x_new)
```

## Detecting Concept Drift

Online models can become stale when data distributions change. Use drift detectors:

```julia
using OnlineML
using OnlineML.Drift: ADWIN
using OnlineML.Linear: LogisticRegression

model = LogisticRegression()
detector = ADWIN()

for (x, y) in data_stream
    # Make prediction
    if nobs(model) > 0
        y_pred = predict(model, x)
        error = (y_pred != y) ? 1.0 : 0.0

        # Update drift detector with error rate
        status = update!(detector, error)

        if status == DriftDetected
            println("Drift detected! Resetting model...")
            reset!(model)
        end
    end

    fit!(model, (x, y))
end
```

Or use the automatic wrapper:

```julia
using OnlineML.Drift: DriftRetrainingWrapper, DDM

# Wrap model with automatic drift handling
wrapped = DriftRetrainingWrapper(LogisticRegression(), DDM())

for (x, y) in data_stream
    fit!(wrapped, (x, y))  # Automatically resets on drift
end

println("Number of drifts detected: ", wrapped.n_drifts)
```

## Evaluation Methodology

Use prequential (test-then-train) evaluation for honest performance estimates:

```julia
using OnlineML
using OnlineML.Linear: LogisticRegression
using OnlineML.Metrics: Accuracy, progressive_val_score
using OnlineML.Streams: SEAGenerator

model = LogisticRegression()
gen = SEAGenerator()
metric = Accuracy()

# Run prequential evaluation
final_score, trajectory = progressive_val_score(model, gen, metric, n_samples=1000)

println("Final accuracy: ", final_score)
println("Learning curve length: ", length(trajectory))
```

## Working with Tables

OnlineML integrates with Tables.jl for easy data loading:

```julia
using OnlineML
using OnlineML.Linear: LogisticRegression
using OnlineML.Streams: DataStream
using DataFrames

# Create a DataFrame
df = DataFrame(
    feature1 = randn(100),
    feature2 = randn(100),
    target = rand([0, 1], 100)
)

# Create a stream from the table
stream = DataStream(df, target=:target)

# Train model
model = LogisticRegression()
for (x, y) in stream
    fit!(model, (x, y))
end
```

## Using NamedTuples

You can also pass NamedTuple rows directly to `fit!`, which is convenient when working with row-oriented data:

### Supervised Learning with NamedTuples

```julia
using OnlineML
using OnlineML.Linear: LogisticRegression

model = LogisticRegression()

# Fit with NamedTuple input (specify target column)
fit!(model, (; x1=1.0, x2=2.0, y=1), :y)
fit!(model, (; x1=0.5, x2=1.5, y=0), :y)

println("Observations: ", nobs(model))  # 2
```

### Streaming NamedTuples

```julia
using OnlineML
using OnlineML.Linear: Perceptron

model = Perceptron()

# Process a stream of NamedTuple observations
stream = [
    (; feature1=1.0, feature2=0.5, label=1),
    (; feature1=2.0, feature2=1.5, label=0),
    (; feature1=3.0, feature2=2.5, label=1),
]

for row in stream
    fit!(model, row, :label)
end

println("Trained on: ", nobs(model), " samples")  # 3
```

### Unsupervised Learning with NamedTuples

For unsupervised learners, all fields become features by default:

```julia
using OnlineML
using OnlineML.Cluster: StreamingKMeans

model = StreamingKMeans(k=3)

# All fields are used as features
fit!(model, (; x1=1.0, x2=2.0))
fit!(model, (; x1=0.5, x2=1.5))

# Exclude non-feature fields with the exclude parameter
fit!(model, (; x1=1.0, x2=2.0, id=1); exclude=[:id])

println("Centroids: ", centroids(model))
```

### Integration with DataFrames

NamedTuples are the row type returned by `Tables.rows()`, making integration seamless:

```julia
using OnlineML
using OnlineML.Linear: LogisticRegression
using DataFrames

df = DataFrame(
    feature1 = [1.0, 2.0, 3.0],
    feature2 = [4.0, 5.0, 6.0],
    label = [0, 1, 0]
)

model = LogisticRegression()

# Convert each row to a NamedTuple and fit
for row in Tables.rows(df)
    nt = (; (k => getproperty(row, k) for k in propertynames(row))...)
    fit!(model, nt, :label)
end

# Or use the table-based fit! directly
model2 = LogisticRegression()
fit!(model2, df, :label)

# Both approaches produce the same result
println("nobs: ", nobs(model), " == ", nobs(model2))
```

## Synthetic Data Generators

Test your models with controlled synthetic streams:

```julia
using OnlineML.Streams

# SEA Generator - classic benchmark with concept drift
sea = SEAGenerator(function_id=1, noise=0.1)

# Agrawal Generator - 10 classification functions
agrawal = AgrawalGenerator(function_id=1)

# Hyperplane Generator - rotating decision boundary
hyperplane = HyperplaneGenerator(n_features=10, n_drift_features=2)

# RandomRBF Generator - Gaussian centroids
rbf = RandomRBFGenerator(n_classes=3, n_features=5, n_centroids=10)

# LED Generator - 7-segment display recognition
led = LEDGenerator(noise=0.1)

# Sine Generator - for regression tasks
sine = SineGenerator(amplitude=1.0, noise=0.05)

# Generate samples
for _ in 1:100
    x, y = generate(sea)
    # process sample...
end
```

## Next Steps

- See the [API Reference](api/index.md) for detailed documentation
- Check the [Migration from River](migration_from_river.md) guide if coming from Python
- Explore the benchmark suite for performance characteristics
