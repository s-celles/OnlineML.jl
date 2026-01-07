# Streams and Generators

Data stream utilities and synthetic data generators.

## DataStream

```@docs
OnlineML.Streams.DataStream
OnlineML.Streams.batch_stream
OnlineML.Streams.take_stream
OnlineML.Streams.skip_stream
```

## Generators

```@docs
OnlineML.Streams.SEAGenerator
OnlineML.Streams.AgrawalGenerator
OnlineML.Streams.HyperplaneGenerator
OnlineML.Streams.RandomRBFGenerator
OnlineML.Streams.LEDGenerator
OnlineML.Streams.SineGenerator
OnlineML.Streams.ConceptDriftStream
```

## Examples

### Creating Data Streams

```julia
using OnlineML.Streams: DataStream
using DataFrames

# From DataFrame
df = DataFrame(x1=randn(100), x2=randn(100), y=rand([0,1], 100))
stream = DataStream(df, target=:y)

for (x, y) in stream
    fit!(model, (x, y))
end

# From matrices
X = randn(100, 5)
y = rand(1:3, 100)
stream = DataStream(X, y)

# From generator with limit
gen = SEAGenerator()
stream = DataStream(gen, n=1000)
```

### Batching

```julia
using OnlineML.Streams: DataStream, batch_stream

gen = SEAGenerator()
stream = DataStream(gen, n=100)

for (X_batch, y_batch) in batch_stream(stream, batch_size=10)
    # X_batch is 10×d matrix
    # y_batch is length-10 vector
end
```

### Synthetic Generators

```julia
using OnlineML.Streams

# SEA - Streaming Ensemble Algorithm benchmark
sea = SEAGenerator(function_id=1, noise=0.1)

# Agrawal - demographic classification
agrawal = AgrawalGenerator(function_id=1, perturbation=0.1)

# Hyperplane - rotating decision boundary
hyperplane = HyperplaneGenerator(
    n_features=10,
    n_drift_features=2,
    mag_change=0.001
)

# RandomRBF - Gaussian centroids
rbf = RandomRBFGenerator(n_classes=3, n_features=5, n_centroids=10)

# LED - digit recognition
led = LEDGenerator(noise=0.1, irrelevant_features=true)

# Sine - regression
sine = SineGenerator(amplitude=1.0, noise=0.05)

# Generate samples
for _ in 1:1000
    x, y = generate(sea)
end
```

### Concept Drift

```julia
using OnlineML.Streams: ConceptDriftStream, RandomRBFGenerator

# Create stream with periodic drift
base = RandomRBFGenerator(n_classes=2)
stream = ConceptDriftStream(
    base,
    drift_interval=1000,
    drift_type=:sudden  # or :gradual
)
```
