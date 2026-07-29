# Metrics

Online evaluation metrics.

## Classification Metrics

```@docs
OnlineML.Metrics.Accuracy
OnlineML.Metrics.Precision
OnlineML.Metrics.Recall
OnlineML.Metrics.F1Score
```

## Regression Metrics

```@docs
OnlineML.Metrics.MAE
OnlineML.Metrics.MSE
OnlineML.Metrics.RMSE
OnlineML.Metrics.R2
```

## Rolling Metrics

```@docs
OnlineML.Metrics.RollingMetric
```

## Evaluation Functions

```@docs
OnlineML.Metrics.progressive_val_score
OnlineML.Metrics.holdout_score
OnlineML.Metrics.interleaved_test_then_train
```

## Examples

### Basic Usage

```julia
using OnlineML.Metrics: Accuracy, F1Score

acc = Accuracy()
f1 = F1Score()

for (y_pred, y_true) in predictions
    fit!(acc, (y_pred, y_true))
    fit!(f1, (y_pred, y_true))
end

println("Accuracy: ", value(acc))
println("F1 Score: ", value(f1))
```

### Rolling Window Metric

```julia
using OnlineML.Metrics: RollingMetric, Accuracy

# Track accuracy over last 100 predictions
metric = RollingMetric(Accuracy, window_size=100)

for (y_pred, y_true) in predictions
    fit!(metric, (y_pred, y_true))
    println("Rolling accuracy: ", value(metric))
end
```

### Prequential Evaluation

```julia
using OnlineML.Metrics: progressive_val_score, Accuracy
using OnlineML.Linear: LogisticRegression
using OnlineML.Streams: SEAGenerator

model = LogisticRegression()
gen = SEAGenerator()
metric = Accuracy()

score, trajectory = progressive_val_score(model, gen, metric, n_samples=1000)
```
