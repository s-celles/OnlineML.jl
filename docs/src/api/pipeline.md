# Pipeline

Pipeline composition for online ML workflows.

## OnlinePipeline

```@docs
OnlineML.Pipeline.OnlinePipeline
```

## TransformerUnion

```@docs
OnlineML.Pipeline.TransformerUnion
```

## ColumnTransformer

```@docs
OnlineML.Pipeline.ColumnTransformer
```

## Examples

### Basic Pipeline

```julia
using OnlineML.Pipeline
using OnlineML.Transform: StandardScaler
using OnlineML.Linear: LogisticRegression

# Create pipeline with |> operator
pipeline = StandardScaler() |> LogisticRegression()

# Train
for (x, y) in data_stream
    fit!(pipeline, (x, y))
end

# Predict
y_pred = predict(pipeline, x_new)
```

### Multi-Step Pipeline

```julia
using OnlineML.Pipeline
using OnlineML.Transform: StandardScaler, OneHotEncoder
using OnlineML.Linear: LogisticRegression

# Chain multiple transformers
pipeline = StandardScaler() |> OneHotEncoder() |> LogisticRegression()
```

### Parallel Feature Processing

```julia
using OnlineML.Pipeline: TransformerUnion
using OnlineML.Transform: StandardScaler, MinMaxScaler

# Apply multiple transformers and concatenate
union = TransformerUnion(StandardScaler(), MinMaxScaler())

for x in data_stream
    x_transformed = fit_transform!(union, x)
    # x_transformed contains both scaled versions
end
```

### Column Selection

```julia
using OnlineML.Pipeline: ColumnTransformer
using OnlineML.Transform: StandardScaler, OneHotEncoder

# Apply different transformers to different columns
ct = ColumnTransformer(
    StandardScaler() => 1:5,      # Scale columns 1-5
    OneHotEncoder() => 6:10       # Encode columns 6-10
)
```
