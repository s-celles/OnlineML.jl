# Transformers

Online feature preprocessing transformers.

## Scalers

```@docs
OnlineML.Transform.StandardScaler
OnlineML.Transform.MinMaxScaler
OnlineML.Transform.MaxAbsScaler
```

## Encoders

```@docs
OnlineML.Transform.OneHotEncoder
OnlineML.Transform.OrdinalEncoder
OnlineML.Transform.TargetEncoder
```

## Imputers

```@docs
OnlineML.Transform.MeanImputer
OnlineML.Transform.ModeImputer
```

## Examples

### Standard Scaler

```julia
using OnlineML.Transform: StandardScaler

scaler = StandardScaler()

for x in data_stream
    x_scaled = fit_transform!(scaler, x)
end

# For new data (don't update statistics)
x_new_scaled = transform(scaler, x_new)
```

### One-Hot Encoder

```julia
using OnlineML.Transform: OneHotEncoder

encoder = OneHotEncoder()

for x in categorical_stream
    x_encoded = fit_transform!(encoder, x)
end
```
