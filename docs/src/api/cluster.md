# Clustering

Online clustering algorithms.

## Streaming KMeans

```@docs
OnlineML.Cluster.StreamingKMeans
```

## Examples

```julia
using OnlineML.Cluster: StreamingKMeans

model = StreamingKMeans(k=3)

for x in data_stream
    fit!(model, x)
end

cluster = predict(model, x_new)
centers = value(model)
```
