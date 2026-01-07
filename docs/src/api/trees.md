# Decision Trees

Online decision tree algorithms.

## Hoeffding Tree

```@docs
OnlineML.Trees.HoeffdingTree
```

## Examples

```julia
using OnlineML.Trees: HoeffdingTree

# Create tree with custom parameters
model = HoeffdingTree(
    grace_period=200,
    split_criterion=:info_gain,
    delta=1e-7,
    max_depth=20
)

# Train incrementally
for (x, y) in data_stream
    fit!(model, (x, y))
end

# Make predictions
y_pred = predict(model, x_new)

# Inspect tree structure
println("Nodes: ", n_nodes(model))
println("Leaves: ", n_leaves(model))
println("Height: ", height(model))
```
