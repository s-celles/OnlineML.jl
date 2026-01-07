### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000001
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    using OnlineML
    using OnlineML.Transform: StandardScaler, MinMaxScaler
    using OnlineML.Linear: LogisticRegression
    using OnlineML.Pipeline: OnlinePipeline
    using OnlineML.Metrics: Accuracy
    using MLDatasets
    using DataFrames
    using Plots
    using Random
	using Statistics
end

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000002
md"""
# Pipeline Composition for Online Learning

This notebook demonstrates how to compose **transformers** and **models** into unified pipelines using OnlineML.jl.

Pipelines allow you to:
- Preprocess features (scaling, encoding) before training
- Ensure consistent transformations between training and prediction
- Build modular, reusable workflows
"""

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000003
md"""
## Load the Wine Dataset

We'll use the Wine dataset which has features at different scales, making normalization important.
"""

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000004
begin
    # Load Wine dataset (using Iris as Wine may not be available)
    # For demonstration, we'll use Iris with different scaling scenarios
    iris = Iris()

    # Prepare data with artificial scale differences
    Random.seed!(42)
    data = []
    for obs in iris
        x = Vector(obs.features)
        # Artificially scale features differently
        x_scaled = [x[1] * 100, x[2] * 0.01, x[3] * 1000, x[4] * 0.001]
        y = obs.targets.class == "Iris-setosa" ? 0 : 1
        push!(data, (x_scaled, y))
    end
    shuffle!(data)

    md"Prepared **$(length(data))** samples with artificially scaled features"
end

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000005
md"""
## Feature Scale Visualization

Let's visualize how different features have vastly different scales.
"""

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000006
begin
    # Extract feature values
    f1 = [d[1][1] for d in data]
    f2 = [d[1][2] for d in data]
    f3 = [d[1][3] for d in data]
    f4 = [d[1][4] for d in data]

    p_scales = bar(
        ["Feature 1\n(×100)", "Feature 2\n(×0.01)", "Feature 3\n(×1000)", "Feature 4\n(×0.001)"],
        [mean(f1), mean(f2), mean(f3), mean(f4)],
        title="Mean Feature Values (Different Scales)",
        ylabel="Mean Value",
        legend=false,
        size=(600, 300),
        color=[:blue, :orange, :green, :red]
    )
end

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000007
md"""
## Comparing Pipelines

We'll compare three approaches:
1. **No scaling** - Raw features
2. **StandardScaler** - Zero mean, unit variance
3. **MinMaxScaler** - Scale to [0, 1] range
"""

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000008
function train_with_pipeline(pipeline, data)
    """Train a pipeline and return accuracy trajectory."""
    accuracies = Float64[]
    correct = 0
    total = 0

    for (x, y) in data
        # Test-then-train
        if nobs(pipeline) > 0
            y_pred = predict(pipeline, x)
            total += 1
            if y_pred == y
                correct += 1
            end
            push!(accuracies, correct / total)
        end

        # Update pipeline (handles scaling + training)
        fit!(pipeline, (x, y))
    end

    return accuracies
end

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000009
begin
    # Create three pipelines
    pipeline_raw = LogisticRegression()
    pipeline_std = StandardScaler() |> LogisticRegression()
    pipeline_minmax = MinMaxScaler() |> LogisticRegression()

    # Train each
    acc_raw = train_with_pipeline(pipeline_raw, data)
    acc_std = train_with_pipeline(pipeline_std, data)
    acc_minmax = train_with_pipeline(pipeline_minmax, data)

    md"All pipelines trained!"
end

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000010
md"""
## Learning Curves Comparison
"""

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000011
begin
    p_compare = plot(
        title="Effect of Feature Scaling on Online Learning",
        xlabel="Samples Processed",
        ylabel="Cumulative Accuracy",
        legend=:bottomright,
        size=(700, 400),
        ylim=(0, 1.05)
    )

    plot!(p_compare, acc_raw, label="No Scaling", linewidth=2, linestyle=:dash)
    plot!(p_compare, acc_std, label="StandardScaler |> LR", linewidth=2)
    plot!(p_compare, acc_minmax, label="MinMaxScaler |> LR", linewidth=2)

    p_compare
end

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000012
md"""
## Pipeline Composition Syntax

OnlineML.jl supports intuitive pipeline creation with the `|>` operator:

```julia
# Single transformer + model
pipeline = StandardScaler() |> LogisticRegression()

# Multiple transformers
pipeline = StandardScaler() |> OneHotEncoder() |> LogisticRegression()

# Training is seamless
for (x, y) in stream
    fit!(pipeline, (x, y))  # Transforms are applied automatically
end

# Prediction flows through all steps
y_pred = predict(pipeline, x_new)
```
"""

# ╔═╡ 9b8c4d50-0002-11ef-0000-000000000013
md"""
## Results

| Pipeline | Final Accuracy |
|----------|----------------|
| No Scaling | $(round(acc_raw[end], digits=3)) |
| StandardScaler |> LR | $(round(acc_std[end], digits=3)) |
| MinMaxScaler |> LR | $(round(acc_minmax[end], digits=3)) |

**Conclusion**: Feature scaling significantly improves learning when features have different scales.
"""

# ╔═╡ Cell order:
# ╟─9b8c4d50-0002-11ef-0000-000000000002
# ╠═9b8c4d50-0002-11ef-0000-000000000001
# ╟─9b8c4d50-0002-11ef-0000-000000000003
# ╠═9b8c4d50-0002-11ef-0000-000000000004
# ╟─9b8c4d50-0002-11ef-0000-000000000005
# ╠═9b8c4d50-0002-11ef-0000-000000000006
# ╟─9b8c4d50-0002-11ef-0000-000000000007
# ╠═9b8c4d50-0002-11ef-0000-000000000008
# ╠═9b8c4d50-0002-11ef-0000-000000000009
# ╟─9b8c4d50-0002-11ef-0000-000000000010
# ╠═9b8c4d50-0002-11ef-0000-000000000011
# ╟─9b8c4d50-0002-11ef-0000-000000000012
# ╟─9b8c4d50-0002-11ef-0000-000000000013
