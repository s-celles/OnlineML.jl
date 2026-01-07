### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000001
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    using OnlineML
    using OnlineML.Trees: HoeffdingTree, n_nodes, height, n_leaves
    using OnlineML.Linear: LogisticRegression
    using OnlineML.Streams: RandomRBFGenerator, generate
    using OnlineML.Metrics: Accuracy
    using Plots
    using Random
end

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000002
md"""
# Hoeffding Trees for Online Classification

**Hoeffding Trees** (also known as Very Fast Decision Trees - VFDT) are the standard tree-based algorithm for streaming data. They use the Hoeffding bound to decide when sufficient data has been seen to make a split decision.

This notebook demonstrates:
- Training Hoeffding Trees on streaming data
- Comparing with linear models
- Visualizing tree growth over time
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000003
md"""
## Generate Non-Linear Data

We'll use the RandomRBF generator which creates data based on Gaussian clusters - a naturally non-linear problem where trees should excel.
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000004
begin
    Random.seed!(42)

    # Create RBF generator with multiple centroids
    gen = RandomRBFGenerator(
        n_classes=3,
        n_features=5,
        n_centroids=15,
        rng=MersenneTwister(42)
    )

    # Generate dataset
    n_samples = 5000
    rbf_data = [generate(gen) for _ in 1:n_samples]

    md"Generated **$(n_samples)** samples with **3 classes**, **5 features**, **15 centroids**"
end

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000005
md"""
## Hoeffding Tree Configuration

Key parameters:
- `grace_period`: Minimum samples before considering a split
- `split_confidence`: δ in Hoeffding bound (lower = more confident)
- `tie_threshold`: Minimum gain difference to choose best split
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000006
function train_and_track(model, data; track_tree_stats=false)
    """Train model and track accuracy + optional tree statistics."""
    accuracies = Float64[]
    tree_sizes = Int[]
    tree_depths = Int[]
    correct = 0
    total = 0

    for (x, y) in data
        if nobs(model) > 0
            y_pred = predict(model, x)
            total += 1
            correct += (y_pred == y) ? 1 : 0
            push!(accuracies, correct / total)

            if track_tree_stats && model isa HoeffdingTree
                push!(tree_sizes, n_nodes(model))
                push!(tree_depths, height(model))
            end
        end
        fit!(model, (x, y))
    end

    if track_tree_stats
        return accuracies, tree_sizes, tree_depths
    else
        return accuracies
    end
end

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000007
begin
    # Create models
    ht = HoeffdingTree(
        grace_period=100,
        delta=0.01,
        tau=0.05
    )
    lr = LogisticRegression()

    # Train both
    acc_ht, sizes, depths = train_and_track(ht, rbf_data, track_tree_stats=true)
    acc_lr = train_and_track(lr, rbf_data)

    md"Training complete!"
end

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000008
md"""
## Learning Curves Comparison
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000009
begin
    p_acc = plot(
        title="Hoeffding Tree vs Logistic Regression",
        xlabel="Samples Processed",
        ylabel="Cumulative Accuracy",
        legend=:bottomright,
        size=(700, 400)
    )

    plot!(p_acc, acc_ht, label="Hoeffding Tree", linewidth=2)
    plot!(p_acc, acc_lr, label="Logistic Regression", linewidth=2, linestyle=:dash)

    p_acc
end

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000010
md"""
## Tree Growth Over Time

Watch how the Hoeffding Tree grows as it processes more data.
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000011
begin
    p_growth = plot(layout=(1, 2), size=(800, 350))

    # Tree size
    plot!(p_growth[1], sizes,
        title="Tree Size",
        xlabel="Samples",
        ylabel="Number of Nodes",
        label="Nodes",
        linewidth=2,
        color=:blue
    )

    # Tree depth
    plot!(p_growth[2], depths,
        title="Tree Depth",
        xlabel="Samples",
        ylabel="Height",
        label="Depth",
        linewidth=2,
        color=:green
    )

    p_growth
end

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000012
md"""
## Final Tree Statistics

| Metric | Value |
|--------|-------|
| Total Nodes | $(n_nodes(ht)) |
| Tree Height | $(height(ht)) |
| Leaf Nodes | $(n_leaves(ht)) |
| Samples Processed | $(nobs(ht)) |
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000013
md"""
## Performance Comparison

| Model | Final Accuracy | Training Time |
|-------|----------------|---------------|
| Hoeffding Tree | $(round(acc_ht[end], digits=4)) | Fast |
| Logistic Regression | $(round(acc_lr[end], digits=4)) | Very Fast |

**Key insight**: Hoeffding Trees excel on non-linear data where linear models struggle.
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000014
md"""
## How Hoeffding Trees Work

The Hoeffding bound states that with probability 1-δ:

$$\bar{X} - E[X] < \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$

where:
- $\bar{X}$ is the observed mean
- $R$ is the range of the random variable
- $n$ is the number of samples
- $\delta$ is the confidence parameter

This allows the tree to make split decisions with statistical guarantees after seeing only a subset of the data.
"""

# ╔═╡ bd0e6f70-0004-11ef-0000-000000000015
md"""
## Key Takeaways

1. **Hoeffding Trees** make incremental split decisions based on statistical bounds
2. They **grow dynamically** as more data is processed
3. **Non-linear data** is handled naturally through tree splits
4. **Memory efficient** - only store sufficient statistics, not raw data
5. **Theoretically grounded** - splits are statistically optimal given the data seen
"""

# ╔═╡ Cell order:
# ╟─bd0e6f70-0004-11ef-0000-000000000002
# ╠═bd0e6f70-0004-11ef-0000-000000000001
# ╟─bd0e6f70-0004-11ef-0000-000000000003
# ╠═bd0e6f70-0004-11ef-0000-000000000004
# ╟─bd0e6f70-0004-11ef-0000-000000000005
# ╠═bd0e6f70-0004-11ef-0000-000000000006
# ╠═bd0e6f70-0004-11ef-0000-000000000007
# ╟─bd0e6f70-0004-11ef-0000-000000000008
# ╠═bd0e6f70-0004-11ef-0000-000000000009
# ╟─bd0e6f70-0004-11ef-0000-000000000010
# ╠═bd0e6f70-0004-11ef-0000-000000000011
# ╟─bd0e6f70-0004-11ef-0000-000000000012
# ╟─bd0e6f70-0004-11ef-0000-000000000013
# ╟─bd0e6f70-0004-11ef-0000-000000000014
# ╟─bd0e6f70-0004-11ef-0000-000000000015
