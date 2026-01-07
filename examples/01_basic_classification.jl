### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000001
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
	Pkg.add(["Plots", "MLDatasets", "DataFrames"])
    using OnlineML
    using OnlineML.Linear: LogisticRegression, Perceptron
    using OnlineML.Bayes: GaussianNB
    using OnlineML.Metrics: Accuracy, progressive_val_score
    using MLDatasets
    using DataFrames
    using Plots
    using Random
end

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000002
md"""
# Basic Online Classification with Classic Datasets

This notebook demonstrates online classification using the **Iris** dataset from MLDatasets.jl.

We'll compare three online classifiers:
- **Logistic Regression** - Linear classifier with gradient descent
- **Perceptron** - Simple linear classifier
- **Gaussian Naive Bayes** - Probabilistic classifier
"""

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000003
md"""
## Load the Iris Dataset

The Iris dataset contains 150 samples with 4 features each, classified into 3 species.
"""

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000004
begin
    # Load Iris dataset
    iris = Iris()

    # Convert to iterable format
    n_samples = length(iris)

    # Create class mapping (string -> int)
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    # class_names = String.(unique(getproperty.(DataFrame(iris)[!, :targets], :class)))
    class_map = Dict(name => i-1 for (i, name) in enumerate(class_names))

    md"Loaded **$(n_samples)** samples from Iris dataset"
end

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000005
md"""
## Online Learning with Test-Then-Train

We use **prequential evaluation** (test-then-train):
1. Predict on new sample
2. Record error
3. Update model with sample
4. Repeat

This gives an honest estimate of online performance.
"""

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000006
function train_online(model, iris, class_map; shuffle_seed=42)
    """Train a model online and return accuracy trajectory."""
    Random.seed!(shuffle_seed)

    # Collect and shuffle data
    data = [(Vector(obs.features), class_map[obs.targets.class])
            for obs in iris]
    shuffle!(data)

    # Track accuracy over time
    accuracies = Float64[]
    correct = 0
    total = 0

    for (x, y) in data
        # Test-then-train
        if nobs(model) > 0
            y_pred = predict(model, x)
            total += 1
            if y_pred == y
                correct += 1
            end
            push!(accuracies, correct / total)
        end

        # Update model
        fit!(model, (x, y))
    end

    return accuracies
end

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000007
begin
    # Train all three models
    lr_acc = train_online(LogisticRegression(), iris, class_map)
    perceptron_acc = train_online(Perceptron(), iris, class_map)
    gnb_acc = train_online(GaussianNB(), iris, class_map)

    md"Training complete!"
end

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000008
md"""
## Learning Curves

The plot below shows how accuracy evolves as each model processes more samples.
"""

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000009
begin
    p = plot(
        title="Online Classification on Iris Dataset",
        xlabel="Samples Processed",
        ylabel="Cumulative Accuracy",
        legend=:bottomright,
        size=(700, 400),
        ylim=(0, 1.05)
    )

    plot!(p, lr_acc, label="Logistic Regression", linewidth=2)
    plot!(p, perceptron_acc, label="Perceptron", linewidth=2)
    plot!(p, gnb_acc, label="Gaussian NB", linewidth=2)

    p
end

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000010
md"""
## Final Results

| Model | Final Accuracy |
|-------|----------------|
| Logistic Regression | $(round(lr_acc[end], digits=3)) |
| Perceptron | $(round(perceptron_acc[end], digits=3)) |
| Gaussian NB | $(round(gnb_acc[end], digits=3)) |
"""

# ╔═╡ 8a7b3c40-0001-11ef-0000-000000000011
md"""
## Key Takeaways

1. **Online learning** updates models incrementally - no need to store all data
2. **Prequential evaluation** gives honest performance estimates
3. Different algorithms have different learning dynamics
4. **GaussianNB** often learns faster due to probabilistic modeling
"""

# ╔═╡ Cell order:
# ╟─8a7b3c40-0001-11ef-0000-000000000002
# ╠═8a7b3c40-0001-11ef-0000-000000000001
# ╟─8a7b3c40-0001-11ef-0000-000000000003
# ╠═8a7b3c40-0001-11ef-0000-000000000004
# ╟─8a7b3c40-0001-11ef-0000-000000000005
# ╠═8a7b3c40-0001-11ef-0000-000000000006
# ╠═8a7b3c40-0001-11ef-0000-000000000007
# ╟─8a7b3c40-0001-11ef-0000-000000000008
# ╠═8a7b3c40-0001-11ef-0000-000000000009
# ╟─8a7b3c40-0001-11ef-0000-000000000010
# ╟─8a7b3c40-0001-11ef-0000-000000000011
