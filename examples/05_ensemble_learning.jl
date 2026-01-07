### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ ce1f8080-0005-11ef-0000-000000000001
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    using OnlineML
    using OnlineML.Linear: LogisticRegression
    using OnlineML.Trees: HoeffdingTree
    using OnlineML.Ensemble: Bagging, AdaptiveRandomForest
    using OnlineML.Streams: SEAGenerator, generate
    using OnlineML.Metrics: Accuracy
    using MLDatasets
    using DataFrames
    using Plots
    using Random
end

# ╔═╡ ce1f8080-0005-11ef-0000-000000000002
md"""
# Online Ensemble Learning

**Ensemble methods** combine multiple models to improve predictive performance. In online learning, ensembles face unique challenges:
- Models must be updated incrementally
- Diversity must be maintained over time
- Memory and computation are constrained

This notebook demonstrates:
- **Online Bagging** with Poisson(λ=1) resampling
- **Adaptive Random Forest (ARF)** with drift detection
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000003
md"""
## Load the Titanic Dataset

We'll use the Titanic survival dataset - a classic binary classification problem.
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000004
begin
    Random.seed!(42)

    # Load Titanic dataset
    titanic = Titanic()

    # Prepare data (handle missing values, encode categoricals)
    titanic_data = []
    for obs in titanic
        features = obs.features
        target = obs.targets.Survived

        # Skip rows with missing target
        if ismissing(target)
            continue
        end

        # Extract numeric features (Age, Fare, SibSp, Parch, Pclass)
        age = ismissing(features.Age) ? 30.0 : Float64(features.Age)
        fare = ismissing(features.Fare) ? 32.0 : Float64(features.Fare)
        sibsp = Float64(features.SibSp)
        parch = Float64(features.Parch)
        pclass = Float64(features.Pclass)

        # Encode Sex
        sex = features.Sex == "male" ? 1.0 : 0.0

        x = [age, fare, sibsp, parch, pclass, sex]
        y = Int(target)

        push!(titanic_data, (x, y))
    end

    # Shuffle data
    shuffle!(titanic_data)

    md"Prepared **$(length(titanic_data))** samples from Titanic dataset"
end

# ╔═╡ ce1f8080-0005-11ef-0000-000000000005
md"""
## Online Bagging

Online Bagging simulates bootstrap sampling using Poisson(λ=1) weights. Each base model receives each sample k times, where k ~ Poisson(1).

This maintains the diversity of traditional bagging while supporting incremental updates.
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000006
function train_model(model, data)
    """Train a model and return accuracy trajectory."""
    accuracies = Float64[]
    correct = 0
    total = 0

    for (x, y) in data
        if nobs(model) > 0
            y_pred = predict(model, x)
            total += 1
            correct += (y_pred == y) ? 1 : 0
            push!(accuracies, correct / total)
        end
        fit!(model, (x, y))
    end

    return accuracies
end

# ╔═╡ ce1f8080-0005-11ef-0000-000000000007
begin
    # Create models
    single_lr = LogisticRegression()
    single_ht = HoeffdingTree()

    # Bagging ensembles
    bag_lr = Bagging(() -> LogisticRegression(), n_estimators=10)
    bag_ht = Bagging(() -> HoeffdingTree(), n_estimators=10)

    # Train all
    acc_single_lr = train_model(single_lr, titanic_data)
    acc_single_ht = train_model(single_ht, titanic_data)
    acc_bag_lr = train_model(bag_lr, titanic_data)
    acc_bag_ht = train_model(bag_ht, titanic_data)

    md"Training complete!"
end

# ╔═╡ ce1f8080-0005-11ef-0000-000000000008
md"""
## Learning Curves: Single vs Ensemble
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000009
begin
    p_ensemble = plot(
        title="Single Model vs Online Bagging (Titanic)",
        xlabel="Samples Processed",
        ylabel="Cumulative Accuracy",
        legend=:bottomright,
        size=(700, 400),
        ylim=(0.5, 0.9)
    )

    plot!(p_ensemble, acc_single_lr, label="Single LR", linewidth=2, linestyle=:dash, alpha=0.7)
    plot!(p_ensemble, acc_bag_lr, label="Bagging LR (10)", linewidth=2)
    plot!(p_ensemble, acc_single_ht, label="Single HT", linewidth=2, linestyle=:dash, alpha=0.7)
    plot!(p_ensemble, acc_bag_ht, label="Bagging HT (10)", linewidth=2)

    p_ensemble
end

# ╔═╡ ce1f8080-0005-11ef-0000-000000000010
md"""
## Adaptive Random Forest

ARF extends online bagging with:
- **Drift detection** per tree using ADWIN
- **Background learners** that replace drifted trees
- **Random subspace method** for feature diversity
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000011
md"""
## Ensemble Size Comparison

Let's see how ensemble size affects performance.
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000012
begin
    # Train ensembles of different sizes
    sizes = [1, 3, 5, 10, 20]
    final_accs = Float64[]

    for n in sizes
        model = Bagging(() -> LogisticRegression(), n_estimators=n)
        acc = train_model(model, titanic_data)
        push!(final_accs, acc[end])
    end

    p_size = bar(
        string.(sizes),
        final_accs,
        title="Effect of Ensemble Size",
        xlabel="Number of Base Learners",
        ylabel="Final Accuracy",
        legend=false,
        size=(500, 350),
        ylim=(0.6, 0.8)
    )
end

# ╔═╡ ce1f8080-0005-11ef-0000-000000000013
md"""
## Performance Summary

| Model | Ensemble Size | Final Accuracy |
|-------|---------------|----------------|
| Single LR | 1 | $(round(acc_single_lr[end], digits=4)) |
| Bagging LR | 10 | $(round(acc_bag_lr[end], digits=4)) |
| Single HT | 1 | $(round(acc_single_ht[end], digits=4)) |
| Bagging HT | 10 | $(round(acc_bag_ht[end], digits=4)) |
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000014
md"""
## How Online Bagging Works

Traditional bagging draws bootstrap samples. Online Bagging simulates this by:

1. For each incoming sample (x, y):
2. For each base learner m:
   - Draw k ~ Poisson(λ=1)
   - Train m on (x, y) exactly k times

This produces equivalent diversity to bootstrap sampling in expectation.

```julia
# OnlineML.jl implementation
bag = Bagging(() -> LogisticRegression(), n_estimators=10)

for (x, y) in stream
    fit!(bag, (x, y))  # Each base model gets Poisson-weighted updates
end

y_pred = predict(bag, x_new)  # Majority vote
```
"""

# ╔═╡ ce1f8080-0005-11ef-0000-000000000015
md"""
## Key Takeaways

1. **Online Bagging** uses Poisson(1) resampling to simulate bootstrap
2. **Ensembles reduce variance** and improve generalization
3. **Diminishing returns** after ~10 base learners
4. **ARF adds drift adaptation** for non-stationary streams
5. **Diverse base learners** (trees vs linear) can be combined
"""

# ╔═╡ Cell order:
# ╟─ce1f8080-0005-11ef-0000-000000000002
# ╠═ce1f8080-0005-11ef-0000-000000000001
# ╟─ce1f8080-0005-11ef-0000-000000000003
# ╠═ce1f8080-0005-11ef-0000-000000000004
# ╟─ce1f8080-0005-11ef-0000-000000000005
# ╠═ce1f8080-0005-11ef-0000-000000000006
# ╠═ce1f8080-0005-11ef-0000-000000000007
# ╟─ce1f8080-0005-11ef-0000-000000000008
# ╠═ce1f8080-0005-11ef-0000-000000000009
# ╟─ce1f8080-0005-11ef-0000-000000000010
# ╟─ce1f8080-0005-11ef-0000-000000000011
# ╠═ce1f8080-0005-11ef-0000-000000000012
# ╟─ce1f8080-0005-11ef-0000-000000000013
# ╟─ce1f8080-0005-11ef-0000-000000000014
# ╟─ce1f8080-0005-11ef-0000-000000000015
