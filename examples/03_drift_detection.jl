### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000001
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    using OnlineML
    using OnlineML.Linear: LogisticRegression
    using OnlineML.Drift: ADWIN, DDM, DriftRetrainingWrapper
    using OnlineML.Streams: SEAGenerator, ConceptDriftStream, generate
    using OnlineML.Metrics: Accuracy
    using Plots
    using Random
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000002
md"""
# Concept Drift Detection in Online Learning

**Concept drift** occurs when the relationship between inputs and outputs changes over time. This is a fundamental challenge in streaming data.

This notebook demonstrates:
- Detecting drift with **ADWIN** and **DDM**
- Automatic model retraining when drift is detected
- Visualizing drift events in learning curves
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000003
md"""
## Creating a Drifting Data Stream

We'll use the SEA Generator with concept drift. The SEA generator has 4 classification functions, and we'll switch between them to simulate drift.
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000004
begin
    Random.seed!(42)

    # Create a stream that drifts from function 1 to function 3
    stream1 = SEAGenerator(function_id=1, noise=0.1, rng=MersenneTwister(42))
    stream2 = SEAGenerator(function_id=3, noise=0.1, rng=MersenneTwister(43))

    # Generate 2000 samples: 1000 from each concept
    n_before_drift = 1000
    n_after_drift = 1000
    n_total = n_before_drift + n_after_drift

    drift_data = []
    for i in 1:n_total
        if i <= n_before_drift
            x, y = generate(stream1)
        else
            x, y = generate(stream2)
        end
        push!(drift_data, (x, y))
    end

    md"Created stream with **$(n_total)** samples, drift at sample **$(n_before_drift)**"
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000005
md"""
## Training Without Drift Detection

First, let's see what happens when we ignore drift entirely.
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000006
function train_no_drift(data)
    model = LogisticRegression()
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

    return accuracies, Int[]
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000007
md"""
## Training With ADWIN Drift Detection

ADWIN (ADaptive WINdowing) detects changes in data distribution by maintaining a variable-size window and checking for significant mean differences.
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000008
function train_with_adwin(data)
    model = LogisticRegression()
    detector = ADWIN()
    accuracies = Float64[]
    drift_points = Int[]
    correct = 0
    total = 0

    for (i, (x, y)) in enumerate(data)
        if nobs(model) > 0
            y_pred = predict(model, x)
            total += 1
            is_correct = (y_pred == y)
            correct += is_correct ? 1 : 0
            push!(accuracies, correct / total)

            # Update drift detector with error
            error = is_correct ? 0.0 : 1.0
            update!(detector, error)

            # Check for drift
            if detected_drift(detector)
                push!(drift_points, i)
                reset!(model)
                reset!(detector)
            end
        end
        fit!(model, (x, y))
    end

    return accuracies, drift_points
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000009
md"""
## Training With DDM Drift Detection

DDM (Drift Detection Method) monitors error rate and its standard deviation, triggering warnings and drift alerts based on statistical thresholds.
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000010
function train_with_ddm(data)
    model = LogisticRegression()
    detector = DDM()
    accuracies = Float64[]
    drift_points = Int[]
    correct = 0
    total = 0

    for (i, (x, y)) in enumerate(data)
        if nobs(model) > 0
            y_pred = predict(model, x)
            total += 1
            is_correct = (y_pred == y)
            correct += is_correct ? 1 : 0
            push!(accuracies, correct / total)

            # Update drift detector
            error = is_correct ? 0.0 : 1.0
            update!(detector, error)

            # Check for drift
            if detected_drift(detector)
                push!(drift_points, i)
                reset!(model)
                reset!(detector)
            end
        end
        fit!(model, (x, y))
    end

    return accuracies, drift_points
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000011
begin
    # Run all experiments
    acc_no_drift, _ = train_no_drift(drift_data)
    acc_adwin, drifts_adwin = train_with_adwin(drift_data)
    acc_ddm, drifts_ddm = train_with_ddm(drift_data)

    md"Experiments complete!"
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000012
md"""
## Learning Curves with Drift Detection
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000013
begin
    p_drift = plot(
        title="Concept Drift Detection Comparison",
        xlabel="Samples Processed",
        ylabel="Cumulative Accuracy",
        legend=:bottomright,
        size=(800, 450),
        ylim=(0.4, 1.0)
    )

    # Plot learning curves
    plot!(p_drift, acc_no_drift, label="No Drift Detection", linewidth=2, alpha=0.7)
    plot!(p_drift, acc_adwin, label="With ADWIN", linewidth=2)
    plot!(p_drift, acc_ddm, label="With DDM", linewidth=2)

    # Mark true drift point
    vline!(p_drift, [n_before_drift], label="True Drift", linestyle=:dash,
           linewidth=2, color=:black)

    # Mark detected drift points
    for d in drifts_adwin
        vline!(p_drift, [d], label="", linestyle=:dot, color=:blue, alpha=0.5)
    end
    for d in drifts_ddm
        vline!(p_drift, [d], label="", linestyle=:dot, color=:green, alpha=0.5)
    end

    p_drift
end

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000014
md"""
## Drift Detection Summary

| Method | Drifts Detected | Detection Points |
|--------|-----------------|------------------|
| No Detection | 0 | - |
| ADWIN | $(length(drifts_adwin)) | $(isempty(drifts_adwin) ? "-" : join(drifts_adwin, ", ")) |
| DDM | $(length(drifts_ddm)) | $(isempty(drifts_ddm) ? "-" : join(drifts_ddm, ", ")) |

**True drift occurred at sample $(n_before_drift)**
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000015
md"""
## Using DriftRetrainingWrapper

For convenience, OnlineML provides `DriftRetrainingWrapper` that automatically handles model retraining:

```julia
using OnlineML.Drift: DriftRetrainingWrapper, ADWIN

# Wrap any model with automatic drift handling
wrapped = DriftRetrainingWrapper(LogisticRegression(), ADWIN())

for (x, y) in stream
    fit!(wrapped, (x, y))  # Automatically resets on drift
end

println("Drifts detected: ", wrapped.n_drifts)
```
"""

# ╔═╡ ac9d5e60-0003-11ef-0000-000000000016
md"""
## Key Takeaways

1. **Concept drift** can severely degrade model performance if ignored
2. **ADWIN** uses adaptive windowing for robust drift detection
3. **DDM** monitors error rate statistics for drift signals
4. **Early detection** allows quick model adaptation
5. **DriftRetrainingWrapper** provides automatic drift handling
"""

# ╔═╡ Cell order:
# ╟─ac9d5e60-0003-11ef-0000-000000000002
# ╠═ac9d5e60-0003-11ef-0000-000000000001
# ╟─ac9d5e60-0003-11ef-0000-000000000003
# ╠═ac9d5e60-0003-11ef-0000-000000000004
# ╟─ac9d5e60-0003-11ef-0000-000000000005
# ╠═ac9d5e60-0003-11ef-0000-000000000006
# ╟─ac9d5e60-0003-11ef-0000-000000000007
# ╠═ac9d5e60-0003-11ef-0000-000000000008
# ╟─ac9d5e60-0003-11ef-0000-000000000009
# ╠═ac9d5e60-0003-11ef-0000-000000000010
# ╠═ac9d5e60-0003-11ef-0000-000000000011
# ╟─ac9d5e60-0003-11ef-0000-000000000012
# ╠═ac9d5e60-0003-11ef-0000-000000000013
# ╟─ac9d5e60-0003-11ef-0000-000000000014
# ╟─ac9d5e60-0003-11ef-0000-000000000015
# ╟─ac9d5e60-0003-11ef-0000-000000000016
