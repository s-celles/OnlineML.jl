# River (Python) vs OnlineML.jl Comparison Benchmark
#
# This script benchmarks OnlineML.jl performance and provides comparison
# data for River (Python). Run River benchmarks separately with compare_river.py.
#
# Usage:
#   julia --project benchmark/compare_river.jl
#
# Expected Results (from constitution):
#   - Linear models: OnlineML >= 1M obs/sec (2x+ faster than River)
#   - Tree models: OnlineML >= 100K obs/sec (2x+ faster than River)

using OnlineML
using OnlineML.Linear: LogisticRegression, Perceptron, Regression
using Dates
using OnlineML.Trees: HoeffdingTree
using OnlineML.Bayes: GaussianNB
using OnlineML.Ensemble: Bagging
using OnlineML.Drift: ADWIN, DDM
using OnlineML.Streams: RandomRBFGenerator, SEAGenerator, generate
using OnlineML.Metrics: Accuracy
using Random
using Printf

println("=" ^ 70)
println("OnlineML.jl vs River Performance Comparison")
println("=" ^ 70)
println()

# Benchmark configuration
const N_SAMPLES = 100_000
const N_FEATURES = 10
const N_CLASSES = 2
const RNG_SEED = 42

# Helper function to benchmark throughput
function benchmark_throughput(name::String, setup_fn, train_fn, n_samples::Int)
    # Setup
    model, gen = setup_fn()

    # Warmup
    for _ in 1:100
        x, y = generate(gen)
        train_fn(model, x, y)
    end

    # Fresh model for benchmark
    model, gen = setup_fn()

    # Time
    elapsed = @elapsed begin
        for _ in 1:n_samples
            x, y = generate(gen)
            train_fn(model, x, y)
        end
    end

    throughput = n_samples / elapsed
    @printf("%-30s %10d samples  %8.2f sec  %10.0f obs/sec\n",
            name, n_samples, elapsed, throughput)

    return throughput
end

println("Configuration:")
println("  Samples: $N_SAMPLES")
println("  Features: $N_FEATURES")
println("  Classes: $N_CLASSES")
println()

println("-" ^ 70)
println("LINEAR MODELS")
println("-" ^ 70)

# LogisticRegression
lr_throughput = benchmark_throughput(
    "LogisticRegression",
    () -> (LogisticRegression(), RandomRBFGenerator(n_classes=N_CLASSES, n_features=N_FEATURES, rng=MersenneTwister(RNG_SEED))),
    (m, x, y) -> fit!(m, (x, y)),
    N_SAMPLES
)

# Perceptron
perc_throughput = benchmark_throughput(
    "Perceptron",
    () -> (Perceptron(), RandomRBFGenerator(n_classes=N_CLASSES, n_features=N_FEATURES, rng=MersenneTwister(RNG_SEED+1))),
    (m, x, y) -> fit!(m, (x, y)),
    N_SAMPLES
)

# Linear Regression
function gen_regression_sample(gen)
    x, _ = generate(gen)
    y = sum(x) + 0.1 * randn()
    return x, y
end

reg_throughput = benchmark_throughput(
    "LinearRegression",
    () -> (Regression(), RandomRBFGenerator(n_classes=2, n_features=N_FEATURES, rng=MersenneTwister(RNG_SEED+2))),
    (m, x, y) -> fit!(m, (x, Float64(y))),
    N_SAMPLES
)

println()
println("-" ^ 70)
println("TREE MODELS")
println("-" ^ 70)

# HoeffdingTree
ht_throughput = benchmark_throughput(
    "HoeffdingTree",
    () -> (HoeffdingTree(), RandomRBFGenerator(n_classes=N_CLASSES, n_features=N_FEATURES, rng=MersenneTwister(RNG_SEED+3))),
    (m, x, y) -> fit!(m, (x, y)),
    N_SAMPLES
)

println()
println("-" ^ 70)
println("NAIVE BAYES")
println("-" ^ 70)

# GaussianNB
gnb_throughput = benchmark_throughput(
    "GaussianNB",
    () -> (GaussianNB(), RandomRBFGenerator(n_classes=N_CLASSES, n_features=N_FEATURES, rng=MersenneTwister(RNG_SEED+4))),
    (m, x, y) -> fit!(m, (x, y)),
    N_SAMPLES
)

println()
println("-" ^ 70)
println("DRIFT DETECTION")
println("-" ^ 70)

# ADWIN
function benchmark_drift(name, detector_fn, n_samples)
    detector = detector_fn()

    # Warmup
    for i in 1:100
        update!(detector, rand())
    end

    detector = detector_fn()

    elapsed = @elapsed begin
        for i in 1:n_samples
            # Simulate error stream with drift
            error = i < n_samples ÷ 2 ? rand() * 0.3 : 0.5 + rand() * 0.3
            update!(detector, error)
        end
    end

    throughput = n_samples / elapsed
    @printf("%-30s %10d samples  %8.2f sec  %10.0f obs/sec\n",
            name, n_samples, elapsed, throughput)
    return throughput
end

adwin_throughput = benchmark_drift("ADWIN", () -> ADWIN(), N_SAMPLES)
ddm_throughput = benchmark_drift("DDM", () -> DDM(), N_SAMPLES)

println()
println("-" ^ 70)
println("ENSEMBLE")
println("-" ^ 70)

# Bagging with LogisticRegression
bag_throughput = benchmark_throughput(
    "Bagging(LR, n=5)",
    () -> (Bagging(() -> LogisticRegression(), n_estimators=5),
           RandomRBFGenerator(n_classes=N_CLASSES, n_features=N_FEATURES, rng=MersenneTwister(RNG_SEED+5))),
    (m, x, y) -> fit!(m, (x, y)),
    N_SAMPLES ÷ 10  # Smaller sample for ensemble
)

println()
println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println()

results = [
    ("LogisticRegression", lr_throughput, 500_000),
    ("Perceptron", perc_throughput, 1_000_000),
    ("LinearRegression", reg_throughput, 1_000_000),
    ("HoeffdingTree", ht_throughput, 100_000),
    ("GaussianNB", gnb_throughput, 500_000),
    ("ADWIN", adwin_throughput, 500_000),
    ("DDM", ddm_throughput, 1_000_000),
]

println("Model                          Throughput      Target        Status")
println("-" ^ 70)

all_pass = true
for (name, throughput, target) in results
    status = throughput >= target ? "✓ PASS" : "✗ FAIL"
    if throughput < target
        global all_pass = false
    end
    @printf("%-30s %10.0f/s    %10.0f/s    %s\n", name, throughput, target, status)
end

println()
println("-" ^ 70)

if all_pass
    println("All performance targets met!")
else
    println("Some targets not met - optimization may be needed")
end

println()
println("Note: River (Python) benchmarks should be run separately.")
println("Expected speedup: OnlineML.jl >= 2x faster than River")
println()

# Write results to file for comparison
open("benchmark/results_onlineml.txt", "w") do f
    println(f, "# OnlineML.jl Benchmark Results")
    println(f, "# Generated: $(now())")
    println(f, "# Samples: $N_SAMPLES")
    println(f, "")
    for (name, throughput, target) in results
        println(f, "$name: $throughput obs/sec (target: $target)")
    end
end

println("Results saved to benchmark/results_onlineml.txt")
