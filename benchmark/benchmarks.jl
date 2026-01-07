using BenchmarkTools
using OnlineML
using OnlineML.Linear: Regression, LogisticRegression
using OnlineML.Trees: HoeffdingTree
using OnlineML.Metrics: Accuracy
using Random

const SUITE = BenchmarkGroup()

# Linear model benchmarks
SUITE["linear"] = BenchmarkGroup()

function benchmark_linear_regression(n_obs, n_features)
    model = Regression()
    X = randn(n_obs, n_features)
    y = randn(n_obs)
    for i in 1:n_obs
        fit!(model, X[i, :], y[i])
    end
    return model
end

SUITE["linear"]["regression_1M_d100"] = @benchmarkable benchmark_linear_regression(1_000_000, 100)
SUITE["linear"]["regression_100K_d10"] = @benchmarkable benchmark_linear_regression(100_000, 10)

# Tree benchmarks
SUITE["trees"] = BenchmarkGroup()

function benchmark_hoeffding_tree(n_obs, n_features, n_classes)
    rng = MersenneTwister(42)
    model = HoeffdingTree()
    X = randn(rng, n_obs, n_features)
    y = rand(rng, 1:n_classes, n_obs)
    for i in 1:n_obs
        fit!(model, X[i, :], y[i])
    end
    return model
end

SUITE["trees"]["hoeffding_100K_d10_c3"] = @benchmarkable benchmark_hoeffding_tree(100_000, 10, 3)

# Allocation tests
SUITE["allocations"] = BenchmarkGroup()

function test_allocation_linear()
    model = Regression()
    x = randn(10)
    y = randn()
    # Warm up
    fit!(model, x, y)
    # Measure
    return @allocated fit!(model, x, y)
end

SUITE["allocations"]["linear_fit"] = @benchmarkable test_allocation_linear()

# Run benchmarks
# tune!(SUITE)
# results = run(SUITE, verbose=true)
# show(results)
