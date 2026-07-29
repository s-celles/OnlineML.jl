#!/usr/bin/env python3
"""
River (Python) benchmark for comparison with OnlineML.jl

Usage:
    python benchmark/compare_river.py

Requirements:
    pip install river numpy
"""

import time
import numpy as np
from river import linear_model, tree, naive_bayes, drift, ensemble
from river.datasets import synth

# Configuration (same as Julia benchmark)
N_SAMPLES = 100_000
N_FEATURES = 10
N_CLASSES = 2
RNG_SEED = 42

def benchmark_model(name, model, stream_seed, n_samples):
    """Benchmark a River model."""
    stream = synth.RandomRBF(seed_model=stream_seed, seed_sample=stream_seed, n_classes=N_CLASSES, n_features=N_FEATURES)

    # Warmup
    for i, (x, y) in enumerate(stream):
        if i >= 100:
            break
        model.learn_one(x, y)

    # Fresh model and stream
    model = model.clone()
    stream = synth.RandomRBF(seed_model=stream_seed, seed_sample=stream_seed, n_classes=N_CLASSES, n_features=N_FEATURES)

    # Benchmark
    start = time.perf_counter()
    for i, (x, y) in enumerate(stream):
        if i >= n_samples:
            break
        model.learn_one(x, y)
    elapsed = time.perf_counter() - start

    throughput = n_samples / elapsed
    print(f"{name:30s} {n_samples:10d} samples  {elapsed:8.2f} sec  {throughput:10.0f} obs/sec")
    return throughput

def benchmark_drift(name, detector, n_samples):
    """Benchmark a drift detector."""
    np.random.seed(RNG_SEED)

    # Warmup
    for _ in range(100):
        detector.update(np.random.random())

    # Fresh detector
    detector = detector.clone()

    # Benchmark
    start = time.perf_counter()
    for i in range(n_samples):
        # Simulate error stream with drift
        error = np.random.random() * 0.3 if i < n_samples // 2 else 0.5 + np.random.random() * 0.3
        detector.update(error)
    elapsed = time.perf_counter() - start

    throughput = n_samples / elapsed
    print(f"{name:30s} {n_samples:10d} samples  {elapsed:8.2f} sec  {throughput:10.0f} obs/sec")
    return throughput

print("=" * 70)
print("River (Python) Performance Benchmark")
print("=" * 70)
print()
print(f"Configuration:")
print(f"  Samples: {N_SAMPLES}")
print(f"  Features: {N_FEATURES}")
print(f"  Classes: {N_CLASSES}")
print()

results = {}

# Linear Models
print("-" * 70)
print("LINEAR MODELS")
print("-" * 70)

results["LogisticRegression"] = benchmark_model(
    "LogisticRegression",
    linear_model.LogisticRegression(),
    RNG_SEED,
    N_SAMPLES
)

results["Perceptron"] = benchmark_model(
    "Perceptron",
    linear_model.Perceptron(),
    RNG_SEED+1,
    N_SAMPLES
)

results["LinearRegression"] = benchmark_model(
    "LinearRegression",
    linear_model.LinearRegression(),
    RNG_SEED+2,
    N_SAMPLES
)

# Tree Models
print()
print("-" * 70)
print("TREE MODELS")
print("-" * 70)

results["HoeffdingTree"] = benchmark_model(
    "HoeffdingTreeClassifier",
    tree.HoeffdingTreeClassifier(),
    RNG_SEED+3,
    N_SAMPLES
)

# Naive Bayes
print()
print("-" * 70)
print("NAIVE BAYES")
print("-" * 70)

results["GaussianNB"] = benchmark_model(
    "GaussianNB",
    naive_bayes.GaussianNB(),
    RNG_SEED+4,
    N_SAMPLES
)

# Drift Detection
print()
print("-" * 70)
print("DRIFT DETECTION")
print("-" * 70)

results["ADWIN"] = benchmark_drift("ADWIN", drift.ADWIN(), N_SAMPLES)
results["PageHinkley"] = benchmark_drift("PageHinkley", drift.PageHinkley(), N_SAMPLES)

# Summary
print()
print("=" * 70)
print("SUMMARY - River (Python)")
print("=" * 70)
print()
print(f"{'Model':<30} {'Throughput':>15}")
print("-" * 70)
for name, throughput in results.items():
    print(f"{name:<30} {throughput:>12.0f}/s")

# Save results
with open("benchmark/results_river.txt", "w") as f:
    f.write("# River (Python) Benchmark Results\n")
    f.write(f"# Samples: {N_SAMPLES}\n\n")
    for name, throughput in results.items():
        f.write(f"{name}: {throughput:.0f} obs/sec\n")

print()
print("Results saved to benchmark/results_river.txt")
print()
print("Compare with OnlineML.jl results in benchmark/results_onlineml.txt")
