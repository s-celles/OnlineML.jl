# Prequential (test-then-train) evaluation functions

"""
    progressive_val_score(model, stream, metric; n_samples=1000)

Prequential (test-then-train) evaluation.

For each sample:
1. Make a prediction (test)
2. Update metric with prediction
3. Train model with true label (train)

Returns (final_score, trajectory) where trajectory is the metric
value after each sample.

# Arguments
- `model` - The online learning model
- `stream` - Iterator of (x, y) pairs or generator with `generate` function
- `metric` - The metric to track (e.g., Accuracy())
- `n_samples` - Number of samples to process

# Example
```julia
model = LogisticRegression()
gen = SEAGenerator()
metric = Accuracy()

score, trajectory = progressive_val_score(model, gen, metric, n_samples=1000)
println("Final accuracy: ", score)
plot(trajectory)  # Visualize learning curve
```
"""
function progressive_val_score(model, stream, metric; n_samples::Int=1000)
    trajectory = Float64[]

    for i in 1:n_samples
        # Get next sample
        x, y = _get_sample(stream)

        # Test - make prediction before training
        if nobs(model) > 0
            y_pred = predict(model, x)
            fit!(metric, (y_pred, y))
        end

        # Train
        fit!(model, (x, y))

        # Record trajectory
        if nobs(metric) > 0
            push!(trajectory, value(metric))
        end
    end

    final_score = nobs(metric) > 0 ? value(metric) : NaN
    return final_score, trajectory
end

# Helper to get sample from different stream types
function _get_sample(stream)
    # Use generate function - all our generators implement this
    return generate(stream)
end


"""
    holdout_score(model, train_stream, test_stream, metric; train_size=1000)

Holdout evaluation: train on train_stream, evaluate on test_stream.

# Arguments
- `model` - The online learning model
- `train_stream` - Training data iterator/generator
- `test_stream` - Test data iterator/generator
- `metric` - The metric to track
- `train_size` - Number of training samples

# Example
```julia
model = LogisticRegression()
train_gen = SEAGenerator(rng=MersenneTwister(42))
test_gen = SEAGenerator(rng=MersenneTwister(123))
metric = Accuracy()

score = holdout_score(model, train_gen, test_gen, metric, train_size=1000)
println("Holdout accuracy: ", score)
```
"""
function holdout_score(model, train_stream, test_stream, metric;
                       train_size::Int=1000, test_size::Int=500)
    # Train phase
    for i in 1:train_size
        x, y = _get_sample(train_stream)
        fit!(model, (x, y))
    end

    # Test phase
    for i in 1:test_size
        x, y = _get_sample(test_stream)
        y_pred = predict(model, x)
        fit!(metric, (y_pred, y))
    end

    return nobs(metric) > 0 ? value(metric) : NaN
end


"""
    interleaved_test_then_train(model, stream, metric; period=100)

Interleaved test-then-train with periodic holdout evaluation.

Every `period` samples, evaluates on a small holdout before training.

Returns (final_score, periodic_scores)
"""
function interleaved_test_then_train(model, stream, metric;
                                     n_samples::Int=1000, period::Int=100)
    periodic_scores = Float64[]
    holdout_metric = typeof(metric)()

    for i in 1:n_samples
        x, y = _get_sample(stream)

        # Periodic evaluation
        if i % period == 0 && nobs(model) > 0
            y_pred = predict(model, x)
            fit!(holdout_metric, (y_pred, y))
            push!(periodic_scores, value(holdout_metric))
            reset!(holdout_metric)
        end

        # Always predict and update metric
        if nobs(model) > 0
            y_pred = predict(model, x)
            fit!(metric, (y_pred, y))
        end

        # Train
        fit!(model, (x, y))
    end

    final_score = nobs(metric) > 0 ? value(metric) : NaN
    return final_score, periodic_scores
end


# Export evaluation functions
import ..OnlineML: predict
