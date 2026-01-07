# Synthetic data stream generators

"""
    SineGenerator

Generates a continuous sine wave stream for regression tasks.

# Parameters
- `amplitude::Float64 = 1.0` - Wave amplitude
- `frequency::Float64 = 0.1` - Wave frequency
- `noise::Float64 = 0.1` - Gaussian noise level

# Example
```jldoctest
julia> gen = SineGenerator(noise=0.1)
SineGenerator(1.0, 0.1, 0.1, 0, TaskLocalRNG())

julia> x, y = generate(gen);

julia> length(x)
1
```
"""
mutable struct SineGenerator
    amplitude::Float64
    frequency::Float64
    noise::Float64
    t::Int
    rng::Random.AbstractRNG

    function SineGenerator(;
        amplitude::Float64 = 1.0,
        frequency::Float64 = 0.1,
        noise::Float64 = 0.1,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        new(amplitude, frequency, noise, 0, rng)
    end
end

function generate(gen::SineGenerator)
    gen.t += 1
    x = Float64[gen.t]
    y = gen.amplitude * sin(gen.frequency * gen.t) + gen.noise * randn(gen.rng)
    return x, y
end


"""
    RandomRBFGenerator

Generates a stream using Random Radial Basis Functions for classification.

# Parameters
- `n_classes::Int = 2` - Number of classes
- `n_features::Int = 10` - Number of features
- `n_centroids::Int = 50` - Number of RBF centroids

# Example
```jldoctest
julia> gen = RandomRBFGenerator(n_classes=3, n_features=5);

julia> x, y = generate(gen);

julia> length(x)
5
```
"""
mutable struct RandomRBFGenerator
    n_classes::Int
    n_features::Int
    n_centroids::Int
    centroids::Vector{Vector{Float64}}
    centroid_classes::Vector{Int}
    centroid_weights::Vector{Float64}
    rng::Random.AbstractRNG

    function RandomRBFGenerator(;
        n_classes::Int = 2,
        n_features::Int = 10,
        n_centroids::Int = 50,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        gen = new(n_classes, n_features, n_centroids, [], [], [], rng)
        initialize!(gen)
        return gen
    end
end

function initialize!(gen::RandomRBFGenerator)
    gen.centroids = [rand(gen.rng, gen.n_features) for _ in 1:gen.n_centroids]
    gen.centroid_classes = rand(gen.rng, 1:gen.n_classes, gen.n_centroids)
    gen.centroid_weights = rand(gen.rng, gen.n_centroids)
    gen.centroid_weights ./= sum(gen.centroid_weights)
end

function generate(gen::RandomRBFGenerator)
    # Select centroid based on weights
    r = rand(gen.rng)
    cumsum = 0.0
    selected = 1
    for (i, w) in enumerate(gen.centroid_weights)
        cumsum += w
        if r <= cumsum
            selected = i
            break
        end
    end

    # Generate point near selected centroid
    centroid = gen.centroids[selected]
    x = centroid .+ 0.1 .* randn(gen.rng, gen.n_features)
    y = gen.centroid_classes[selected]

    return x, y
end


"""
    ConceptDriftStream

Wraps a generator and introduces concept drift.

# Parameters
- `base_generator` - The underlying generator
- `drift_interval::Int = 1000` - Observations between drifts
- `drift_type::Symbol = :sudden` - Type of drift (:sudden, :gradual, :incremental)

# Example
```jldoctest
julia> base = RandomRBFGenerator();

julia> stream = ConceptDriftStream(base, drift_interval=500);

julia> x, y = generate(stream);

julia> length(x) > 0
true
```
"""
mutable struct ConceptDriftStream
    base_generator::Any
    drift_interval::Int
    drift_type::Symbol
    t::Int

    function ConceptDriftStream(
        base_generator;
        drift_interval::Int = 1000,
        drift_type::Symbol = :sudden
    )
        new(base_generator, drift_interval, drift_type, 0)
    end
end

function generate(stream::ConceptDriftStream)
    stream.t += 1

    # Check for drift
    if stream.t % stream.drift_interval == 0
        if stream.drift_type == :sudden
            # Reinitialize the generator
            if hasmethod(initialize!, Tuple{typeof(stream.base_generator)})
                initialize!(stream.base_generator)
            end
        end
    end

    return generate(stream.base_generator)
end


"""
    LEDGenerator

LED (Light-Emitting Diode) display generator.

Generates patterns representing digits 0-9 on a 7-segment LED display,
with optional noise that flips attribute values.

# Parameters
- `noise::Float64 = 0.1` - Probability of flipping each attribute
- `irrelevant_features::Bool = true` - Include 17 irrelevant features (total 24)

# Example
```jldoctest
julia> gen = LEDGenerator(noise=0.1);

julia> x, y = generate(gen);

julia> length(x) == 24  # 7 relevant + 17 irrelevant features
true
```
"""
mutable struct LEDGenerator
    noise::Float64
    irrelevant_features::Bool
    rng::Random.AbstractRNG

    # 7-segment display patterns for digits 0-9
    # Each digit has 7 segments: [top, top-right, bottom-right, bottom, bottom-left, top-left, middle]
    patterns::Matrix{Int}

    function LEDGenerator(;
        noise::Float64 = 0.1,
        irrelevant_features::Bool = true,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        # Binary patterns for digits 0-9
        # Columns: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
        patterns = [
            1 1 1 1 1 1 0;  # 0
            0 1 1 0 0 0 0;  # 1
            1 1 0 1 1 0 1;  # 2
            1 1 1 1 0 0 1;  # 3
            0 1 1 0 0 1 1;  # 4
            1 0 1 1 0 1 1;  # 5
            1 0 1 1 1 1 1;  # 6
            1 1 1 0 0 0 0;  # 7
            1 1 1 1 1 1 1;  # 8
            1 1 1 1 0 1 1;  # 9
        ]
        new(noise, irrelevant_features, rng, patterns)
    end
end

function generate(gen::LEDGenerator)
    # Select random digit
    digit = rand(gen.rng, 0:9)
    pattern = gen.patterns[digit + 1, :]

    # Apply noise (flip bits with probability noise)
    x = Float64[]
    for val in pattern
        if rand(gen.rng) < gen.noise
            push!(x, 1.0 - val)  # Flip
        else
            push!(x, Float64(val))
        end
    end

    # Add irrelevant features if requested
    if gen.irrelevant_features
        for _ in 1:17
            push!(x, rand(gen.rng) < 0.5 ? 1.0 : 0.0)
        end
    end

    return x, digit
end


"""
    AgrawalGenerator

Agrawal generator for classification.

Generates synthetic data based on one of 10 different functions
over a set of demographic attributes.

# Parameters
- `function_id::Int = 1` - Function to use (1-10)
- `perturbation::Float64 = 0.05` - Perturbation factor

# Example
```jldoctest
julia> gen = AgrawalGenerator(function_id=1);

julia> x, y = generate(gen);

julia> length(x)
9
```
"""
mutable struct AgrawalGenerator
    function_id::Int
    perturbation::Float64
    rng::Random.AbstractRNG

    function AgrawalGenerator(;
        function_id::Int = 1,
        perturbation::Float64 = 0.05,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        @assert 1 <= function_id <= 10 "function_id must be between 1 and 10"
        new(function_id, perturbation, rng)
    end
end

function generate(gen::AgrawalGenerator)
    # Generate demographic attributes
    salary = 20000.0 + 130000.0 * rand(gen.rng)
    commission = rand(gen.rng) < 0.5 ? 0.0 : 10000.0 + 65000.0 * rand(gen.rng)
    age = rand(gen.rng, 20:80)
    elevel = rand(gen.rng, 0:4)  # Education level
    car = rand(gen.rng, 1:20)  # Car type
    zipcode = rand(gen.rng, 0:8)  # Zip code category
    hvalue = 50000.0 + 450000.0 * rand(gen.rng) + 100000.0 * zipcode  # House value
    hyears = rand(gen.rng, 1:30)  # Years in house
    loan = rand(gen.rng) * 500000.0  # Loan amount

    x = Float64[salary, commission, Float64(age), Float64(elevel),
                Float64(car), Float64(zipcode), hvalue, Float64(hyears), loan]

    # Apply perturbation
    if gen.perturbation > 0
        for i in eachindex(x)
            x[i] *= 1.0 + gen.perturbation * (2 * rand(gen.rng) - 1)
        end
    end

    # Classification function
    y = agrawal_classify(gen.function_id, salary, commission, age, elevel,
                         car, zipcode, hvalue, hyears, loan)

    return x, y
end

function agrawal_classify(func_id, salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan)
    if func_id == 1
        return age < 40 && salary < 50000 ? 0 : 1
    elseif func_id == 2
        return age < 40 && salary >= 50000 && salary <= 100000 ? 0 : 1
    elseif func_id == 3
        return age >= 40 && age <= 60 && salary < 75000 ? 0 : 1
    elseif func_id == 4
        return age >= 40 && age <= 60 && salary >= 75000 && salary <= 125000 ? 0 : 1
    elseif func_id == 5
        return age > 60 && salary < 100000 ? 0 : 1
    elseif func_id == 6
        return salary < 50000 && elevel in [0, 1] ? 0 : 1
    elseif func_id == 7
        return salary >= 75000 && elevel >= 3 ? 0 : 1
    elseif func_id == 8
        return age >= 25 && age <= 45 && (car <= 5 || car >= 15) ? 0 : 1
    elseif func_id == 9
        return hvalue >= 200000 && hyears >= 10 ? 0 : 1
    elseif func_id == 10
        return loan < 150000 && hvalue >= 300000 ? 0 : 1
    else
        return 0
    end
end


"""
    SEAGenerator

SEA (Streaming Ensemble Algorithm) concepts generator.

Generates data with 3 numerical features and a binary classification
based on different boundary conditions.

# Parameters
- `function_id::Int = 1` - Concept function (1-4)
- `noise::Float64 = 0.1` - Class noise probability

# Example
```jldoctest
julia> gen = SEAGenerator(function_id=1, noise=0.1);

julia> x, y = generate(gen);

julia> length(x)
3
```
"""
mutable struct SEAGenerator
    function_id::Int
    noise::Float64
    thresholds::Vector{Float64}
    rng::Random.AbstractRNG

    function SEAGenerator(;
        function_id::Int = 1,
        noise::Float64 = 0.1,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        @assert 1 <= function_id <= 4 "function_id must be between 1 and 4"
        thresholds = [8.0, 9.0, 7.0, 9.5]
        new(function_id, noise, thresholds, rng)
    end
end

function generate(gen::SEAGenerator)
    # Generate 3 features in [0, 10]
    x = 10.0 .* rand(gen.rng, 3)

    # Classification: f1 + f2 <= threshold
    threshold = gen.thresholds[gen.function_id]
    y = (x[1] + x[2]) <= threshold ? 0 : 1

    # Apply noise
    if rand(gen.rng) < gen.noise
        y = 1 - y
    end

    return x, y
end


"""
    HyperplaneGenerator

Generates data from a rotating hyperplane for classification.

The hyperplane boundary rotates over time to simulate gradual concept drift.

# Parameters
- `n_features::Int = 10` - Number of features
- `n_drift_features::Int = 2` - Features that drift
- `mag_change::Float64 = 0.0` - Magnitude of change per sample
- `noise::Float64 = 0.05` - Class noise probability

# Example
```jldoctest
julia> gen = HyperplaneGenerator(n_features=10, mag_change=0.001);

julia> x, y = generate(gen);

julia> length(x)
10
```
"""
mutable struct HyperplaneGenerator
    n_features::Int
    n_drift_features::Int
    mag_change::Float64
    noise::Float64
    weights::Vector{Float64}
    rng::Random.AbstractRNG

    function HyperplaneGenerator(;
        n_features::Int = 10,
        n_drift_features::Int = 2,
        mag_change::Float64 = 0.0,
        noise::Float64 = 0.05,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        weights = rand(rng, n_features)
        new(n_features, min(n_drift_features, n_features), mag_change, noise, weights, rng)
    end
end

function generate(gen::HyperplaneGenerator)
    # Generate features in [0, 1]
    x = rand(gen.rng, gen.n_features)

    # Compute sum
    total = sum(gen.weights .* x)

    # Classification
    y = total >= sum(gen.weights) / 2 ? 1 : 0

    # Apply noise
    if rand(gen.rng) < gen.noise
        y = 1 - y
    end

    # Update drifting weights
    for i in 1:gen.n_drift_features
        direction = rand(gen.rng) < 0.5 ? -1 : 1
        gen.weights[i] += direction * gen.mag_change
        gen.weights[i] = clamp(gen.weights[i], 0.0, 1.0)
    end

    return x, y
end
