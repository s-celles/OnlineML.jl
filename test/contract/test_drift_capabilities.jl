using OnlineML
using OnlineML.Drift:
    ADWIN,
    DDM,
    EDDM,
    KSWIN,
    PageHinkley,
    current_window,
    status,
    window_size

@testset "detector lifecycle" begin
    detectors = (
        ADWIN(),
        DDM(min_instances=10),
        EDDM(min_instances=5),
        PageHinkley(min_instances=5, threshold=5.0),
        KSWIN(window_size=20, stat_size=10),
    )

    for detector in detectors
        @test nobs(detector) == 0
        @test status(detector) == NoDrift
        @test update!(detector, 0.0) isa DriftStatus
        @test nobs(detector) == 1
        @test reset!(detector) === detector
        @test nobs(detector) == 0
        @test status(detector) == NoDrift
        @test !detected_drift(detector)
        @test !detected_warning(detector)
    end
end

@testset "constructor and input validation" begin
    @test_throws ArgumentError ADWIN(delta=0.0)
    @test_throws ArgumentError ADWIN(delta=1.0)
    @test_throws ArgumentError ADWIN(min_window_length=0)
    @test_throws ArgumentError DDM(warning_level=0.0)
    @test_throws ArgumentError DDM(warning_level=3.0, drift_level=2.0)
    @test_throws ArgumentError DDM(min_instances=0)
    @test_throws ArgumentError EDDM(drift_level=0.96, warning_level=0.95)
    @test_throws ArgumentError EDDM(min_instances=0)
    @test_throws ArgumentError PageHinkley(min_instances=0)
    @test_throws ArgumentError PageHinkley(delta=-0.1)
    @test_throws ArgumentError PageHinkley(threshold=0.0)
    @test_throws ArgumentError PageHinkley(alpha=1.0)

    @test_throws ArgumentError fit!(DDM(), 0.5)
    @test_throws ArgumentError fit!(EDDM(), -1.0)
    @test_throws ArgumentError fit!(ADWIN(), Inf)
end

@testset "ADWIN retains stable data and shrinks after an abrupt shift" begin
    detector = ADWIN(delta=0.01, min_window_length=5)
    stable_states = [update!(detector, 0.0) for _ in 1:40]
    @test all(==(NoDrift), stable_states)
    @test window_size(detector) == 40
    @test nobs(detector) == 40

    shifted_states = [update!(detector, 1.0) for _ in 1:40]
    @test DriftDetected in shifted_states
    @test window_size(detector) < nobs(detector)
    @test window_size(detector) <= 40
    @test value(detector) > 0.9
    @test !detected_warning(detector)
end

@testset "DDM stable errors and deterministic shift" begin
    stable = DDM(min_instances=10)
    for error in zeros(40)
        @test update!(stable, error) == NoDrift
    end
    @test nobs(stable) == 40

    shifted = DDM(min_instances=20)
    low_error_period = repeat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 5)
    high_error_period = repeat([1.0, 1.0, 1.0, 1.0, 0.0], 20)
    foreach(error -> fit!(shifted, error), low_error_period)

    events = DriftStatus[]
    for error in high_error_period
        state = update!(shifted, error)
        state == NoDrift || push!(events, state)
        state == DriftDetected && break
    end
    @test Warning in events
    @test DriftDetected in events
end

@testset "PageHinkley reports one abrupt shift, not a stable level" begin
    detector = PageHinkley(min_instances=10, threshold=5.0)
    foreach(x -> fit!(detector, x), zeros(30))

    drift_indices = Int[]
    for (i, x) in enumerate(fill(5.0, 40))
        update!(detector, x) == DriftDetected && push!(drift_indices, i)
    end

    @test length(drift_indices) == 1
    @test first(drift_indices) <= 10
    @test status(detector) == NoDrift
end

@testset "KSWIN has a bounded window and deterministic shift" begin
    detector = KSWIN(alpha=0.01, window_size=20, stat_size=10)
    foreach(x -> fit!(detector, x), zeros(20))
    @test window_size(detector) == 20
    @test current_window(detector) == zeros(20)

    states = [update!(detector, x) for x in ones(20)]
    @test DriftDetected in states
    @test window_size(detector) == 20
    @test current_window(detector) == ones(20)
    @test nobs(detector) == 40
end
