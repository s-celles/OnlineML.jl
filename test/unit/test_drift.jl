using Test
using OnlineML
import OnlineML: Drift, Linear
using .Drift: ADWIN, DDM, EDDM, PageHinkley, KSWIN, DriftDetectorWrapper, DriftRetrainingWrapper, status, drift_count
using .Linear: LogisticRegression

@testset "Drift Detection" begin
    @testset "ADWIN" begin
        detector = ADWIN()
        @test nobs(detector) == 0
        @test status(detector) == NoDrift

        # Fit with stable data
        for i in 1:100
            fit!(detector, 0.0)
        end
        @test nobs(detector) == 100
        @test status(detector) == NoDrift
        @test !detected_drift(detector)

        # Reset
        reset!(detector)
        @test nobs(detector) == 0
    end

    @testset "ADWIN drift detection on synthetic shift" begin
        detector = ADWIN(delta=0.01)

        # First period: mean = 0
        for _ in 1:500
            fit!(detector, randn() * 0.1)
        end

        # Introduce abrupt shift: mean = 5
        change_detected = false
        for _ in 1:500
            fit!(detector, 5.0 + randn() * 0.1)
            # Accept either drift or warning as detection of change
            if detected_drift(detector) || detected_warning(detector)
                change_detected = true
                break
            end
        end

        # ADWIN should detect some change (drift or warning)
        @test change_detected
    end

    @testset "DDM" begin
        detector = DDM()
        @test nobs(detector) == 0
        @test status(detector) == NoDrift

        # Fit with low error rate
        for _ in 1:100
            fit!(detector, 0.0)  # No errors
        end
        @test !detected_drift(detector)

        # Reset
        reset!(detector)
        @test nobs(detector) == 0
    end

    @testset "DDM/EDDM on error rate changes" begin
        # DDM test
        ddm = DDM(min_instances=20)

        # Low error rate period
        for _ in 1:50
            error = rand() < 0.1 ? 1.0 : 0.0  # 10% error
            fit!(ddm, error)
        end
        initial_status = status(ddm)

        # High error rate period
        drift_detected = false
        for _ in 1:100
            error = rand() < 0.8 ? 1.0 : 0.0  # 80% error
            fit!(ddm, error)
            if detected_drift(ddm)
                drift_detected = true
                break
            end
        end

        # DDM should detect the change
        @test drift_detected || status(ddm) != NoDrift

        # EDDM test
        eddm = EDDM(min_instances=20)

        # Feed errors at regular intervals
        for i in 1:100
            error = i % 10 == 0 ? 1.0 : 0.0  # Regular errors
            fit!(eddm, error)
        end
        @test nobs(eddm) > 0
    end

    @testset "PageHinkley" begin
        detector = PageHinkley()
        @test nobs(detector) == 0
        @test status(detector) == NoDrift

        # Stable period
        for _ in 1:50
            fit!(detector, 0.0)
        end
        @test !detected_drift(detector)

        # Reset
        reset!(detector)
        @test nobs(detector) == 0
    end

    @testset "PageHinkley change detection" begin
        detector = PageHinkley(min_instances=30, threshold=20.0)

        # Stable period: values around 0
        for _ in 1:50
            fit!(detector, randn() * 0.1)
        end

        # Shift: values around 5
        shift_detected = false
        for _ in 1:100
            fit!(detector, 5.0 + randn() * 0.1)
            if detected_drift(detector)
                shift_detected = true
                break
            end
        end

        @test shift_detected
    end

    @testset "DriftRetrainingWrapper auto-reset" begin
        # Create a simple wrapper
        lr = LogisticRegression()
        detector = DDM(min_instances=10)
        wrapper = DriftRetrainingWrapper(lr, detector)

        @test nobs(wrapper) == 0
        @test drift_count(wrapper) == 0

        # Fit with (x, y) + error
        for _ in 1:50
            x = randn(3)
            y = x[1] > 0 ? 1 : 0
            error = 0.0  # No errors initially
            fit!(wrapper, ((x, y), error))
        end

        @test nobs(wrapper) == 50

        # Reset
        reset!(wrapper)
        @test nobs(wrapper) == 0
        @test drift_count(wrapper) == 0
    end

    @testset "DriftRetrainingWrapper predicts" begin
        lr = LogisticRegression()
        wrapper = DriftRetrainingWrapper(lr, DDM())

        # Train
        for _ in 1:30
            x = randn(2)
            y = x[1] > 0 ? 1 : 0
            fit!(wrapper, ((x, y), 0.0))
        end

        # Predict
        y_pred = OnlineML.predict(wrapper, randn(2))
        @test y_pred in [0, 1]
    end
end
