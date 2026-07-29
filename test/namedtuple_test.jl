@testset "NamedTuple Input" begin
    # T008: Test fit! supervised learner with NamedTuple and target symbol
    @testset "Supervised Learner with NamedTuple" begin
        model = LogisticRegression()
        fit!(model, (; x1=1.0, x2=2.0, y=1), :y)
        @test nobs(model) == 1
    end

    # T009: Test nobs increments correctly after fit!
    @testset "nobs increments correctly" begin
        model = LogisticRegression()
        @test nobs(model) == 0
        fit!(model, (; x1=1.0, x2=2.0, y=1), :y)
        @test nobs(model) == 1
        fit!(model, (; x1=0.5, x2=1.5, y=0), :y)
        @test nobs(model) == 2
    end

    # T010: Test multiple sequential NamedTuple fits accumulate
    @testset "Sequential fits accumulate" begin
        model = Perceptron()
        data = [
            (; a=1.0, b=2.0, label=1),
            (; a=0.5, b=1.5, label=0),
            (; a=2.0, b=3.0, label=1),
            (; a=0.1, b=0.2, label=0),
        ]
        for row in data
            fit!(model, row, :label)
        end
        @test nobs(model) == 4
    end

    # T011: Test ArgumentError for missing target column
    @testset "ArgumentError for missing target" begin
        model = LogisticRegression()
        @test_throws ArgumentError fit!(model, (; x1=1.0, x2=2.0), :missing_col)
    end

    # T012: Test ArgumentError for no features after target exclusion
    @testset "ArgumentError for no features" begin
        model = LogisticRegression()
        @test_throws ArgumentError fit!(model, (; y=1), :y)
    end

    # T016: Test iterate stream of NamedTuples
    @testset "Stream of NamedTuples" begin
        model = LogisticRegression()
        stream = [(; x=1.0, y=0), (; x=2.0, y=1), (; x=3.0, y=0)]
        for row in stream
            fit!(model, row, :y)
        end
        @test nobs(model) == 3
    end

    # T017: Test fit! from Tables.rows produces equivalent results to table-based fit!
    @testset "Tables.rows integration" begin
        using DataFrames
        df = DataFrame(feature1=[1.0, 2.0, 3.0], feature2=[4.0, 5.0, 6.0], label=[0, 1, 0])

        # Method 1: Use existing table-based fit!
        model1 = LogisticRegression()
        fit!(model1, df, :label)

        # Method 2: Convert rows to NamedTuples and use NamedTuple fit!
        model2 = LogisticRegression()
        for row in Tables.rows(df)
            # Convert Tables.AbstractRow to NamedTuple
            nt = (; (k => getproperty(row, k) for k in propertynames(row))...)
            fit!(model2, nt, :label)
        end

        @test nobs(model1) == 3
        @test nobs(model2) == 3
    end

    # T020: Test unsupervised learner with NamedTuple (all fields as features)
    @testset "Unsupervised Learner with NamedTuple" begin
        model = StreamingKMeans(k=3)
        fit!(model, (; x1=1.0, x2=2.0))
        @test nobs(model) == 1
    end

    # T021: Test unsupervised learner with exclude parameter
    @testset "Unsupervised with exclude" begin
        model = StreamingKMeans(k=3)
        fit!(model, (; x1=1.0, x2=2.0, id=1); exclude=[:id])
        @test nobs(model) == 1
    end

    # T022: Test ArgumentError for empty features after exclusion (unsupervised)
    @testset "ArgumentError for empty features (unsupervised)" begin
        model = StreamingKMeans(k=3)
        @test_throws ArgumentError fit!(model, (; id=1); exclude=[:id])
    end
end
