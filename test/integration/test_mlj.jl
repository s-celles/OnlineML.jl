import MLJBase

const OnlineMLMLJExt = Base.get_extension(OnlineML, :OnlineMLMLJExt)
@test OnlineMLMLJExt !== nothing

@testset "metadata and scitypes" begin
    M = OnlineMLMLJExt.OnlineKNNClassifier
    @test MLJBase.package_uuid(M) == "e06e4760-77c8-4553-90da-06185c08ed2e"
    @test MLJBase.package_url(M) == "https://github.com/s-celles/OnlineML.jl"
    @test !MLJBase.is_wrapper(M)
    @test MLJBase.load_path(M) == "OnlineMLMLJExt.OnlineKNNClassifier"
    @test MLJBase.input_scitype(M) == MLJBase.Table(MLJBase.Continuous)
    @test MLJBase.target_scitype(M) == AbstractVector{<:MLJBase.Finite}
    @test MLJBase.target_scitype(OnlineMLMLJExt.OnlineLogisticRegression) ==
          AbstractVector{<:MLJBase.Binary}
end

@testset "categorical targets and typed predictions" begin
    X = MLJBase.table([0.0 0.0; 1.0 1.0])
    y = MLJBase.categorical(["negative", "positive"])

    deterministic_models = (
        OnlineMLMLJExt.OnlinePerceptronClassifier(),
        OnlineMLMLJExt.OnlineKNNClassifier(k=1, window_size=10),
    )
    for model in deterministic_models
        fitresult, _, _ = MLJBase.fit(model, 0, X, y)
        yhat = MLJBase.predict(model, fitresult, X)
        @test eltype(yhat) == eltype(fitresult.classes)
        @test MLJBase.classes(first(yhat)) == MLJBase.classes(first(y))
        @test OnlineML.nobs(fitresult.model) == 2
    end

    probabilistic_models = (
        OnlineMLMLJExt.OnlineLogisticRegression(),
        OnlineMLMLJExt.OnlineNaiveBayes(),
    )
    for model in probabilistic_models
        fitresult, _, _ = MLJBase.fit(model, 0, X, y)
        yhat = MLJBase.predict(model, fitresult, X)
        @test length(yhat) == 2
        @test MLJBase.classes(first(yhat)) == MLJBase.classes(first(y))
        @test OnlineML.nobs(fitresult.model) == 2
    end
end

@testset "explicit observation delta preserves fitted state" begin
    model = OnlineMLMLJExt.OnlineNaiveBayes()
    X1 = MLJBase.table([0.0 0.0; 1.0 1.0])
    y1 = MLJBase.categorical(["negative", "positive"])
    fitresult, _, _ = MLJBase.fit(model, 0, X1, y1)
    learner = fitresult.model

    X2 = MLJBase.table([0.1 0.2; 0.9 0.8])
    y2 = MLJBase.categorical(["negative", "positive"])
    returned = OnlineMLMLJExt.update_observations!(model, fitresult, X2, y2)

    @test returned === fitresult
    @test returned.model === learner
    @test OnlineML.nobs(returned.model) == 4
end

@testset "a new class is rejected before mutation" begin
    model = OnlineMLMLJExt.OnlineNaiveBayes()
    X = MLJBase.table([0.0 0.0; 1.0 1.0])
    y = MLJBase.categorical(["negative", "positive"])
    fitresult, _, _ = MLJBase.fit(model, 0, X, y)
    observations = OnlineML.nobs(fitresult.model)

    ynew = MLJBase.categorical(["unknown"])
    @test_throws ArgumentError OnlineMLMLJExt.update_observations!(
        model,
        fitresult,
        MLJBase.table([2.0 2.0]),
        ynew,
    )
    @test OnlineML.nobs(fitresult.model) == observations
end

@testset "fit rejects incompatible targets before training" begin
    model = OnlineMLMLJExt.OnlineLogisticRegression()
    X = MLJBase.table([0.0; 1.0;;])
    y = MLJBase.categorical(["a", "b", "c"])
    @test_throws ArgumentError MLJBase.fit(model, 0, X, y)
end
