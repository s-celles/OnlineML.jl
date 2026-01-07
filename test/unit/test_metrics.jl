using Test
using OnlineML
import OnlineML: Metrics
using .Metrics: Accuracy, Precision, Recall, F1Score, MAE, MSE, RMSE, R2, RollingMetric

@testset "Metrics" begin
    @testset "Classification Metrics" begin
        @testset "Accuracy" begin
            acc = Accuracy()
            @test nobs(acc) == 0
            @test value(acc) == 0.0

            # Perfect predictions
            for i in 1:10
                fit!(acc, (1, 1))
            end
            @test value(acc) == 1.0
            @test nobs(acc) == 10

            # Add some wrong predictions
            for i in 1:10
                fit!(acc, (0, 1))
            end
            @test value(acc) == 0.5

            reset!(acc)
            @test nobs(acc) == 0
        end

        @testset "Precision" begin
            prec = Precision()  # positive_class=1

            # (y_pred, y_true) format
            # True positives: predicted 1, actual 1
            for i in 1:8
                fit!(prec, (1, 1))
            end
            # False positives: predicted 1, actual 0
            for i in 1:2
                fit!(prec, (1, 0))
            end
            # True negatives (don't affect precision): predicted 0, actual 0
            for i in 1:5
                fit!(prec, (0, 0))
            end

            # Precision = TP / (TP + FP) = 8 / 10 = 0.8
            @test isapprox(value(prec), 0.8, atol=1e-6)
        end

        @testset "Recall" begin
            rec = Recall()  # positive_class=1

            # (y_pred, y_true) format
            # True positives: predicted 1, actual 1
            for i in 1:6
                fit!(rec, (1, 1))
            end
            # False negatives: predicted 0, actual 1
            for i in 1:4
                fit!(rec, (0, 1))
            end

            # Recall = TP / (TP + FN) = 6 / 10 = 0.6
            @test isapprox(value(rec), 0.6, atol=1e-6)
        end

        @testset "F1Score" begin
            f1 = F1Score()

            # Balanced case: precision=recall=0.8
            for i in 1:80
                fit!(f1, (1, 1))
            end
            for i in 1:20
                fit!(f1, (0, 1))  # FP
            end
            for i in 1:20
                fit!(f1, (1, 0))  # FN
            end

            # Precision = 80/100 = 0.8, Recall = 80/100 = 0.8
            # F1 = 2 * 0.8 * 0.8 / (0.8 + 0.8) = 0.8
            @test isapprox(value(f1), 0.8, atol=0.05)
        end
    end

    @testset "Regression Metrics" begin
        @testset "MAE" begin
            mae = MAE()
            @test nobs(mae) == 0

            # Errors of 1, 2, 3
            fit!(mae, (0.0, 1.0))  # error = 1
            fit!(mae, (0.0, 2.0))  # error = 2
            fit!(mae, (0.0, 3.0))  # error = 3

            # MAE = (1 + 2 + 3) / 3 = 2
            @test isapprox(value(mae), 2.0, atol=1e-6)
        end

        @testset "MSE" begin
            mse = MSE()

            # Errors of 1, 2, 3
            fit!(mse, (0.0, 1.0))  # squared error = 1
            fit!(mse, (0.0, 2.0))  # squared error = 4
            fit!(mse, (0.0, 3.0))  # squared error = 9

            # MSE = (1 + 4 + 9) / 3 = 14/3
            @test isapprox(value(mse), 14/3, atol=1e-6)
        end

        @testset "RMSE" begin
            rmse = RMSE()

            # Same as MSE test
            fit!(rmse, (0.0, 1.0))
            fit!(rmse, (0.0, 2.0))
            fit!(rmse, (0.0, 3.0))

            # RMSE = sqrt(14/3)
            @test isapprox(value(rmse), sqrt(14/3), atol=1e-6)
        end
    end

    @testset "Rolling Metric" begin
        @testset "RollingMetric" begin
            rolling_acc = RollingMetric(Accuracy(), window_size=5)

            # First 5 correct
            for i in 1:5
                fit!(rolling_acc, (1, 1))
            end
            @test isapprox(value(rolling_acc), 1.0, atol=1e-6)

            # Add 5 incorrect - window should now only contain these
            for i in 1:5
                fit!(rolling_acc, (0, 1))
            end
            @test isapprox(value(rolling_acc), 0.0, atol=1e-6)
        end
    end
end
