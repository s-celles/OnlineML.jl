# API Reference

This section provides detailed documentation for all OnlineML.jl modules.

## Core Types

```@docs
OnlineML.Learner
OnlineML.UnsupervisedLearner
OnlineML.Transformer
OnlineML.Detector
OnlineML.Metric
OnlineML.DriftStatus
```

## Core Functions

```@docs
OnlineML.predict
OnlineML.predict_proba
OnlineML.transform
OnlineML.fit_transform!
OnlineML.inverse_transform
OnlineML.reset!
OnlineML.update!
OnlineML.status
OnlineML.detected_drift
OnlineML.detected_warning
OnlineML.score
OnlineML.score_one
OnlineML.is_anomaly
OnlineML.fit_score!
OnlineML.fit_predict!
OnlineML.generate
```

## Modules

- [Linear Models](linear.md) - Regression, classification
- [Trees](trees.md) - Hoeffding trees
- [Ensemble](ensemble.md) - Bagging, Leveraging Bagging, Drift-Aware Bagging
- [Instance](instance.md) - KNN
- [Bayes](bayes.md) - Naive Bayes classifiers
- [Cluster](cluster.md) - KMeans
- [Anomaly](anomaly.md) - HalfSpaceTrees, GaussianProjectionDetector, RandomCutForestApproximation
- [Transform](transform.md) - Scalers, encoders
- [Drift](drift.md) - ADWIN, DDM, EDDM
- [Metrics](metrics.md) - Accuracy, F1, MAE, etc.
- [Streams](streams.md) - Generators, DataStream
- [Pipeline](pipeline.md) - Pipeline composition
