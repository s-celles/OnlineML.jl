"""
    Bayes

Naive Bayes classifiers for online learning.

# Exports
- `GaussianNB` - Gaussian Naive Bayes
- `MultinomialNB` - Multinomial Naive Bayes
- `BernoulliNB` - Bernoulli Naive Bayes
"""
module Bayes

using OnlineStatsBase
using OnlineStats: Variance, CountMap, Mean, mean, var
using ..OnlineML: Learner, dict_argmax
import ..OnlineML: predict, predict_proba, reset!

include("gaussian_nb.jl")
include("multinomial_nb.jl")
include("bernoulli_nb.jl")

export GaussianNB, MultinomialNB, BernoulliNB
export classes, class_prior

end # module Bayes
