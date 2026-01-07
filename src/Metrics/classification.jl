# Online classification metrics

"""
    Accuracy <: Metric{Int}

Online accuracy metric for classification.

# Example
```jldoctest
julia> metric = Accuracy()
Accuracy: n=0 | value=0.0

julia> fit!(metric, (1, 1));

julia> fit!(metric, (0, 1));

julia> value(metric)
0.5

julia> nobs(metric)
2
```
"""
mutable struct Accuracy <: Metric{Int}
    correct::Int
    total::Int

    Accuracy() = new(0, 0)
end

function OnlineStatsBase._fit!(acc::Accuracy, xy::Tuple)
    y_pred, y_true = xy
    acc.total += 1
    if y_pred == y_true
        acc.correct += 1
    end
    return acc
end

OnlineStatsBase.value(acc::Accuracy) = acc.total > 0 ? acc.correct / acc.total : 0.0
OnlineStatsBase.nobs(acc::Accuracy) = acc.total

function reset!(acc::Accuracy)
    acc.correct = 0
    acc.total = 0
    return acc
end


"""
    Precision <: Metric{Int}

Online precision metric (positive class = 1).

precision = TP / (TP + FP)
"""
mutable struct Precision <: Metric{Int}
    true_positives::Int
    false_positives::Int
    positive_class::Int

    Precision(; positive_class::Int=1) = new(0, 0, positive_class)
end

function OnlineStatsBase._fit!(prec::Precision, xy::Tuple)
    y_pred, y_true = xy

    if y_pred == prec.positive_class
        if y_true == prec.positive_class
            prec.true_positives += 1
        else
            prec.false_positives += 1
        end
    end
    return prec
end

OnlineStatsBase.value(prec::Precision) = begin
    denom = prec.true_positives + prec.false_positives
    denom > 0 ? prec.true_positives / denom : 0.0
end
OnlineStatsBase.nobs(prec::Precision) = prec.true_positives + prec.false_positives

function reset!(prec::Precision)
    prec.true_positives = 0
    prec.false_positives = 0
    return prec
end


"""
    Recall <: Metric{Int}

Online recall metric (positive class = 1).

recall = TP / (TP + FN)
"""
mutable struct Recall <: Metric{Int}
    true_positives::Int
    false_negatives::Int
    positive_class::Int

    Recall(; positive_class::Int=1) = new(0, 0, positive_class)
end

function OnlineStatsBase._fit!(rec::Recall, xy::Tuple)
    y_pred, y_true = xy

    if y_true == rec.positive_class
        if y_pred == rec.positive_class
            rec.true_positives += 1
        else
            rec.false_negatives += 1
        end
    end
    return rec
end

OnlineStatsBase.value(rec::Recall) = begin
    denom = rec.true_positives + rec.false_negatives
    denom > 0 ? rec.true_positives / denom : 0.0
end
OnlineStatsBase.nobs(rec::Recall) = rec.true_positives + rec.false_negatives

function reset!(rec::Recall)
    rec.true_positives = 0
    rec.false_negatives = 0
    return rec
end


"""
    F1Score <: Metric{Int}

Online F1 score (harmonic mean of precision and recall).

F1 = 2 * (precision * recall) / (precision + recall)
"""
mutable struct F1Score <: Metric{Int}
    precision::Precision
    recall::Recall

    F1Score(; positive_class::Int=1) = new(
        Precision(positive_class=positive_class),
        Recall(positive_class=positive_class)
    )
end

function OnlineStatsBase._fit!(f1::F1Score, xy::Tuple)
    fit!(f1.precision, xy)
    fit!(f1.recall, xy)
    return f1
end

OnlineStatsBase.value(f1::F1Score) = begin
    p = value(f1.precision)
    r = value(f1.recall)
    (p + r) > 0 ? 2 * p * r / (p + r) : 0.0
end
OnlineStatsBase.nobs(f1::F1Score) = nobs(f1.precision) + nobs(f1.recall)

function reset!(f1::F1Score)
    reset!(f1.precision)
    reset!(f1.recall)
    return f1
end
