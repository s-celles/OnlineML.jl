# ExtremelyFastTree (EFDT) implementation
# Extends HoeffdingTree with split re-evaluation for improved accuracy

"""
    ExtremelyFastTree{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Extremely Fast Decision Tree (EFDT) for streaming classification.

EFDT improves on HoeffdingTree by re-evaluating splits at internal nodes.
When a better split is found, the subtree is replaced, leading to
improved accuracy especially on non-stationary data.

# Parameters
- `grace_period::Int = 200` - Observations between split evaluations
- `re_eval_period::Int = 200` - Observations between re-evaluations at split nodes
- `delta::Float64 = 1e-7` - Hoeffding bound confidence parameter
- `tau::Float64 = 0.05` - Tie-breaking threshold
- `split_criterion::Symbol = :info_gain` - `:gini`, `:info_gain`, or `:variance`
- `leaf_prediction::Symbol = :nba` - `:mc` (majority), `:nb` (naive bayes), `:nba` (adaptive)
- `max_depth::Union{Int,Nothing} = nothing` - Maximum tree depth (unlimited if nothing)

# Example
```jldoctest
julia> tree = ExtremelyFastTree(grace_period=100, delta=1e-5)
ExtremelyFastTree: n=0 | nodes=1 | leaves=1 | replacements=0

julia> fit!(tree, ([1.0, 2.0], 1));

julia> predict(tree, [1.0, 2.0])
1

julia> nobs(tree)
1
```

# Reference
Manapragada, C., Webb, G. I., & Salehi, M. (2018). Extremely Fast Decision Tree. KDD.
"""
mutable struct ExtremelyFastTree{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    root::EFDTNode
    grace_period::Int
    re_eval_period::Int
    delta::T
    tau::T
    split_criterion::Symbol
    leaf_prediction::Symbol
    max_depth::Union{Int, Nothing}
    n::Int
    n_nodes::Int
    n_leaves::Int
    n_replacements::Int

    function ExtremelyFastTree{T}(;
        grace_period::Int = 200,
        re_eval_period::Int = 200,
        delta::Real = 1e-7,
        tau::Real = 0.05,
        split_criterion::Symbol = :info_gain,
        leaf_prediction::Symbol = :nba,
        max_depth::Union{Int, Nothing} = nothing
    ) where {T<:AbstractFloat}
        grace_period > 0 || throw(ArgumentError("grace_period must be positive"))
        0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1)"))
        tau >= 0 || throw(ArgumentError("tau must be non-negative"))
        split_criterion in (:gini, :info_gain, :variance) ||
            throw(ArgumentError("split_criterion must be :gini, :info_gain, or :variance"))
        leaf_prediction in (:mc, :nb, :nba) ||
            throw(ArgumentError("leaf_prediction must be :mc, :nb, or :nba"))

        new{T}(
            EFDTLeaf{T}(0),
            grace_period,
            re_eval_period,
            T(delta),
            T(tau),
            split_criterion,
            leaf_prediction,
            max_depth,
            0,
            1,  # Start with 1 node (root leaf)
            1,  # Start with 1 leaf
            0   # No replacements yet
        )
    end
end

ExtremelyFastTree(; kwargs...) = ExtremelyFastTree{Float64}(; kwargs...)

# OnlineStatsBase interface
function OnlineStatsBase._fit!(tree::ExtremelyFastTree{T}, xy::Tuple) where {T}
    x, y = xy
    x = collect(T, x)
    y = Int(y)

    # Find the leaf for this observation
    leaf = find_leaf(tree.root, x)

    # Update the leaf
    update!(leaf, x, y)

    # Update split nodes along the path (for re-evaluation)
    update_path!(tree.root, x, y)

    tree.n += 1

    # Check if we should attempt a split at the leaf
    if leaf.n > 0 && leaf.n % tree.grace_period == 0
        maybe_split!(tree, leaf, x)
    end

    # Check for re-evaluation at split nodes
    if tree.n % tree.re_eval_period == 0
        maybe_reevaluate!(tree)
    end

    return tree
end

OnlineStatsBase.value(tree::ExtremelyFastTree) = tree.root
OnlineStatsBase.nobs(tree::ExtremelyFastTree) = tree.n

function reset!(tree::ExtremelyFastTree{T}) where {T}
    tree.root = EFDTLeaf{T}(0)
    tree.n = 0
    tree.n_nodes = 1
    tree.n_leaves = 1
    tree.n_replacements = 0
    return tree
end

# Update statistics along the path (for split nodes that need re-evaluation)
function update_path!(node::EFDTLeaf, x, y)
    # Already updated
end

function update_path!(node::EFDTSplit{T}, x::AbstractVector, y::Int) where {T}
    update!(node, x, y)
    child = route(node, x)
    update_path!(child, x, y)
end

# Learner interface
function predict(tree::ExtremelyFastTree, x::AbstractVector)
    leaf = find_leaf(tree.root, x)
    return majority_class(leaf)
end

function predict_proba(tree::ExtremelyFastTree{T}, x::AbstractVector) where {T}
    leaf = find_leaf(tree.root, x)
    probs = class_probabilities(leaf)
    if isempty(probs)
        return Dict{Int, T}()
    end
    return probs
end

# Split evaluation for leaves
function maybe_split!(tree::ExtremelyFastTree{T}, leaf::EFDTLeaf{T}, x::AbstractVector) where {T}
    # Check depth constraint
    if tree.max_depth !== nothing && leaf.depth >= tree.max_depth
        return
    end

    # Need at least 2 classes to split
    n_classes = length(leaf.stats)
    if n_classes < 2
        return
    end

    # Compute split candidates and their merits
    best_split, best_merit = find_best_split_efdt(leaf, tree.split_criterion)
    second_best_merit = find_second_best_merit_efdt(leaf, tree.split_criterion, best_split)

    # Hoeffding bound
    R = log2(n_classes)
    ε = hoeffding_bound_efdt(R, tree.delta, leaf.n)

    # Split if best is significantly better than second best
    if best_merit - second_best_merit > ε || ε < tree.tau
        if best_split !== nothing
            perform_split_efdt!(tree, leaf, best_split)
        end
    end
end

# Re-evaluate splits at internal nodes
function maybe_reevaluate!(tree::ExtremelyFastTree{T}) where {T}
    reevaluate_recursive!(tree, tree.root)
end

function reevaluate_recursive!(tree::ExtremelyFastTree, node::EFDTLeaf)
    # Nothing to re-evaluate at leaves
end

function reevaluate_recursive!(tree::ExtremelyFastTree{T}, node::EFDTSplit{T}) where {T}
    # Check if this node should be re-evaluated
    if node.n > 0 && node.n - node.last_eval >= tree.re_eval_period
        node.last_eval = node.n

        # Find the best split for current statistics
        n_classes = length(node.stats)
        if n_classes >= 2
            best_split, best_merit = find_best_split_from_stats(node.stats, tree.split_criterion)

            # Current split merit
            current_merit = compute_current_split_merit(node, tree.split_criterion)

            # Hoeffding bound
            R = log2(n_classes)
            ε = hoeffding_bound_efdt(R, tree.delta, node.n)

            # Replace if new split is significantly better
            if best_split !== nothing && best_merit - current_merit > ε
                # Replace this subtree
                replace_subtree!(tree, node, best_split)
            end
        end
    end

    # Recurse to children
    reevaluate_recursive!(tree, node.left)
    reevaluate_recursive!(tree, node.right)
end

function hoeffding_bound_efdt(R::Float64, delta::Float64, n::Int)
    return sqrt(R^2 * log(1/delta) / (2 * n))
end

function hoeffding_bound_efdt(R::Float64, delta::T, n::Int) where {T<:AbstractFloat}
    return sqrt(R^2 * log(1/delta) / (2 * n))
end

# Find the best split for an EFDT leaf
function find_best_split_efdt(leaf::EFDTLeaf{T}, criterion::Symbol) where {T}
    best_split = nothing
    best_merit = -Inf

    n_attrs = maximum(length(ss.gaussians) for (_, ss) in leaf.stats; init=0)

    for attr in 1:n_attrs
        threshold = compute_threshold_efdt(leaf.stats, attr)
        if threshold === nothing
            continue
        end

        merit = compute_split_merit_efdt(leaf.stats, attr, threshold, criterion, T)

        if merit > best_merit
            best_merit = merit
            best_split = (attr, threshold)
        end
    end

    return best_split, best_merit
end

function find_best_split_from_stats(stats::Dict{Int, SufficientStats{T}}, criterion::Symbol) where {T}
    best_split = nothing
    best_merit = -Inf

    n_attrs = maximum(length(ss.gaussians) for (_, ss) in stats; init=0)

    for attr in 1:n_attrs
        threshold = compute_threshold_efdt(stats, attr)
        if threshold === nothing
            continue
        end

        merit = compute_split_merit_efdt(stats, attr, threshold, criterion, T)

        if merit > best_merit
            best_merit = merit
            best_split = (attr, threshold)
        end
    end

    return best_split, best_merit
end

function find_second_best_merit_efdt(leaf::EFDTLeaf{T}, criterion::Symbol, best_split) where {T}
    second_best = -Inf

    n_attrs = maximum(length(ss.gaussians) for (_, ss) in leaf.stats; init=0)

    for attr in 1:n_attrs
        if best_split !== nothing && attr == best_split[1]
            continue
        end

        threshold = compute_threshold_efdt(leaf.stats, attr)
        if threshold === nothing
            continue
        end

        merit = compute_split_merit_efdt(leaf.stats, attr, threshold, criterion, T)
        if merit > second_best
            second_best = merit
        end
    end

    return second_best
end

function compute_threshold_efdt(stats::Dict{Int, SufficientStats{T}}, attr::Int) where {T}
    total_weight = 0
    weighted_sum = zero(T)

    for (class, ss) in stats
        if attr <= length(ss.gaussians)
            v = ss.gaussians[attr]
            if nobs(v) > 0
                m = mean(v)
                w = ss.count
                weighted_sum += w * m
                total_weight += w
            end
        end
    end

    if total_weight == 0
        return nothing
    end

    return weighted_sum / total_weight
end

function compute_split_merit_efdt(stats::Dict{Int, SufficientStats{T}}, attr::Int, threshold, criterion::Symbol, ::Type{T}) where {T}
    if criterion == :info_gain
        return information_gain_efdt(stats, attr, threshold, T)
    elseif criterion == :gini
        return gini_gain_efdt(stats, attr, threshold, T)
    else
        return variance_reduction_efdt(stats, attr, threshold, T)
    end
end

function information_gain_efdt(stats::Dict{Int, SufficientStats{T}}, attr::Int, threshold, ::Type{T}) where {T}
    # Parent entropy
    total = sum(ss.count for (_, ss) in stats)
    if total == 0
        return zero(T)
    end

    parent_entropy = zero(T)
    for (_, ss) in stats
        p = ss.count / total
        if p > 0
            parent_entropy -= p * log2(p)
        end
    end

    # Estimate split based on Gaussian statistics
    left_counts = Dict{Int, Int}()
    right_counts = Dict{Int, Int}()

    for (class, ss) in stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            μ = mean(gauss)
            σ² = var(gauss)
            if σ² > 0
                z = (threshold - μ) / sqrt(σ²)
                p_left = one(T) / (one(T) + exp(-T(1.7) * z))
            else
                p_left = threshold >= μ ? one(T) : zero(T)
            end
            left_counts[class] = round(Int, p_left * ss.count)
            right_counts[class] = ss.count - left_counts[class]
        end
    end

    n_left = sum(values(left_counts); init=0)
    n_right = sum(values(right_counts); init=0)
    n_total = n_left + n_right

    if n_total == 0 || n_left == 0 || n_right == 0
        return zero(T)
    end

    # Child entropies
    left_entropy = zero(T)
    for c in values(left_counts)
        if c > 0
            p = T(c) / n_left
            left_entropy -= p * log2(p)
        end
    end

    right_entropy = zero(T)
    for c in values(right_counts)
        if c > 0
            p = T(c) / n_right
            right_entropy -= p * log2(p)
        end
    end

    weighted_child = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy

    return parent_entropy - weighted_child
end

function gini_gain_efdt(stats::Dict{Int, SufficientStats{T}}, attr::Int, threshold, ::Type{T}) where {T}
    total = sum(ss.count for (_, ss) in stats)
    if total == 0
        return zero(T)
    end

    # Parent Gini
    parent_gini = one(T)
    for (_, ss) in stats
        p = ss.count / total
        parent_gini -= p * p
    end

    # Estimate split
    left_counts = Dict{Int, Int}()
    right_counts = Dict{Int, Int}()

    for (class, ss) in stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            μ = mean(gauss)
            σ² = var(gauss)
            if σ² > 0
                z = (threshold - μ) / sqrt(σ²)
                p_left = one(T) / (one(T) + exp(-T(1.7) * z))
            else
                p_left = threshold >= μ ? one(T) : zero(T)
            end
            left_counts[class] = round(Int, p_left * ss.count)
            right_counts[class] = ss.count - left_counts[class]
        end
    end

    n_left = sum(values(left_counts); init=0)
    n_right = sum(values(right_counts); init=0)
    n_total = n_left + n_right

    if n_total == 0 || n_left == 0 || n_right == 0
        return zero(T)
    end

    # Child Gini
    left_gini = one(T)
    for c in values(left_counts)
        p = T(c) / n_left
        left_gini -= p * p
    end

    right_gini = one(T)
    for c in values(right_counts)
        p = T(c) / n_right
        right_gini -= p * p
    end

    weighted_child = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

    return parent_gini - weighted_child
end

function variance_reduction_efdt(stats::Dict{Int, SufficientStats{T}}, attr::Int, threshold, ::Type{T}) where {T}
    # Simplified variance reduction
    total_var = zero(T)
    total_n = 0

    for (_, ss) in stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            n = nobs(gauss)
            if n > 0
                total_var += var(gauss) * n
                total_n += n
            end
        end
    end

    if total_n == 0
        return zero(T)
    end

    return total_var / total_n  # Higher variance = more room for reduction
end

function compute_current_split_merit(node::EFDTSplit{T}, criterion::Symbol) where {T}
    # Compute merit of current split
    return compute_split_merit_efdt(node.stats, node.attribute, node.threshold, criterion, T)
end

# Perform a split at a leaf
function perform_split_efdt!(tree::ExtremelyFastTree{T}, leaf::EFDTLeaf{T}, split::Tuple{Int, Any}) where {T}
    attr, threshold = split

    # Create new leaf nodes
    left_leaf = EFDTLeaf{T}(leaf.depth + 1)
    right_leaf = EFDTLeaf{T}(leaf.depth + 1)

    # Create split node
    split_node = EFDTSplit{T}(attr, T(threshold), left_leaf, right_leaf, leaf.depth)

    # Replace leaf with split node
    replace_efdt_node!(tree, leaf, split_node)

    tree.n_nodes += 2
    tree.n_leaves += 1
end

function replace_subtree!(tree::ExtremelyFastTree{T}, old_split::EFDTSplit{T}, new_split::Tuple{Int, Any}) where {T}
    attr, threshold = new_split

    # Create new leaf nodes
    left_leaf = EFDTLeaf{T}(old_split.depth + 1)
    right_leaf = EFDTLeaf{T}(old_split.depth + 1)

    # Replace children with fresh leaves
    old_split.attribute = attr
    old_split.threshold = T(threshold)
    old_split.left = left_leaf
    old_split.right = right_leaf

    # Reset statistics
    empty!(old_split.stats)
    old_split.class_counts = CountMap{Int}()
    old_split.n = 0
    old_split.last_eval = 0

    tree.n_replacements += 1
end

function replace_efdt_node!(tree::ExtremelyFastTree, old_leaf::EFDTLeaf, new_node::EFDTSplit)
    if tree.root === old_leaf
        tree.root = new_node
    else
        replace_in_efdt_subtree!(tree.root, old_leaf, new_node)
    end
end

function replace_in_efdt_subtree!(node::EFDTLeaf, old_leaf, new_node)
    # Can't replace in leaf
end

function replace_in_efdt_subtree!(node::EFDTSplit, old_leaf, new_node)
    if node.left === old_leaf
        node.left = new_node
    elseif node.right === old_leaf
        node.right = new_node
    else
        replace_in_efdt_subtree!(node.left, old_leaf, new_node)
        replace_in_efdt_subtree!(node.right, old_leaf, new_node)
    end
end

# Inspection methods
"""
    n_nodes(tree::ExtremelyFastTree) -> Int

Return the total number of nodes in the tree.
"""
n_nodes(tree::ExtremelyFastTree) = tree.n_nodes

"""
    n_leaves(tree::ExtremelyFastTree) -> Int

Return the number of leaf nodes in the tree.
"""
n_leaves(tree::ExtremelyFastTree) = tree.n_leaves

"""
    n_replacements(tree::ExtremelyFastTree) -> Int

Return the number of subtree replacements due to re-evaluation.
"""
n_replacements(tree::ExtremelyFastTree) = tree.n_replacements

"""
    height(tree::ExtremelyFastTree) -> Int

Return the maximum depth of the tree.
"""
function height(tree::ExtremelyFastTree)
    return compute_height_efdt(tree.root)
end

function compute_height_efdt(node::EFDTLeaf)
    return node.depth
end

function compute_height_efdt(node::EFDTSplit)
    return max(compute_height_efdt(node.left), compute_height_efdt(node.right))
end

# Show method
function Base.show(io::IO, tree::ExtremelyFastTree)
    print(io, "ExtremelyFastTree: n=$(nobs(tree)) | nodes=$(n_nodes(tree)) | leaves=$(n_leaves(tree)) | replacements=$(n_replacements(tree))")
end

export ExtremelyFastTree, n_replacements
