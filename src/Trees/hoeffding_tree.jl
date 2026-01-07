# Hoeffding Tree (VFDT) implementation

"""
    HoeffdingTree{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Streaming decision tree classifier using the VFDT algorithm.

# Parameters
- `grace_period::Int = 200` - Observations between split evaluations
- `delta::Float64 = 1e-7` - Hoeffding bound confidence parameter
- `tau::Float64 = 0.05` - Tie-breaking threshold
- `split_criterion::Symbol = :info_gain` - `:gini`, `:info_gain`, or `:variance`
- `leaf_prediction::Symbol = :nba` - `:mc` (majority), `:nb` (naive bayes), `:nba` (adaptive)
- `max_depth::Union{Int,Nothing} = nothing` - Maximum tree depth (unlimited if nothing)

# Example
```jldoctest
julia> tree = HoeffdingTree(grace_period=100, delta=1e-5)
HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0)

julia> fit!(tree, ([1.0, 2.0], 1));

julia> predict(tree, [1.0, 2.0])
1

julia> nobs(tree)
1
```

# Reference
Domingos, P. & Hulten, G. (2000). Mining High-Speed Data Streams. KDD.
"""
mutable struct HoeffdingTree{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    root::TreeNode
    grace_period::Int
    delta::T
    tau::T
    split_criterion::Symbol
    leaf_prediction::Symbol
    max_depth::Union{Int, Nothing}
    n::Int
    n_nodes::Int
    n_leaves::Int

    function HoeffdingTree{T}(;
        grace_period::Int = 200,
        delta::Real = 1e-7,
        tau::Real = 0.05,
        split_criterion::Symbol = :info_gain,
        leaf_prediction::Symbol = :nba,
        max_depth::Union{Int, Nothing} = nothing
    ) where {T<:AbstractFloat}
        @assert split_criterion in (:gini, :info_gain, :variance)
        @assert leaf_prediction in (:mc, :nb, :nba)
        new{T}(
            LeafNode{T}(0),
            grace_period,
            T(delta),
            T(tau),
            split_criterion,
            leaf_prediction,
            max_depth,
            0,
            1,  # Start with 1 node (root leaf)
            1   # Start with 1 leaf
        )
    end
end

HoeffdingTree(; kwargs...) = HoeffdingTree{Float64}(; kwargs...)

# OnlineStatsBase interface
function OnlineStatsBase._fit!(tree::HoeffdingTree{T}, xy::Tuple) where {T}
    x, y = xy
    x = collect(T, x)
    y = Int(y)

    # Find the leaf for this observation
    leaf = find_leaf(tree.root, x)

    # Update the leaf
    update!(leaf, x, y)

    tree.n += 1

    # Check if we should attempt a split
    if leaf.n > 0 && leaf.n % tree.grace_period == 0
        maybe_split!(tree, leaf, x)
    end

    return tree
end

OnlineStatsBase.value(tree::HoeffdingTree) = tree.root
OnlineStatsBase.nobs(tree::HoeffdingTree) = tree.n

function reset!(tree::HoeffdingTree{T}) where {T}
    tree.root = LeafNode{T}(0)
    tree.n = 0
    tree.n_nodes = 1
    tree.n_leaves = 1
    return tree
end

# Find the leaf node for an observation
function find_leaf(node::LeafNode, x)
    return node
end

function find_leaf(node::SplitNode, x)
    child = route(node, x)
    return find_leaf(child, x)
end

# Learner interface
function predict(tree::HoeffdingTree, x::AbstractVector)
    leaf = find_leaf(tree.root, x)
    return majority_class(leaf)
end

function predict_proba(tree::HoeffdingTree{T}, x::AbstractVector) where {T}
    leaf = find_leaf(tree.root, x)
    probs = class_probabilities(leaf)
    if isempty(probs)
        return Dict{Int, T}()
    end
    return probs
end

# Split evaluation
function maybe_split!(tree::HoeffdingTree{T}, leaf::LeafNode{T}, x::AbstractVector) where {T}
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
    best_split, best_merit = find_best_split(leaf, tree.split_criterion)
    second_best_merit = find_second_best_merit(leaf, tree.split_criterion, best_split)

    # Hoeffding bound
    R = log2(n_classes)  # Range of merit
    ε = hoeffding_bound(R, tree.delta, leaf.n)

    # Split if best is significantly better than second best
    if best_merit - second_best_merit > ε || ε < tree.tau
        if best_split !== nothing
            perform_split!(tree, leaf, best_split)
        end
    end
end

# Hoeffding bound
function hoeffding_bound(R::Float64, delta::Float64, n::Int)
    return sqrt(R^2 * log(1/delta) / (2 * n))
end

# Find the best split for a leaf
function find_best_split(leaf::LeafNode{T}, criterion::Symbol) where {T}
    best_split = nothing
    best_merit = -Inf

    # Try each attribute
    n_attrs = maximum(length(ss.gaussians) for (_, ss) in leaf.stats; init=0)

    for attr in 1:n_attrs
        # Get threshold candidates (use mean of attribute for simplicity)
        threshold = compute_threshold(leaf, attr)
        if threshold === nothing
            continue
        end

        # Compute merit for this split
        merit = compute_split_merit(leaf, attr, threshold, criterion)

        if merit > best_merit
            best_merit = merit
            best_split = (attr, threshold)
        end
    end

    return best_split, best_merit
end

function find_second_best_merit(leaf::LeafNode{T}, criterion::Symbol, best_split) where {T}
    second_best = -Inf

    n_attrs = maximum(length(ss.gaussians) for (_, ss) in leaf.stats; init=0)

    for attr in 1:n_attrs
        threshold = compute_threshold(leaf, attr)
        if threshold === nothing
            continue
        end

        if best_split !== nothing && attr == best_split[1]
            continue
        end

        merit = compute_split_merit(leaf, attr, threshold, criterion)
        if merit > second_best
            second_best = merit
        end
    end

    return second_best
end

# Compute split threshold (use weighted mean)
function compute_threshold(leaf::LeafNode{T}, attr::Int) where {T}
    total_weight = 0
    weighted_sum = zero(T)

    for (class, ss) in leaf.stats
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

# Compute split merit using information gain
function compute_split_merit(leaf::LeafNode{T}, attr::Int, threshold::T, criterion::Symbol) where {T}
    if criterion == :info_gain
        return information_gain(leaf, attr, threshold)
    elseif criterion == :gini
        return gini_gain(leaf, attr, threshold)
    else
        return variance_reduction(leaf, attr, threshold)
    end
end

# Information gain calculation
function information_gain(leaf::LeafNode{T}, attr::Int, threshold::T) where {T}
    parent_entropy = compute_entropy(leaf)
    if parent_entropy == zero(T) || leaf.n == 0
        return zero(T)
    end

    # Estimate split based on Gaussian statistics per class
    left_counts = Dict{Int, Int}()
    right_counts = Dict{Int, Int}()

    for (class, ss) in leaf.stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            μ = mean(gauss)
            # Estimate proportion below threshold using Gaussian CDF approximation
            # For standard normal: Φ(z) ≈ 0.5 for z=0
            if var(gauss) > 0
                z = (threshold - μ) / sqrt(var(gauss))
                # Approximate CDF using logistic function
                p_left = one(T) / (one(T) + exp(-T(1.7) * z))
            else
                p_left = threshold >= μ ? one(T) : zero(T)
            end
            class_count = ss.count
            left_counts[class] = round(Int, p_left * class_count)
            right_counts[class] = class_count - left_counts[class]
        end
    end

    n_left = sum(values(left_counts); init=0)
    n_right = sum(values(right_counts); init=0)
    n_total = n_left + n_right

    if n_total == 0 || n_left == 0 || n_right == 0
        return zero(T)
    end

    # Compute child entropies
    left_entropy = compute_entropy_from_counts(left_counts, T)
    right_entropy = compute_entropy_from_counts(right_counts, T)

    # Weighted average of child entropies
    weighted_child_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy

    return parent_entropy - weighted_child_entropy
end

function gini_gain(leaf::LeafNode{T}, attr::Int, threshold::T) where {T}
    parent_gini = compute_gini(leaf)
    if leaf.n == 0
        return zero(T)
    end

    # Estimate split based on Gaussian statistics per class
    left_counts = Dict{Int, Int}()
    right_counts = Dict{Int, Int}()

    for (class, ss) in leaf.stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            μ = mean(gauss)
            if var(gauss) > 0
                z = (threshold - μ) / sqrt(var(gauss))
                p_left = one(T) / (one(T) + exp(-T(1.7) * z))
            else
                p_left = threshold >= μ ? one(T) : zero(T)
            end
            class_count = ss.count
            left_counts[class] = round(Int, p_left * class_count)
            right_counts[class] = class_count - left_counts[class]
        end
    end

    n_left = sum(values(left_counts); init=0)
    n_right = sum(values(right_counts); init=0)
    n_total = n_left + n_right

    if n_total == 0 || n_left == 0 || n_right == 0
        return zero(T)
    end

    # Compute child Gini impurities
    left_gini = compute_gini_from_counts(left_counts, T)
    right_gini = compute_gini_from_counts(right_counts, T)

    # Weighted average
    weighted_child_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

    return parent_gini - weighted_child_gini
end

function variance_reduction(leaf::LeafNode{T}, attr::Int, threshold::T) where {T}
    if leaf.n == 0
        return zero(T)
    end

    # Total variance across all observations
    parent_var = zero(T)
    total_n = 0

    for (_, ss) in leaf.stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            n = nobs(gauss)
            if n > 0
                parent_var += var(gauss) * n
                total_n += n
            end
        end
    end

    if total_n == 0
        return zero(T)
    end
    parent_var /= total_n

    # Estimate child variances using Gaussian split
    left_var = zero(T)
    right_var = zero(T)
    n_left = 0
    n_right = 0

    for (_, ss) in leaf.stats
        if attr <= length(ss.gaussians)
            gauss = ss.gaussians[attr]
            μ = mean(gauss)
            σ² = var(gauss)
            n = nobs(gauss)

            if n > 0 && σ² > 0
                z = (threshold - μ) / sqrt(σ²)
                p_left = one(T) / (one(T) + exp(-T(1.7) * z))
                nl = round(Int, p_left * n)
                nr = n - nl

                if nl > 0
                    # Truncated variance approximation
                    left_var += σ² * T(0.5) * nl
                    n_left += nl
                end
                if nr > 0
                    right_var += σ² * T(0.5) * nr
                    n_right += nr
                end
            end
        end
    end

    if n_left > 0
        left_var /= n_left
    end
    if n_right > 0
        right_var /= n_right
    end

    total_child = n_left + n_right
    if total_child == 0
        return zero(T)
    end

    weighted_child_var = (n_left / total_child) * left_var + (n_right / total_child) * right_var

    return parent_var - weighted_child_var
end

# Helper: compute entropy from class counts
function compute_entropy_from_counts(counts::Dict{Int, Int}, ::Type{T}) where {T}
    total = sum(values(counts))
    if total == 0
        return zero(T)
    end
    entropy = zero(T)
    for c in values(counts)
        if c > 0
            p = T(c) / total
            entropy -= p * log2(p)
        end
    end
    return entropy
end

# Helper: compute Gini impurity from class counts
function compute_gini_from_counts(counts::Dict{Int, Int}, ::Type{T}) where {T}
    total = sum(values(counts))
    if total == 0
        return zero(T)
    end
    gini = one(T)
    for c in values(counts)
        p = T(c) / total
        gini -= p * p
    end
    return gini
end

# Compute Gini impurity for a leaf
function compute_gini(leaf::LeafNode{T}) where {T}
    if leaf.n == 0
        return zero(T)
    end
    probs = class_probabilities(leaf)
    if isempty(probs)
        return zero(T)
    end
    return one(T) - sum(p * p for p in values(probs))
end

function compute_entropy(leaf::LeafNode{T}) where {T}
    probs = class_probabilities(leaf)
    if isempty(probs)
        return zero(T)
    end
    return -sum(p > 0 ? p * log2(p) : zero(T) for p in values(probs))
end

# Perform the split
function perform_split!(tree::HoeffdingTree{T}, leaf::LeafNode{T}, split::Tuple{Int, T}) where {T}
    attr, threshold = split

    # Create new leaf nodes
    left_leaf = LeafNode{T}(leaf.depth + 1)
    right_leaf = LeafNode{T}(leaf.depth + 1)

    # Create split node
    split_node = SplitNode{T}(attr, threshold, left_leaf, right_leaf)

    # Replace leaf with split node in tree
    replace_node!(tree, leaf, split_node)

    # Update counts
    tree.n_nodes += 2  # Added split node and one leaf (other replaces existing)
    tree.n_leaves += 1  # Net gain of 1 leaf
end

# Replace a leaf node with a split node
function replace_node!(tree::HoeffdingTree, old_leaf::LeafNode, new_node::SplitNode)
    if tree.root === old_leaf
        tree.root = new_node
    else
        replace_in_subtree!(tree.root, old_leaf, new_node)
    end
end

function replace_in_subtree!(node::LeafNode, old_leaf, new_node)
    # Can't replace in leaf
end

function replace_in_subtree!(node::SplitNode, old_leaf, new_node)
    if node.left === old_leaf
        node.left = new_node
    elseif node.right === old_leaf
        node.right = new_node
    else
        replace_in_subtree!(node.left, old_leaf, new_node)
        replace_in_subtree!(node.right, old_leaf, new_node)
    end
end

# Inspection methods
"""
    n_nodes(tree::HoeffdingTree) -> Int

Return the total number of nodes in the tree.
"""
n_nodes(tree::HoeffdingTree) = tree.n_nodes

"""
    n_leaves(tree::HoeffdingTree) -> Int

Return the number of leaf nodes in the tree.
"""
n_leaves(tree::HoeffdingTree) = tree.n_leaves

"""
    height(tree::HoeffdingTree) -> Int

Return the maximum depth of the tree.
"""
function height(tree::HoeffdingTree)
    return compute_height(tree.root)
end

function compute_height(node::LeafNode)
    return node.depth
end

function compute_height(node::SplitNode)
    return max(compute_height(node.left), compute_height(node.right))
end
