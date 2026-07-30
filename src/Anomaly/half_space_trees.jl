# Half-Space Trees for streaming anomaly detection

# Internal tree node - must be defined first for forward reference
mutable struct HSNode
    split_dim::Int
    split_val::Float64
    left::Union{HSNode, Nothing}
    right::Union{HSNode, Nothing}
    r::Int  # Reference mass
    s::Int  # Latest window mass
    depth::Int
end

# Internal tree structure
struct HSTree
    root::HSNode
    min_vals::Vector{Float64}
    max_vals::Vector{Float64}
end

"""
    HalfSpaceTrees <: UnsupervisedLearner{AbstractVector}

Online anomaly detection using Half-Space Trees (HS-Trees).

# Parameters
- `n_trees::Int = 25` - Number of trees in ensemble
- `height::Int = 8` - Maximum tree depth
- `window_size::Int = 250` - Reference window size

# Example
```jldoctest
julia> model = HalfSpaceTrees(n_trees=25, height=8)
HalfSpaceTrees: n=0 | value=0

julia> fit!(model, [1.0, 2.0]);

julia> nobs(model)
1
```
"""
mutable struct HalfSpaceTrees <: UnsupervisedLearner{AbstractVector}
    n_trees::Int
    height::Int
    window_size::Int
    trees::Vector{HSTree}
    n::Int
    rng::Random.AbstractRNG

    function HalfSpaceTrees(;
        n_trees::Int = 25,
        height::Int = 8,
        window_size::Int = 250,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        n_trees > 0 || throw(ArgumentError("n_trees must be positive"))
        height >= 0 || throw(ArgumentError("height must be non-negative"))
        window_size > 0 || throw(ArgumentError("window_size must be positive"))
        new(n_trees, height, window_size, HSTree[], 0, rng)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(hst::HalfSpaceTrees, x)
    x = collect(Float64, x)

    # Initialize trees on first observation
    if isempty(hst.trees)
        init_trees!(hst, x)
    else
        expected = length(first(hst.trees).min_vals)
        length(x) == expected ||
            throw(DimensionMismatch("expected $expected features, got $(length(x))"))
    end

    # Update each tree
    for tree in hst.trees
        update_tree!(tree, x, hst.window_size, hst.n)
    end

    hst.n += 1
    return hst
end

OnlineStatsBase.value(hst::HalfSpaceTrees) = hst.n
OnlineStatsBase.nobs(hst::HalfSpaceTrees) = hst.n

function reset!(hst::HalfSpaceTrees)
    empty!(hst.trees)
    hst.n = 0
    return hst
end

# Initialize trees with random splits
function init_trees!(hst::HalfSpaceTrees, x::Vector{Float64})
    d = length(x)
    for _ in 1:hst.n_trees
        # Random bounding box
        min_vals = rand(hst.rng, d) .* 2 .- 1
        max_vals = min_vals .+ rand(hst.rng, d) .* 2

        root = build_tree(hst.rng, min_vals, max_vals, 0, hst.height)
        push!(hst.trees, HSTree(root, min_vals, max_vals))
    end
end

# Build random tree recursively
function build_tree(rng, min_vals, max_vals, depth, max_depth)
    node = HSNode(0, 0.0, nothing, nothing, 0, 0, depth)

    if depth < max_depth
        d = length(min_vals)
        node.split_dim = rand(rng, 1:d)
        node.split_val = min_vals[node.split_dim] +
            rand(rng) * (max_vals[node.split_dim] - min_vals[node.split_dim])

        left_max = copy(max_vals)
        left_max[node.split_dim] = node.split_val

        right_min = copy(min_vals)
        right_min[node.split_dim] = node.split_val

        node.left = build_tree(rng, min_vals, left_max, depth + 1, max_depth)
        node.right = build_tree(rng, right_min, max_vals, depth + 1, max_depth)
    end

    return node
end

# Update tree with new point
function update_tree!(tree::HSTree, x::Vector{Float64}, window_size::Int, n::Int)
    # Reset reference window periodically
    if n > 0 && n % window_size == 0
        reset_masses!(tree.root)
    end

    update_node!(tree.root, x)
end

function update_node!(node::HSNode, x::Vector{Float64})
    node.s += 1

    if node.left !== nothing && node.right !== nothing
        if x[node.split_dim] < node.split_val
            update_node!(node.left, x)
        else
            update_node!(node.right, x)
        end
    end
end

function reset_masses!(node::HSNode)
    node.r = node.s
    node.s = 0

    if node.left !== nothing
        reset_masses!(node.left)
    end
    if node.right !== nothing
        reset_masses!(node.right)
    end
end

# Compute anomaly score
function score(hst::HalfSpaceTrees, x::AbstractVector)
    if isempty(hst.trees)
        return 0.0
    end

    x = collect(Float64, x)
    expected = length(first(hst.trees).min_vals)
    length(x) == expected ||
        throw(DimensionMismatch("expected $expected features, got $(length(x))"))
    total_score = 0.0

    for tree in hst.trees
        total_score += score_tree(tree, x)
    end

    return total_score / length(hst.trees)
end

function score_tree(tree::HSTree, x::Vector{Float64})
    return score_node(tree.root, x)
end

function score_node(node::HSNode, x::Vector{Float64})
    if node.left === nothing || node.right === nothing
        # Leaf node: score based on mass ratio
        return node.r > 0 ? Float64(node.r) * 2.0^node.depth : 0.0
    end

    if x[node.split_dim] < node.split_val
        return score_node(node.left, x)
    else
        return score_node(node.right, x)
    end
end

"""
    score_one(hst::HalfSpaceTrees, x) -> Float64

Return anomaly score normalized to [0, 1] range.
Higher scores indicate more anomalous observations.
"""
function score_one(hst::HalfSpaceTrees, x::AbstractVector)
    isempty(hst.trees) && return 0.5
    raw_score = score(hst, x)
    # Use sigmoid to normalize
    # Lower raw scores = more anomalous in HS-Trees
    # Invert so higher = more anomalous
    max_score = hst.window_size * 2.0^hst.height
    if max_score > 0
        normalized = 1.0 - raw_score / max_score
    else
        normalized = 0.5
    end
    return clamp(normalized, 0.0, 1.0)
end
