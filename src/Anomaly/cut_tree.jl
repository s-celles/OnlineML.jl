# Random Cut Tree structures for RobustRandomCutForest
# Implements the RCT data structure with efficient insertion/deletion

using Random
using DataStructures: CircularBuffer

"""
    CutNode

Abstract type for nodes in a Random Cut Tree.
"""
abstract type CutNode end

"""
    CutLeaf <: CutNode

Leaf node containing a single point index.
"""
mutable struct CutLeaf <: CutNode
    point_idx::Int
    depth::Int
end

"""
    CutBranch <: CutNode

Internal node with a random axis-aligned cut.
"""
mutable struct CutBranch <: CutNode
    cut_dim::Int
    cut_val::Float64
    left::CutNode
    right::CutNode
    n::Int  # Number of points in subtree

    function CutBranch(cut_dim::Int, cut_val::Float64, left::CutNode, right::CutNode)
        n = count_points(left) + count_points(right)
        new(cut_dim, cut_val, left, right, n)
    end
end

"""
Count points in a subtree.
"""
count_points(::CutLeaf) = 1
count_points(branch::CutBranch) = branch.n

"""
    CutTree

A single Random Cut Tree for RRCF.
"""
mutable struct CutTree
    root::Union{CutNode, Nothing}
    n_points::Int
    bbox_min::Vector{Float64}
    bbox_max::Vector{Float64}
    rng::Random.AbstractRNG

    function CutTree(; rng::Random.AbstractRNG=Random.default_rng())
        new(nothing, 0, Float64[], Float64[], rng)
    end
end

"""
Insert a point into the cut tree.
Returns the leaf node created.
"""
function insert_point!(tree::CutTree, point::Vector{Float64}, point_idx::Int)
    if tree.root === nothing
        # First point - create a leaf
        tree.root = CutLeaf(point_idx, 0)
        tree.n_points = 1
        tree.bbox_min = copy(point)
        tree.bbox_max = copy(point)
        return tree.root
    end

    # Update bounding box
    for i in eachindex(point)
        if i <= length(tree.bbox_min)
            tree.bbox_min[i] = min(tree.bbox_min[i], point[i])
            tree.bbox_max[i] = max(tree.bbox_max[i], point[i])
        else
            push!(tree.bbox_min, point[i])
            push!(tree.bbox_max, point[i])
        end
    end

    # Insert point recursively
    tree.root = insert_recursive!(tree, tree.root, point, point_idx, tree.bbox_min, tree.bbox_max, 0)
    tree.n_points += 1

    return tree.root
end

"""
Recursive insertion into the tree.
"""
function insert_recursive!(tree::CutTree, node::CutLeaf, point::Vector{Float64}, point_idx::Int,
                           bbox_min::Vector{Float64}, bbox_max::Vector{Float64}, depth::Int)
    # Generate a random cut
    cut_dim, cut_val = random_cut(tree.rng, bbox_min, bbox_max)

    # Create new leaf for the new point
    new_leaf = CutLeaf(point_idx, depth + 1)

    # Determine which side each point goes
    # We need to look up the existing point (not stored in leaf, so use a placeholder)
    # In a full implementation, we'd store point data or reference a point buffer
    # For now, use the cut value to determine placement

    # Create branch with the existing leaf and new leaf
    if rand(tree.rng) < 0.5
        branch = CutBranch(cut_dim, cut_val, node, new_leaf)
    else
        branch = CutBranch(cut_dim, cut_val, new_leaf, node)
    end

    return branch
end

function insert_recursive!(tree::CutTree, node::CutBranch, point::Vector{Float64}, point_idx::Int,
                           bbox_min::Vector{Float64}, bbox_max::Vector{Float64}, depth::Int)
    # Choose a dimension and value for potential new cut
    range_sum = sum(bbox_max .- bbox_min)

    if range_sum <= 0
        # Degenerate case - all dimensions have same min/max
        # Just insert to the left
        node.left = insert_recursive!(tree, node.left, point, point_idx, bbox_min, bbox_max, depth + 1)
        node.n += 1
        return node
    end

    # With some probability, cut above this node
    cut_dim, cut_val = random_cut(tree.rng, bbox_min, bbox_max)

    # Probability of cutting above current node
    p_cut_above = 0.0
    if node.cut_dim <= length(bbox_max)
        node_range = bbox_max[node.cut_dim] - bbox_min[node.cut_dim]
        if node_range > 0
            p_cut_above = (max(point[node.cut_dim], bbox_max[node.cut_dim]) -
                          min(point[node.cut_dim], bbox_min[node.cut_dim]) -
                          node_range) / range_sum
        end
    end

    if rand(tree.rng) < clamp(p_cut_above, 0.0, 0.5)
        # Insert above current node
        new_leaf = CutLeaf(point_idx, depth)
        if point[cut_dim] <= cut_val
            return CutBranch(cut_dim, cut_val, new_leaf, node)
        else
            return CutBranch(cut_dim, cut_val, node, new_leaf)
        end
    else
        # Insert below - route to appropriate child
        if node.cut_dim <= length(point) && point[node.cut_dim] <= node.cut_val
            node.left = insert_recursive!(tree, node.left, point, point_idx, bbox_min, bbox_max, depth + 1)
        else
            node.right = insert_recursive!(tree, node.right, point, point_idx, bbox_min, bbox_max, depth + 1)
        end
        node.n += 1
        return node
    end
end

"""
Generate a random cut dimension and value.
"""
function random_cut(rng::Random.AbstractRNG, bbox_min::Vector{Float64}, bbox_max::Vector{Float64})
    ranges = bbox_max .- bbox_min
    total_range = sum(ranges)

    if total_range <= 0
        # Degenerate case
        return 1, (bbox_min[1] + bbox_max[1]) / 2
    end

    # Choose dimension proportional to range
    r = rand(rng) * total_range
    cumsum = 0.0
    cut_dim = 1

    for (i, range) in enumerate(ranges)
        cumsum += range
        if r <= cumsum
            cut_dim = i
            break
        end
    end

    # Choose value uniformly in range
    cut_val = bbox_min[cut_dim] + rand(rng) * (bbox_max[cut_dim] - bbox_min[cut_dim])

    return cut_dim, cut_val
end

"""
Delete a point from the tree by its index.
"""
function delete_point!(tree::CutTree, point_idx::Int)
    if tree.root === nothing
        return false
    end

    tree.root, found = delete_recursive!(tree.root, point_idx)
    if found
        tree.n_points -= 1
    end
    return found
end

function delete_recursive!(node::CutLeaf, point_idx::Int)
    if node.point_idx == point_idx
        return nothing, true
    end
    return node, false
end

function delete_recursive!(node::CutBranch, point_idx::Int)
    # Try left subtree
    new_left, found = delete_recursive!(node.left, point_idx)
    if found
        if new_left === nothing
            # Left subtree is now empty, return right
            return node.right, true
        else
            node.left = new_left
            node.n -= 1
            return node, true
        end
    end

    # Try right subtree
    new_right, found = delete_recursive!(node.right, point_idx)
    if found
        if new_right === nothing
            # Right subtree is now empty, return left
            return node.left, true
        else
            node.right = new_right
            node.n -= 1
            return node, true
        end
    end

    return node, false
end

"""
Compute the CoDisplacement score for a point.
This is the expected change in model complexity if the point is removed.
"""
function codisplacement(tree::CutTree, point_idx::Int)
    if tree.root === nothing
        return 0.0
    end
    return codisplacement_recursive(tree.root, point_idx, 0)
end

function codisplacement_recursive(node::CutLeaf, point_idx::Int, depth::Int)
    if node.point_idx == point_idx
        # Found the point - return its depth as base score
        return Float64(depth)
    end
    return 0.0
end

function codisplacement_recursive(node::CutBranch, point_idx::Int, depth::Int)
    # Check left subtree
    left_score = codisplacement_recursive(node.left, point_idx, depth + 1)
    if left_score > 0
        # Point is in left subtree
        # CoDisplacement = depth contribution + sibling subtree size
        return left_score + count_points(node.right)
    end

    # Check right subtree
    right_score = codisplacement_recursive(node.right, point_idx, depth + 1)
    if right_score > 0
        return right_score + count_points(node.left)
    end

    return 0.0
end

"""
Reset the cut tree to empty state.
"""
function reset!(tree::CutTree)
    tree.root = nothing
    tree.n_points = 0
    empty!(tree.bbox_min)
    empty!(tree.bbox_max)
    return tree
end

export CutNode, CutLeaf, CutBranch, CutTree
export insert_point!, delete_point!, codisplacement, count_points
