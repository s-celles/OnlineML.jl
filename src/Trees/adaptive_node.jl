# Adaptive Node wrapper for HoeffdingAdaptiveTree
# Wraps tree nodes with ADWIN drift detection and alternate subtree management

using ..Drift: ADWIN, status, detected_drift, detected_warning, DriftStatus, NoDrift, Warning, DriftDetected

"""
    AdaptiveNode{T<:AbstractFloat}

Node wrapper that adds drift detection and alternate subtree management
for use in HoeffdingAdaptiveTree.

Each adaptive node:
- Wraps an underlying tree node (leaf or split)
- Contains an ADWIN detector monitoring classification errors
- Can grow an alternate subtree during warning phase
- Replaces its subtree when drift is confirmed and alternate performs better
"""
mutable struct AdaptiveNode{T<:AbstractFloat}
    # Underlying node (can be leaf or split)
    node::Union{LeafNode{T}, SplitNode{T}}
    # ADWIN detector for error monitoring
    adwin::ADWIN
    # Alternate subtree (grown during warning)
    alternate::Union{AdaptiveNode{T}, Nothing}
    # Statistics for drift handling
    error_count::Int
    sample_count::Int
    # Children (if this wraps a split node)
    left::Union{AdaptiveNode{T}, Nothing}
    right::Union{AdaptiveNode{T}, Nothing}

    function AdaptiveNode{T}(node::Union{LeafNode{T}, SplitNode{T}}; adwin_delta::Float64=0.002) where {T}
        new{T}(node, ADWIN(delta=adwin_delta), nothing, 0, 0, nothing, nothing)
    end
end

AdaptiveNode(node::Union{LeafNode{Float64}, SplitNode{Float64}}; adwin_delta::Float64=0.002) =
    AdaptiveNode{Float64}(node; adwin_delta=adwin_delta)

"""
Create a new adaptive leaf node.
"""
function adaptive_leaf(::Type{T}, depth::Int=0; adwin_delta::Float64=0.002) where {T<:AbstractFloat}
    AdaptiveNode{T}(LeafNode{T}(depth); adwin_delta=adwin_delta)
end

adaptive_leaf(depth::Int=0; adwin_delta::Float64=0.002) = adaptive_leaf(Float64, depth; adwin_delta=adwin_delta)

"""
Check if this adaptive node is a leaf.
"""
is_leaf(anode::AdaptiveNode) = anode.node isa LeafNode

"""
Check if this adaptive node is a split.
"""
is_split(anode::AdaptiveNode) = anode.node isa SplitNode

"""
Get the depth of the underlying node.
"""
function node_depth(anode::AdaptiveNode)
    if anode.node isa LeafNode
        return anode.node.depth
    else
        return 0  # Split nodes don't track depth directly
    end
end

"""
Update the adaptive node with a classification result.
Returns the current drift status.
"""
function update_error!(anode::AdaptiveNode, is_error::Bool)
    # Update ADWIN with error indicator
    error_val = is_error ? 1.0 : 0.0
    fit!(anode.adwin, error_val)

    if is_error
        anode.error_count += 1
    end
    anode.sample_count += 1

    return status(anode.adwin)
end

"""
Get the error rate for this adaptive node.
"""
function error_rate(anode::AdaptiveNode)
    if anode.sample_count == 0
        return 0.0
    end
    return anode.error_count / anode.sample_count
end

"""
Reset the adaptive node's drift detector and statistics.
"""
function reset_drift!(anode::AdaptiveNode)
    reset!(anode.adwin)
    anode.error_count = 0
    anode.sample_count = 0
    anode.alternate = nothing
    return anode
end

"""
Route an observation through the adaptive node.
"""
function route(anode::AdaptiveNode{T}, x::AbstractVector) where {T}
    if is_split(anode)
        node = anode.node::SplitNode{T}
        if x[node.attribute] <= node.threshold
            return anode.left
        else
            return anode.right
        end
    else
        return nothing  # Can't route from a leaf
    end
end

"""
Find the adaptive leaf for an observation.
"""
function find_adaptive_leaf(anode::AdaptiveNode, x)
    if is_leaf(anode)
        return anode
    else
        child = route(anode, x)
        if child === nothing
            return anode
        end
        return find_adaptive_leaf(child, x)
    end
end

"""
Check if we should start growing an alternate subtree.
"""
should_grow_alternate(anode::AdaptiveNode) =
    detected_warning(anode.adwin) && anode.alternate === nothing

"""
Check if we should evaluate replacing the subtree.
"""
should_evaluate_replacement(anode::AdaptiveNode) =
    detected_drift(anode.adwin) && anode.alternate !== nothing

export AdaptiveNode, adaptive_leaf, is_leaf, is_split
export update_error!, error_rate, reset_drift!
export find_adaptive_leaf, should_grow_alternate, should_evaluate_replacement
