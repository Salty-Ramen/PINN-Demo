"""
Abstract supertype for state-space transforms used by the PINN training pipeline.

A concrete transform is responsible for three things:
1. mapping raw biological states into the coordinates seen by the network,
2. mapping transformed predictions back into raw coordinates, and
3. converting a raw-coordinate RHS into the transformed coordinates used in
   `loss_ODE` via the chain rule.
"""
abstract type AbstractStateTransform end

"""
No-op transform.

Use this when training directly in the raw state coordinates.
"""
struct IdentityTransform <: AbstractStateTransform end

"""
Per-state log transform with optional masking.

Fields
- `base`: logarithm base (`ℯ`, `10f0`, etc.)
- `shift`: per-state additive offset used before taking logs
- `mask`: indicates which state rows should be logged; unmasked rows are left
  unchanged
"""
struct LogTransform{T<:Real,V<:AbstractVector{T},B<:AbstractVector{Bool}} <: AbstractStateTransform
    base::T
    shift::V
    mask::B
end

"""
Affine transform `z = (y - μ) ./ σ` applied row-wise.

This covers ordinary z-scoring and also simple scale-only normalization when
`μ == 0`.
"""
struct ZScoreTransform{T<:Real,V<:AbstractVector{T}} <: AbstractStateTransform
    μ::V
    σ::V
end

# Identity leaves the state untouched in both directions.
forward_state(::IdentityTransform, Y) = Y
inverse_state(::IdentityTransform, Z) = Z

"""
Map raw state trajectories `Y` (states × batch) into transformed coordinates.

For `LogTransform`, only masked rows are logged; the remaining rows stay in the
original coordinates.
"""
function forward_state(tr::LogTransform, Y::AbstractMatrix)
    Z = copy(Y)
    for i in axes(Y, 1)
        if tr.mask[i]
            if tr.base == ℯ
                Z[i, :] .= log.(Y[i, :] .+ tr.shift[i])
            else
                Z[i, :] .= log.(tr.base, Y[i, :] .+ tr.shift[i])
            end
        end
    end
    return Z
end

"""
Map transformed trajectories `Z` back into raw biological coordinates.
"""
function inverse_state(tr::LogTransform, Z::AbstractMatrix)
    Y = copy(Z)
    for i in axes(Z, 1)
        if tr.mask[i]
            if tr.base == ℯ
                Y[i, :] .= exp.(Z[i, :]) .- tr.shift[i]
            else
                Y[i, :] .= tr.base .^ Z[i, :] .- tr.shift[i]
            end
        end
    end
    return Y
end

# Z-score/scale transform in the forward direction.
function forward_state(tr::ZScoreTransform, Y::AbstractMatrix)
    (Y .- tr.μ) ./ tr.σ
end

# Z-score/scale transform in the reverse direction.
function inverse_state(tr::ZScoreTransform, Z::AbstractMatrix)
    tr.σ .* Z .+ tr.μ
end

# Convenience overloads for callers working with a single state vector instead of
# a states×batch matrix.
forward_state(tr::AbstractStateTransform, y::AbstractVector) =
    vec(forward_state(tr, reshape(y, :, 1)))

inverse_state(tr::AbstractStateTransform, z::AbstractVector) =
    vec(inverse_state(tr, reshape(z, :, 1)))

"""
Wrap a raw-coordinate RHS so it can be compared against derivatives in the
transformed coordinates predicted by the state network.

Arguments
- `tr`: state transform used by the current training context
- `z`: transformed state trajectory from the network
- `g`: auxiliary neural outputs used by the model RHS
- `p`: model parameters in raw/physical coordinates
- `raw_rhs`: function that expects raw states and returns `dy/dt`

The returned value is `dz/dt`, i.e. the RHS expressed in the transformed state
coordinates. This is the key bridge between raw model equations and transform-
aware ODE losses.
"""
transformed_rhs(::IdentityTransform, z, g, p, raw_rhs) = raw_rhs(z, g, p)

# For affine transforms z = (y - μ) / σ, the chain rule gives dz/dt = (dy/dt)/σ.
function transformed_rhs(tr::ZScoreTransform, z::AbstractMatrix, g, p, raw_rhs)
    y = inverse_state(tr, z)
    f = raw_rhs(y, g, p)
    return f ./ tr.σ
end

# For log transforms z = log_b(y + shift), the chain rule gives
# dz/dt = (dy/dt) / ((y + shift) * log(base)) on the logged rows.
function transformed_rhs(tr::LogTransform, z::AbstractMatrix, g, p, raw_rhs)
    y = inverse_state(tr, z)
    f = raw_rhs(y, g, p)
    out = copy(f)

    for i in eachindex(tr.mask)
        if tr.mask[i]
            denom = y[i, :] .+ tr.shift[i]
            if tr.base == ℯ
                out[i, :] .= f[i, :] ./ denom
            else
                out[i, :] .= f[i, :] ./ (denom .* log(tr.base))
            end
        end
    end

    return out
end
