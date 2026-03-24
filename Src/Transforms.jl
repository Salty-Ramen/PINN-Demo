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

# ──────────────────────────────────────────────────────────
# Identity: leaves the state untouched in both directions.
# ──────────────────────────────────────────────────────────
forward_state(::IdentityTransform, Y) = Y
inverse_state(::IdentityTransform, Z) = Z

# ──────────────────────────────────────────────────────────
# LogTransform: forward / inverse
#
# All implementations are non-mutating so that Zygote can
# differentiate through them. We build each transformed row
# independently and vcat them rather than writing into a
# pre-allocated buffer with .= (which Zygote rejects).
# ──────────────────────────────────────────────────────────

"""
    forward_state(tr::LogTransform, Y)

Map raw state trajectories `Y` (states × batch) into log-transformed
coordinates.  Only masked rows are logged; the remaining rows pass through
unchanged.
"""
function forward_state(tr::LogTransform, Y::AbstractMatrix)
    rows = map(axes(Y, 1)) do i
        row = Y[i:i, :]                       # 1×B slice (no mutation)
        if tr.mask[i]
            if tr.base == ℯ
                log.(row .+ tr.shift[i])
            else
                log.(row .+ tr.shift[i]) ./ log(tr.base)
            end
        else
            row
        end
    end
    return reduce(vcat, rows)                  # reassemble into states×B
end

"""
    inverse_state(tr::LogTransform, Z)

Map log-transformed trajectories `Z` back into raw biological coordinates.
"""
function inverse_state(tr::LogTransform, Z::AbstractMatrix)
    rows = map(axes(Z, 1)) do i
        row = Z[i:i, :]
        if tr.mask[i]
            if tr.base == ℯ
                exp.(row) .- tr.shift[i]
            else
                tr.base .^ row .- tr.shift[i]
            end
        else
            row
        end
    end
    return reduce(vcat, rows)
end

# ──────────────────────────────────────────────────────────
# ZScoreTransform: forward / inverse
# ──────────────────────────────────────────────────────────
function forward_state(tr::ZScoreTransform, Y::AbstractMatrix)
    (Y .- tr.μ) ./ tr.σ
end

function inverse_state(tr::ZScoreTransform, Z::AbstractMatrix)
    tr.σ .* Z .+ tr.μ
end

# ──────────────────────────────────────────────────────────
# Vector convenience overloads
# ──────────────────────────────────────────────────────────
forward_state(tr::AbstractStateTransform, y::AbstractVector) =
    vec(forward_state(tr, reshape(y, :, 1)))

inverse_state(tr::AbstractStateTransform, z::AbstractVector) =
    vec(inverse_state(tr, reshape(z, :, 1)))

# ──────────────────────────────────────────────────────────
# transformed_rhs: chain-rule bridge between raw model
# equations and transform-aware ODE losses
# ──────────────────────────────────────────────────────────

"""
    transformed_rhs(tr, z, g, p, raw_rhs)

Wrap a raw-coordinate RHS so it can be compared against derivatives in the
transformed coordinates predicted by the state network.

The returned value is `dz/dt`, i.e. the RHS expressed in the transformed state
coordinates.
"""
transformed_rhs(::IdentityTransform, z, g, p, raw_rhs) = raw_rhs(z, g, p)

# Affine: dz/dt = (dy/dt) / σ
function transformed_rhs(tr::ZScoreTransform, z::AbstractMatrix, g, p, raw_rhs)
    y = inverse_state(tr, z)
    f = raw_rhs(y, g, p)
    return f ./ tr.σ
end

# Log: dz/dt = (dy/dt) / ((y + shift) · ln(base)) on logged rows; identity on others.
# Built row-by-row with map + vcat to stay non-mutating for Zygote.
function transformed_rhs(tr::LogTransform, z::AbstractMatrix, g, p, raw_rhs)
    y = inverse_state(tr, z)
    f = raw_rhs(y, g, p)

    rows = map(axes(f, 1)) do i
        f_row = f[i:i, :]                      # 1×B slice
        if tr.mask[i]
            denom = y[i:i, :] .+ tr.shift[i]   # 1×B
            if tr.base == ℯ
                f_row ./ denom
            else
                f_row ./ (denom .* log(tr.base))
            end
        else
            f_row
        end
    end
    return reduce(vcat, rows)
end
