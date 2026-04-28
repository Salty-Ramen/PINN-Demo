#!/usr/bin/env julia

# Lightweight sanity checks for ODE parameter dispatch policy.
# Run with:
#   julia Scripts/test_ode_param_dispatch.jl

include(joinpath(@__DIR__, "..", "Src", "Fitting.jl"))

println("Running ODE parameter dispatch checks...")

# --- Case 1: default behaviour (train all) ---
ode_init = (β = 1.0f0, γ = 2.0f0, c = 3.0f0)
all_policy = TrainAllODE()

all_trainables = initial_ode_trainables(ode_init, all_policy)
@assert all_trainables == ode_init

all_resolved = resolve_ode_params((β = 4.0f0, γ = 5.0f0, c = 6.0f0), all_policy, ode_init)
@assert all_resolved == (β = 4.0f0, γ = 5.0f0, c = 6.0f0)

println("✓ TrainAllODE passes")

# --- Case 2: selective behaviour (some fixed, some trainable) ---
sel_policy = SelectiveODE((β = :train, γ = 1.5f0, c = :train))
_validate_ode_policy(ode_init, sel_policy)

sel_trainables = initial_ode_trainables(ode_init, sel_policy)
@assert sel_trainables == (β = 1.0f0, c = 3.0f0)

sel_resolved = resolve_ode_params((β = 7.0f0, c = 9.0f0), sel_policy, ode_init)
@assert sel_resolved == (β = 7.0f0, γ = 1.5f0, c = 9.0f0)

println("✓ SelectiveODE (some fixed) passes")

# --- Case 3: selective behaviour (fix all) ---
fix_all_policy = SelectiveODE((β = 0.1f0, γ = 0.2f0, c = 0.3f0))
fix_all_trainables = initial_ode_trainables(ode_init, fix_all_policy)
@assert isempty(propertynames(fix_all_trainables))

fix_all_resolved = resolve_ode_params((;), fix_all_policy, ode_init)
@assert fix_all_resolved == (β = 0.1f0, γ = 0.2f0, c = 0.3f0)

println("✓ SelectiveODE (fix all) passes")
println("All ODE parameter dispatch checks passed.")
