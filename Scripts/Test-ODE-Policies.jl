#!/usr/bin/env julia
#
# Test-ODE-Policies.jl
#
# Smoke test for ODE parameter assembly, box bounds, and Fminbox
# wrapping.  Exercises only the parameter + bounds machinery —
# no MLPs, no Optimization.jl solves, no Zygote.
#
# Run from the project root:
#   julia --project=. Scripts/Test-ODE-Policies.jl

using Pkg; Pkg.activate(".")

using ComponentArrays

include(joinpath(pwd(), "Src", "Fitting.jl"))

const FAILS = String[]
const PASSES = String[]

function check(fn, label::String)
    try
        fn()
        push!(PASSES, label)
        println("  PASS  $label")
    catch e
        push!(FAILS, "$label  →  $(sprint(showerror, e))")
        println("  FAIL  $label")
        println("        ", sprint(showerror, e))
    end
end

const ODE_INIT   = (β = 1f-4, δ = 1f0, c = 1f0)
const ODE_BOUNDS = (β = Float32[0, 1f-2], δ = Float32[0, 20], c = Float32[0, 20])

const MOCK_STATE = (W1 = randn(Float32, 4, 1), b1 = zeros(Float32, 4))
const MOCK_G     = (W1 = randn(Float32, 2, 1), b1 = zeros(Float32, 2))
const HP         = HyperParams(1f0, 1f0, 1f0, 1f0, 1f0)

println("==== build_trainables ====\n")

check("FixedHyper has StateMLP, gMLP, ODE_par") do
    ps = build_trainables(FixedHyper(), MOCK_STATE, MOCK_G, ODE_INIT, HP)
    @assert hasproperty(ps, :StateMLP)
    @assert hasproperty(ps, :gMLP)
    @assert hasproperty(ps, :ODE_par)
    @assert ps.ODE_par.β == ODE_INIT.β
end

check("AdaptiveHyper adds hyper block") do
    ps = build_trainables(AdaptiveHyper(), MOCK_STATE, MOCK_G, ODE_INIT, HP)
    @assert hasproperty(ps, :hyper)
    @assert hasproperty(ps, :ODE_par)
    @assert ps.hyper.log_ϵ_ic ≈ log(1f0)
end

println("\n==== build_box_bounds ====\n")

check("FixedHyper: ±Inf on MLPs, finite on all ODE keys") do
    ps = build_trainables(FixedHyper(), MOCK_STATE, MOCK_G, ODE_INIT, HP)
    lb, ub = build_box_bounds(ps, ODE_BOUNDS)
    @assert getaxes(lb) == getaxes(ps)
    @assert length(lb) == length(ps)
    @assert all(isinf, lb.StateMLP.W1)
    @assert all(isinf, ub.StateMLP.W1)
    @assert all(isinf, lb.gMLP.W1)
    @assert all(isinf, ub.gMLP.W1)
    @assert lb.ODE_par.β == 0f0
    @assert ub.ODE_par.β ≈ 1f-2
    @assert lb.ODE_par.δ == 0f0
    @assert ub.ODE_par.δ ≈ 20f0
    @assert lb.ODE_par.c == 0f0
    @assert ub.ODE_par.c ≈ 20f0
end

check("AdaptiveHyper: hyper block unconstrained, ODE bounded") do
    ps = build_trainables(AdaptiveHyper(), MOCK_STATE, MOCK_G, ODE_INIT, HP)
    lb, ub = build_box_bounds(ps, ODE_BOUNDS)
    @assert getaxes(lb) == getaxes(ps)
    @assert all(isinf, lb.hyper.log_ϵ_ic)
    @assert all(isinf, ub.hyper.log_ϵ_ic)
    @assert lb.ODE_par.β == 0f0
    @assert ub.ODE_par.β ≈ 1f-2
end

check("Default bounds (0, +Inf) when key missing from ode_bounds") do
    ps = build_trainables(FixedHyper(), MOCK_STATE, MOCK_G, ODE_INIT, HP)
    bounds_partial = (β = Float32[0, 1f-2],)   # δ and c missing
    lb, ub = build_box_bounds(ps, bounds_partial)
    @assert lb.ODE_par.β == 0f0
    @assert ub.ODE_par.β ≈ 1f-2
    @assert lb.ODE_par.δ == 0f0
    @assert isinf(ub.ODE_par.δ) && ub.ODE_par.δ > 0
    @assert lb.ODE_par.c == 0f0
    @assert isinf(ub.ODE_par.c) && ub.ODE_par.c > 0
end

check("No ODE_par block → all ±Inf") do
    ps = ComponentArray(
        StateMLP = MOCK_STATE,
        gMLP     = MOCK_G,
    )
    lb, ub = build_box_bounds(ps, ODE_BOUNDS)
    @assert all(isinf, lb)
    @assert all(x -> x > 0, ub)
    @assert all(x -> x < 0, lb)
end

println("\n==== _maybe_wrap_fminbox ====\n")

import Optim

check("LBFGS wrapped with Fminbox when bounds active") do
    opt = OptimizationOptimJL.LBFGS(m = 25)
    wrapped = _maybe_wrap_fminbox(opt, Val(true))
    @assert wrapped isa Optim.Fminbox
end

check("LBFGS not wrapped when no bounds") do
    opt = OptimizationOptimJL.LBFGS(m = 25)
    result = _maybe_wrap_fminbox(opt, Val(false))
    @assert result === opt
end

check("Adam not wrapped even with bounds") do
    opt = OptimizationOptimisers.Adam(1f-3)
    result = _maybe_wrap_fminbox(opt, Val(true))
    @assert result === opt
end

println("\n", "="^60)
println("  Summary: $(length(PASSES)) passed, $(length(FAILS)) failed")
println("="^60)
if !isempty(FAILS)
    println("\nFailures:")
    for f in FAILS
        println("  - $f")
    end
    exit(1)
end
