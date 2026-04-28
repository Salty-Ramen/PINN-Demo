#=-------------------------------------------------------------------------------
Utils.jl

Internal helper utilities.
Currently contains ODE-parameter dispatch/policy helpers; this file is intended
to host future utility functions as well.
-------------------------------------------------------------------------------=#

"""
    _validate_ode_policy(ode_par_init, policy)

Lightweight validation to ensure policy keys match the ODE parameter keys.
"""
function _validate_ode_policy(ode_par_init::NamedTuple, ::TrainAllODE)
    return nothing
end

function _validate_ode_policy(ode_par_init::NamedTuple, policy::SelectiveODE)
    pkeys = propertynames(ode_par_init)
    skeys = propertynames(policy.spec)
    pkeys == skeys || error("SelectiveODE keys must exactly match ODE_par_init keys.")
    return nothing
end

"""
    initial_ode_trainables(ode_par_init, policy) -> NamedTuple

Return the subset of ODE parameters that should be visible to the optimiser.
"""
initial_ode_trainables(ode_par_init::NamedTuple, ::TrainAllODE) = ode_par_init

function initial_ode_trainables(ode_par_init::NamedTuple, policy::SelectiveODE)
    pairs = Pair{Symbol, Any}[]
    for k in propertynames(ode_par_init)
        v = getproperty(policy.spec, k)
        if v === :train
            push!(pairs, k => getproperty(ode_par_init, k))
        end
    end
    return (; pairs...)
end

"""
    resolve_ode_params(ode_trainables, policy, ode_par_init) -> NamedTuple

Construct the full ODE parameter set used inside the ODE residual/loss by
combining trainable values with fixed values.
"""
resolve_ode_params(ode_trainables::NamedTuple, ::TrainAllODE, ode_par_init::NamedTuple) = ode_trainables

function resolve_ode_params(ode_trainables::NamedTuple, policy::SelectiveODE, ode_par_init::NamedTuple)
    pairs = Pair{Symbol, Any}[]
    for k in propertynames(ode_par_init)
        spec_v = getproperty(policy.spec, k)
        val = spec_v === :train ? getproperty(ode_trainables, k) : spec_v
        push!(pairs, k => val)
    end
    return (; pairs...)
end
