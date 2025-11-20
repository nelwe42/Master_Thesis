using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations
using Plots
using Parameters
using Distributions
using Random
using ComponentArrays
using Accessors

#set seed
Random.seed!(42)

#Overtype of PK Models that will go into full model
#Potentially make step in between, PK_model_component
#then PK_model becomes list of drugs and corresponding model component, can switch out
abstract type PKModel end

#later for checking if random effects need to be handled in inference
abstract type PKModelNonrandom <: PKModel end

abstract type PKModelRandom <: PKModel end
#For this need some sort of getter for which are random effects?

#Every model specification should have: ComponentArray of parameters, list of keys of required covariates
#Every model needs create problem function

#Can that be handled globally?: Solve ODE system function given params, required covariates and doses

#Given that can be handled once for all models: Creation of dosing callbacks, 
#within group random effects Y/N also: creation of noisy measurements and returning likelihood 

#1)Specific model instances with their create problems

#A specific model instance, here very basic
@with_kw struct PKBasic{T<:ComponentArray, T2<:Tuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_el = 1.0, k_abs = 1.0, σ=0.5)) #σ is standard deviation in logscale
    cov::T2 = () #no covariates required
end

function create_ode_system(mod::PKBasic; covariates=nothing) #does not actually need covariates, just for later
        @mtkmodel Internal begin
        @parameters begin
            k_el
            k_abs
            σ
        end
        @variables begin
            d(t) = 0.0  # depot compartment - no drug at beginning
            s(t) = 0.0  # internal/central compartment
            S(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
            obs(t)
        end
        @equations begin
            D(d) ~ -k_abs * d
            D(s) ~ k_abs * d - k_el * s
            D(S) ~ s
            obs ~ Normal(s, σ)
        end
    end
    
    # Create the model with parameters
    θ = mod.θ
    @mtkcompile internal_model = Internal(; θ...)

    return internal_model
end

#2)Dosing for all models
function dose_affect!(integrator; idx_d, dose_amount)
        integrator.u[idx_d] += dose_amount  # Add dose to depot (d)
end

function create_dosing_callbacks(dosing::AbstractVector, ode_system)
    callbacks = [
        PresetTimeCallback(
            dosing[i].t,
            integrator -> dose_affect!(
                integrator,
                idx_d = ModelingToolkit.variable_index(ode_system, dosing[i].state),
                dose_amount = dosing[i].dose
            ), save_positions = (false, false),
            initialize = (cb, t, u, integrator) -> begin
                if cb.condition(t, u, integrator)
                    dose_affect!(
                        integrator;
                        idx_d = ModelingToolkit.variable_index(ode_system, dosing[i].state),
                        dose_amount = dosing[i].dose
                    )
                end
            end
        )
        for i in eachindex(dosing)
    ]
    
    return CallbackSet(callbacks...)
end

#3)Global functions for nonrandom models

#Problem creation and solution for nonrandom models, for random effects might have to do differently?
function create_problem(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat = 10.0)
    
    ode_system = create_ode_system(mod, covariates=covariates)

    # Create individual PresetTimeCallback for each dose
    # initialize is important if you have a dose at t=0
    callback_set = create_dosing_callbacks(dosing, ode_system)
    
    # Create ODE problem with callbacks
    problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, [], (0.0, endpoint), callback = callback_set)
    
    return problem
end

function solve_ODE(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat=10.0, options = [AutoTsit5(Rosenbrock23())])
    prob = create_problem(mod, dosing=dosing, covariates=covariates, endpoint=endpoint)
    sol = solve(prob,options...)
    return sol
end

#θ all PK Model parameters
function solve_PK(mod::PKModelNonrandom, θ::ComponentArray, person::Person; endpoint::AbstractFloat = 10.0, options = [AutoTsit5(Rosenbrock23())])
    cov = NamedTuple{mod.cov}(person.covariates)
    #Magic stuff that will hopefully fix AutoDiff
    ode_system = create_ode_system(mod)
    prob = create_problem(mod, dosing=person.dosing, covariates=cov, endpoint=endpoint)
    indices_θ = [ModelingToolkit.parameter_index(ode_system, x).idx for x in tunable_parameters(ode_system) if !(isinitial(x))]
    #tunable parameters only interested in not initial of a trajectory
    mkt_parameters = prob.p
    #println(mkt_parameters)
    #println(mkt_parameters.tunable)
    #println(indices_θ)
    #println(tunable_parameters(ode_system))
    new_mkt_parameters  =  Accessors.@set mkt_parameters.tunable[indices_θ] = θ
    new_prob = remake(prob, p=new_mkt_parameters)
    T = promote_type(eltype(θ), eltype(new_mkt_parameters.tunable))
    prob_use = remake(new_prob; u0 = T.(new_prob.u0))
    sol = solve(prob_use, options...)
    return sol
end

#likelihood when solution not given
function get_PK_loglikelihood(θ::ComponentArray, m::PKModel, person::Person)
    sol = solve_PK(m, θ, person, endpoint = person.measurements[end].timepoint)
    return get_PK_loglikelihood(θ, person, sol)
end

#likelihood when solution given
function get_PK_loglikelihood(θ::ComponentArray, person::Person; sol)
    loglikeli = 0
    for i in eachindex(person.measurements)
        measure = person.measurements[i]
        loglikeli = loglikeli + logpdf(sol(measure.timepoint, idxs = :obs), measure.measurement)
    end
    return loglikeli
end

#assumed that timepoints are increasing, returns solution for use in seizure model
function generate_measurements!(mod::PKModel, person::Person; timepoints::AbstractVector, options = [AutoTsit5(Rosenbrock23())])
    endpoint = timepoints[end]
    cov = NamedTuple{mod.cov}(person.covariates)
    sol = solve_ODE(mod, dosing = person.dosing, covariates = cov, endpoint = endpoint, options = options)
    for i in eachindex(timepoints)
        value = rand(sol(timepoints[i], idxs = :obs))
        pair = (timepoint = timepoints[i], measurement = value)
        push!(person.measurements,pair)
    end
    return sol
end