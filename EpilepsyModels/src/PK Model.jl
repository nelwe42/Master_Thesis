using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations
using Plots
using Parameters
using Distributions
using Random
using ComponentArrays
using Accessors
using StaticArrays
using DataInterpolations
using DiffEqCallbacks

#Overtype of PK Models that will go into full model
#Potentially make step in between, PK_model_component
#then PK_model becomes list of drugs and corresponding model component, can switch out
abstract type PKModel end

#later for checking if random effects need to be handled in inference
abstract type PKModelNonrandom <: PKModel end

abstract type PKModelRandom <: PKModel end
#For this need some sort of getter for which are random effects?

#Every model specification should have: ComponentArray of parameters, list of keys of required covariates
#Every model needs create problem function, getter for keys of s, S, d

#Given that can be handled once for all models: Creation of dosing callbacks, 
#within group random effects Y/N also: creation of noisy measurements and returning likelihood 
#solve ODE system given params, required covariates and doses

#1)Specific model instances with their create problems

#A specific model instance, here very basic
@with_kw struct PKBasic{T<:ComponentArray, T2<:Tuple, T3<: Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_el = 1.0, k_abs = 1.0, σ=0.5)) 
    cov::T2 = () #no covariates required
    set_daily_doses::T3 = () #no information about daily doses required
    keys::T4 = (d = SA[:d], s = SA[:s], S = SA[:S], obs = SA[(:obs, :s)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKBasic) #does not actually need covariates, just for later
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

#A model for the PK behavior of Levetiracetam
@with_kw struct PKLEV{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs = 1.0, c1 = 1.0, c2 = 1.0, c3 = 1.0, v1 = 40.0, v2 = 1.0, σ=0.5)) 
    cov::T2 = (:weight, :height, :kidney_disease) 
    set_daily_doses::T3 = ()
    keys::T4 = (d = SA[:d_LEV], s = SA[:s_LEV], S = SA[:S_LEV], obs = SA[(:obs_LEV, :s_LEV)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKLEV) 
    #V = v1*(Body surface area normalised)^v2
    #CL = c1*(Weight normalised)^c2*(1-c3*(kidney disease yes/no))
    #Absorption rate k_abs/V, elimination CL/V
    #Define body surface area as function of height and weight
    θ = mod.θ
    BSA_normalised(weight, height) = sqrt(weight*height/3600)/1.68
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters k_abs=θ.k_abs c1=θ.c1 c2=θ.c2 c3=θ.c3 v1=θ.v1 v2=θ.v2 σ=θ.σ #normal system parameters
    #callable parameters for covariates
    @parameters (weight::type_use)(..) [tunable=false] 
    @parameters (height::type_use)(..) [tunable=false] 
    @parameters (kidney_disease::type_use)(..) [tunable=false]
    @variables d_LEV(t) = 0.0  # depot compartment - no drug at beginning
    @variables s_LEV(t) = 0.0  # internal/central compartment
    @variables S_LEV(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_LEV(t)
    #d_LEV is not concentration but dose, so rate there not normalised by volume
    eqs = [D(d_LEV) ~ -k_abs * d_LEV,
            D(s_LEV) ~ (k_abs/(v1*BSA_normalised(weight(t), height(t))^v2)) * d_LEV - (c1*(weight(t)/70)^c2*(1-kidney_disease(t)*c3)/(v1*BSA_normalised(weight(t), height(t))^v2)) * s_LEV,
            D(S_LEV) ~ s_LEV, 
            obs_LEV ~ Normal(s_LEV, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for just testing right now
@with_kw struct PKCBZ{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs = 1.0, c1 = 1.0, c2 = 1.0, c3 = 0.0, v = 100.0, σ = 0.1)) 
    cov::T2 = (:prev_CBZ,) 
    set_daily_doses::T3 = ((:d_CBZ_daily, :d_CBZ),) #parameter to update and corresponding state name for updates
    keys::T4 = (d = SA[:d_CBZ], s = SA[:s_CBZ], S = SA[:S_CBZ], obs = SA[(:obs_CBZ, :s_CBZ)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKCBZ) 
    #V and k_abs constant
    #CL = c1*(c2^received CBZ for more than 14 days yes/no) + ln(daily_dose/400)*c3
    #Absorption rate k_abs/V, elimination CL/V
    #to ensure makes sense for daily_dose = 0 take maximum with ln(1/4), 100mg as minimal dosis makes sense, 
    #so will not change model in actual application
    θ = mod.θ
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters k_abs=θ.k_abs c1 = θ.c1 c2 = θ.c2 c3 = θ.c3 v = θ.v,  σ=θ.σ #normal system parameters
    @parameters d_CBZ_daily = 0.0 [tunable=false] #parameter for daily dose updated by callback
    #callable parameters for covariates
    @parameters (prev_CBZ::type_use)(..) [tunable=false] 
    @variables d_CBZ(t) = 0.0  # depot compartment - no drug at beginning
    @variables s_CBZ(t) = 0.0  # internal/central compartment
    @variables S_CBZ(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_CBZ(t)
    @variable Ind(t) = prev_CBZ(0.0) #indicator if induction occurs, i.e. person has received 14 days of CBZ
    #potentially pass periodic discrete event into system as ; discrete_events = events after t
    #write periodic event as [periodicity => [affect, be sure to use pre() for variables here]]
    #might have to specify discrete_parameters to be able to affect them?
    #potentially just add 1/14 every day where d_CBZ_daily is positive, take min(1, Ind) in eqs
    #d_LEV is not concentration but dose, so rate there not normalised by volume
    eqs = [D(d_CBZ) ~ -k_abs * d_CBZ,
            D(s_CBZ) ~ (k_abs/v) * d_CBZ - (c1*(c2^Ind)+c3*log(max(d_CBZ_daily/400, 1/4)))/v * s_CBZ,
            D(S_CBZ) ~ s_CBZ,
            obs_CBZ ~ Normal(s_CBZ, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#2)Dosing for all models
function dose_affect!(integrator; idx_d, dose_amount)
        integrator.u[idx_d] += dose_amount  # Add dose to depot (d)
end

#For when daily dose as a parameter must be updated in ODE
function daily_dose_affect!(integrator; id_param, daily_dose)
    integrator.p[id_param] = daily_dose
end

function create_dosing_callbacks(dosing::AbstractVector, ode_system; names::NamedTuple, set_daily_doses::Tuple = ())
    #save_positions = (false, false) so no two values at timepoint possible, bad for likelihood calculation/measurement generator
    @inbounds callbacks = [
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
        for i in eachindex(dosing) if dosing[i].state in names.d #check PK model supports this drug
    ]

    #Maybe go to periodic callback here?
    #callbacks_daily = [PresetTimeCallback(day, 
    #                    integrator -> daily_dose_affect!(integrator, id_param = ModelingToolkit.parameter_index(ode_system, d[1]),
    #                                    daily_dose = sum([dose.dose for dose in dosing if (day ≤ dose.t < (day+1) && dose.state == d[2])]))) 
    #                    for d in set_daily_doses for day in 0:endpoint]
    #Maybe need initialize here? Might be different because set = instead of +=
    callbacks_daily = [PeriodicCallback( 
                        integrator -> daily_dose_affect!(integrator, id_param = ModelingToolkit.parameter_index(ode_system, d[1]),
                                        daily_dose = sum([dose.dose for dose in dosing if (integrator.t ≤ dose.t < (integrator.t+1) && dose.state == d[2])])),
                        1.0, initial_affect = true, final_affect = true, save_positions = (false, false)) #affect called every 1.0 time unit (days), also at initial and final point
                        for d in set_daily_doses]
    #set save_positions here?

    return CallbackSet(callbacks..., callbacks_daily...)
end

#3)Global functions for nonrandom models

#Getter for keys of variable roles from PK model
function get_keys_PK(mod::PKModel)
    return mod.keys
end

#Problem creation and solution for nonrandom models, for random effects might have to do differently?
function create_problem(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat = 10.0)
    
    ode_system = create_ode_system(mod)

    # Create individual PresetTimeCallback for each dose
    # initialize is important if you have a dose at t=0
    names = get_keys_PK(mod)
    callback_set = create_dosing_callbacks(dosing, ode_system, names = names, set_daily_doses = mod.set_daily_doses)

    #interpolate covariates constant
    covariate_interpolation = Dict((name => ConstantInterpolation([value, value], [0.0, endpoint])) for (name, value) in pairs(covariates))
    # Create ODE problem with callbacks
    problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, covariate_interpolation, (0.0, endpoint), callback = callback_set)
    
    return problem
end

#same function for ode_system already given, given a person instead of dosing and covariates
function create_problem(mod::PKModelNonrandom, ode_system::ODESystem; person::Person, endpoint::AbstractFloat = 10.0)
    dosing = person.dosing
    covariates = NamedTuple{mod.cov}(person.covariates)
    # Create individual PresetTimeCallback for each dose
    # initialize is important if you have a dose at t=0
    names = get_keys_PK(mod)
    callback_set = create_dosing_callbacks(dosing, ode_system, names = names, set_daily_doses = mod.set_daily_doses)

    #interpolate covariates constant
    covariate_interpolation = Dict((name => ConstantInterpolation([value, value], [0.0, endpoint])) for (name, value) in pairs(covariates))
    #covariate_interpolation = Dict((name => value for (name,value) in pairs(covariates)))

    # Create ODE problem with callbacks
    problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, covariate_interpolation, (0.0, endpoint), callback = callback_set)
    
    return problem
end

#solve ODE without system given
function solve_ODE(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat=10.0, options = (AutoTsit5(Rosenbrock23()),))
    prob = create_problem(mod, dosing=dosing, covariates=covariates, endpoint=endpoint)
    sol = solve(prob,options...; callback = PositiveDomain())
    return sol
end

#solve ODE given system
function solve_ODE(mod::PKModelNonrandom, sys::ODESystem; person::Person, endpoint::AbstractFloat=10.0, options = (AutoTsit5(Rosenbrock23()),))
    prob = create_problem(mod, sys, person=person, endpoint=endpoint)
    sol = solve(prob,options...; callback = PositiveDomain())
    return sol
end

#θ all PK Model parameters, for problem not given
function solve_PK(mod::PKModelNonrandom, θ::ComponentArray, person::Person; endpoint::AbstractFloat = 10.0, options = (AutoTsit5(Rosenbrock23()),))
    cov = NamedTuple{mod.cov}(person.covariates)
    ode_system = create_ode_system(mod)
    prob = create_problem(mod, dosing=person.dosing, covariates=cov, endpoint=endpoint)
    indices_θ = [ModelingToolkit.parameter_index(ode_system, x).idx for x in keys(θ)]
    mkt_parameters = prob.p
    new_mkt_parameters = Accessors.@set mkt_parameters.tunable[indices_θ] = θ
    T = promote_type(eltype(θ), eltype(new_mkt_parameters.tunable))
    prob_use = remake(prob, p=new_mkt_parameters; u0 = T.(prob.u0))
    sol = solve(prob_use, options...; callback = PositiveDomain())
    return sol
end

#solve_PK for problem, indices of parameters given
function solve_PK(prob::ODEProblem, ode_system::ODESystem, θ::ComponentArray; indices_θ::AbstractVector, options = (AutoTsit5(Rosenbrock23()),))
    #indices_θ = [ModelingToolkit.parameter_index(ode_system, x).idx for x in keys(θ)]
    mkt_parameters = prob.p
    new_mkt_parameters = Accessors.@set mkt_parameters.tunable[indices_θ] = θ
    T = promote_type(eltype(θ), eltype(new_mkt_parameters.tunable))
    prob_use = remake(prob, p=new_mkt_parameters; u0 = T.(prob.u0))
    sol = solve(prob_use, options...; callback = PositiveDomain())
    return sol
end

#likelihood when solution not given
function get_PK_loglikelihood(θ::ComponentArray, m::PKModel, person::Person; options = (AutoTsit5(Rosenbrock23()),))
    sol = solve_PK(m, θ, person, endpoint = person.measurements[end].timepoint, options = options)
    return get_PK_loglikelihood(θ, person, sol)
end

#likelihood when solution given
function get_PK_loglikelihood(θ::ComponentArray, person::Person; sol)
    loglikeli = 0
    for measure in person.measurements
        loglikeli += logpdf(sol(measure.timepoint, idxs = measure.state[1]), measure.measurement)
    end
    return loglikeli
end

#assumed that timepoints are increasing, returns solution for use in seizure model
function generate_measurements!(mod::PKModel, person::Person; timepoints::AbstractVector, endpoint::AbstractFloat = timepoints[end], options = (AutoTsit5(Rosenbrock23()),))
    cov = NamedTuple{mod.cov}(person.covariates)
    sol = solve_ODE(mod, dosing = person.dosing, covariates = cov, endpoint = endpoint, options = options)
    if !(SciMLBase.successful_retcode(sol))
        @warn "Unsuccessful ODE solve in data generation, you might want to adjust model parameters"
    end
    names = get_keys_PK(mod)
    measurements = [(timepoint = timepoint, measurement = rand(sol(timepoint, idxs = obs[1])), state = obs) for timepoint in timepoints for obs in names.obs]
    append!(person.measurements, measurements)
    return sol
end

#same function but given ODE system
function generate_measurements!(mod::PKModel, sys::ODESystem, person::Person; timepoints::AbstractVector, endpoint::AbstractFloat = timepoints[end], options = (AutoTsit5(Rosenbrock23()),))
    sol = solve_ODE(mod, sys, person=person, endpoint = endpoint, options = options)
    if !(SciMLBase.successful_retcode(sol))
        @warn "Unsuccessful ODE solve in data generation, you might want to adjust model parameters"
    end
    names = get_keys_PK(mod)
    measurements = [(timepoint = timepoint, measurement = rand(sol(timepoint, idxs = obs[1])), state = obs) for timepoint in timepoints for obs in names.obs]
    append!(person.measurements, measurements)
    return sol
end