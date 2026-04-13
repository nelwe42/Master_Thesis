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
abstract type PKModel end

#later for checking if random effects need to be handled in inference
abstract type PKModelNonrandom <: PKModel end

abstract type PKModelRandom <: PKModel end
#For this need some sort of getter for which are random effects?

#Every PKModelNonrandom specification should have: 
#θ: ComponentArray of parameters to be optimised
#cov: list of keys of required covariates
#set_daily_doses: Tuple explaining where daily_dose and potentially autoinduction has to be set for dose-dependent behavior
#entries are of the form (drug_param = daily_dose_param, drug_var = corresponding dose variable, autoinduction = true/false, ind_param = induction_parameter)
#keys: Named Tuple containing list of keys for s, S, d, obs where obs is list of tuples with (observation, corresponding s being observed)
#every model needs create_ode_system function

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

function create_ode_system(mod::PKBasic)
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

#A model for the PK behavior of Levetiracetam, when absorption is not modelled
@with_kw struct PKLEVNoAbsorption{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((c1 = 1.0, c2 = 1.0, c3 = 1.0, v1 = 40.0, v2 = 1.0, σ=0.5)) 
    cov::T2 = (:weight, :height, :kidney_disease) 
    set_daily_doses::T3 = ()
    keys::T4 = (d = SA[:s_LEV_unnormalised], s = SA[:s_LEV], S = SA[:S_LEV], obs = SA[(:obs_LEV, :s_LEV)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKLEVNoAbsorption) 
    #V = v1*(Body surface area normalised)^v2
    #CL = c1*(Weight normalised)^c2*(1-c3*(kidney disease yes/no))
    #Absorption immediate, just need depot compartment since unnormalised, elimination CL/V
    #Define body surface area as function of height and weight
    θ = mod.θ
    BSA_normalised(weight, height) = sqrt(weight*height/3600)/1.68
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters c1=θ.c1 c2=θ.c2 c3=θ.c3 v1=θ.v1 v2=θ.v2 σ=θ.σ #normal system parameters
    #callable parameters for covariates
    @parameters (weight::type_use)(..) [tunable=false] 
    @parameters (height::type_use)(..) [tunable=false] 
    @parameters (kidney_disease::type_use)(..) [tunable=false]
    @variables s_LEV_unnormalised(t) = 0.0 # depot compartment, here unnormalised
    @variables s_LEV(t)  # internal/central compartment
    @variables S_LEV(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_LEV(t)
    
    eqs = [s_LEV ~ s_LEV_unnormalised/(v1*BSA_normalised(weight(t), height(t))^v2), 
            D(s_LEV_unnormalised) ~ - (c1*(weight(t)/70)^c2*(1-kidney_disease(t)*c3)/(v1*BSA_normalised(weight(t), height(t))^v2)) * s_LEV_unnormalised,
            D(S_LEV) ~ s_LEV, 
            obs_LEV ~ Normal(s_LEV, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for the PK behavior of Carbamazepine
@with_kw struct PKCBZ{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs = 1.0, c1 = 1.0, c2 = 1.0, c3 = 0.0, v1 = 1.0, σ = 0.1)) 
    cov::T2 = (:prev_CBZ, :weight) 
    set_daily_doses::T3 = ((drug_param = :d_CBZ_daily, drug_var = :d_CBZ, autoinduction = true, ind_param = :ind_CBZ),) 
    #parameter to update and corresponding state name for updates, bool if autoinduction, name of autoinduction parameter
    keys::T4 = (d = SA[:d_CBZ], s = SA[:s_CBZ], S = SA[:S_CBZ], obs = SA[(:obs_CBZ, :s_CBZ)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKCBZ) 
    #k_abs constant, V weight dependent for identifiability
    #CL = c1*(c2^received CBZ for more than 14 days yes/no) + ln(daily_dose/400)*c3
    #Absorption rate k_abs/V, elimination CL/V
    #to ensure makes sense for daily_dose = 0 take maximum with ln(1/4), 100mg as minimal dosis makes sense, 
    #so will not change model in actual application
    θ = mod.θ
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters k_abs=θ.k_abs c1 = θ.c1 c2 = θ.c2 c3 = θ.c3 v1 = θ.v1 σ=θ.σ #normal system parameters
    @parameters d_CBZ_daily = 0.0 [tunable=false] #parameter for daily dose updated by callback
    #callable parameters for covariates
    @parameters (prev_CBZ::type_use)(..) [tunable=false] 
    @parameters (weight::type_use)(..) [tunable=false]
    @variables d_CBZ(t) = 0.0  # depot compartment - no drug at beginning
    @variables s_CBZ(t) = 0.0  # internal/central compartment
    @variables S_CBZ(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_CBZ(t)
    @parameters ind_CBZ = 14*prev_CBZ(0.0) [tunable = false] #indicator if induction occurs, needs to be >= 14 for that to be true
    #d_CBZ is not concentration but dose, so rate there not normalised by volume
    eqs = [D(d_CBZ) ~ -k_abs * d_CBZ,
            D(s_CBZ) ~ k_abs/(v1*weight(t)) * d_CBZ - (c1*(c2^(ind_CBZ>=14))+c3*log(max(d_CBZ_daily/400, 1/4)))/(v1*weight(t)) * s_CBZ,
            D(S_CBZ) ~ s_CBZ,
            obs_CBZ ~ Normal(s_CBZ, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for the PK behavior of Valproate
@with_kw struct PKVPA{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs = 1.0, c1 = 1.0, c2 = 1.0, c3 = 0.0, c4 = 0.0, v1 = 1.0, σ = 0.1)) 
    cov::T2 = (:gender, :weight) 
    set_daily_doses::T3 = ((drug_param = :d_VPA_daily, drug_var = :d_VPA, autoinduction = false, ind_param = :none),) 
    #parameter to update and corresponding state name for updates, bool if autoinduction, name of autoinduction parameter (not present here, just for sake of completeness)
    keys::T4 = (d = SA[:d_VPA], s = SA[:s_VPA], S = SA[:S_VPA], obs = SA[(:obs_VPA, :s_VPA)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKVPA) 
    #k_abs constant, V=v1*weight
    #CL = c1*weight^c2*dose(mg/day)^c3*c4^gender(1 for female, 0 for male)
    #for multidrugmodel CBZ and PB dependence in clearance
    #Absorption rate k_abs/V, elimination CL/V
    θ = mod.θ
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters k_abs=θ.k_abs c1 = θ.c1 c2 = θ.c2 c3 = θ.c3 c4 = θ.c4 v1 = θ.v1 σ=θ.σ #normal system parameters
    @parameters d_VPA_daily = 0.0 [tunable=false] #parameter for daily dose updated by callback
    #callable parameters for covariates
    @parameters (gender::type_use)(..) [tunable=false] 
    @parameters (weight::type_use)(..) [tunable=false]
    @variables d_VPA(t) = 0.0  # depot compartment - no drug at beginning
    @variables s_VPA(t) = 0.0  # internal/central compartment
    @variables S_VPA(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_VPA(t)
    #d_VPA is not concentration but dose, so rate there not normalised by volume
    eqs = [D(d_VPA) ~ -k_abs * d_VPA,
            D(s_VPA) ~ k_abs/(v1*weight(t)) * d_VPA - (c1*weight(t)^c2*d_VPA_daily^c3*c4^gender(t))/(v1*weight(t)) * s_VPA,
            D(S_VPA) ~ s_VPA,
            obs_VPA ~ Normal(s_VPA, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#2)Dosing and callback creation for all models
function dose_affect!(integrator; idx_d, dose_amount)
        integrator.u[idx_d] += dose_amount  # Add dose to depot (d)
end

#For when daily dose as a parameter must be updated in ODE
function daily_dose_affect!(integrator; id_param, daily_dose)
    integrator.p[id_param] = daily_dose
end

#For when autoinduction parameter must be updated in ODE
function induction_dose_affect!(integrator; dose_param, ind_param)
    integrator.p[ind_param] += (2*(integrator.p[dose_param]>0)-1)
    if integrator.p[ind_param] < 0
        integrator.p[ind_param] = 0.0
    end
end

function create_dosing_callbacks(dosing::AbstractVector, ode_system; names::NamedTuple, set_daily_doses::Tuple = ())
    #save_positions = (false, false) so no two values at timepoint possible, bad for likelihood calculation/measurement generator
    #callbacks to inject doses
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

    #callbacks to set daily dose parameter where necessary
    callbacks_daily_doses = [PeriodicCallback( 
                        integrator -> daily_dose_affect!(integrator, id_param = ModelingToolkit.parameter_index(ode_system, d.drug_param),
                                        daily_dose = sum([dose.dose for dose in dosing if (integrator.t ≤ dose.t < (integrator.t+1) && dose.state == d.drug_var)])),
                        1.0, initial_affect = true, final_affect = true, save_positions = (false, false)) #affect called every 1.0 time unit (days), also at initial and final point
                        for d in set_daily_doses]
    
    #callbacks for autoinduction parameter where necessary
    callbacks_autoinduction = [PeriodicCallback( 
                        integrator -> induction_dose_affect!(integrator, dose_param = ModelingToolkit.parameter_index(ode_system, d.drug_param),
                                        ind_param = ModelingToolkit.parameter_index(ode_system, d.ind_param)),
                        1.0, initial_affect = true, final_affect = true, save_positions = (false, false)) #affect called every 1.0 time unit (days), also at initial and final point
                        for d in set_daily_doses if d.autoinduction]

    return CallbackSet(callbacks..., callbacks_daily_doses..., callbacks_autoinduction...)
end

#3)Global functions for nonrandom models

#Getter for keys of variable roles from PK model
function get_keys_PK(mod::PKModel)
    return mod.keys
end

#Problem creation with covariates and callbacks
function create_problem(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat = 10.0)
    
    ode_system = create_ode_system(mod)

    #Create Callbacks for doses, autoinduction and other potential dose related behavior
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

    #Create Callbacks for doses, autoinduction and other potential dose related behavior
    names = get_keys_PK(mod)
    callback_set = create_dosing_callbacks(dosing, ode_system, names = names, set_daily_doses = mod.set_daily_doses)

    #interpolate covariates constant
    covariate_interpolation = Dict((name => ConstantInterpolation([value, value], [0.0, endpoint])) for (name, value) in pairs(covariates))

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

#solve for a different θ containing all PK Model parameters, for problem not given
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

#solve_PK for given problem, indices of parameters given
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
    if any(x -> !isfinite(x), θ)
        return -Inf
    end
    loglikeli = zero(eltype(θ))
    for measure in person.measurements
        predicted_state = sol(measure.timepoint, idxs = measure.state[2])
        if !(isfinite(predicted_state))
            return -Inf
        end
        loglikeli += logpdf(sol(measure.timepoint, idxs = measure.state[1]), measure.measurement)
    end
    return loglikeli
end

#generates noisy measurements, returns solution for use in seizure model
#if no endpoint given assumes timepoints are increasing
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

#4)Functions for visualisation

#functions for plotting fit, with or without true or estimated parameters given, solutions given
function plot_fit(mod::PKModel, data::Tuple; sols_true::Union{AbstractVector, Nothing} = nothing, sols_estimated::Union{AbstractVector, Nothing} = nothing, individuals::AbstractVector = [1], display_plot::Bool = true)
    
    output = Plots.Plot[]
    sols = sols_true
    sols2 = sols_estimated
    #Plot PK behavior (for each drug)
    names = get_keys_PK(mod)
    #Iterate over indices for which to plot
    for i in individuals
        #Iterate over drugs 
        for s in names.s
            pl = plot(xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s) for person $(i)")
            #true plot if param specified
            if !isnothing(sols)
                plot!(sols[i], idxs = s, label="Concentration $(s)")
            end
            #add scattered measurements
            x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
            y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
            plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "")
            #add estimate plot if specified
            if !isnothing(sols2)
                plot!(sols2[i], idxs = s, label="Estimated concentration $(s)", linecolor = :red)
            end
            #add to output
            append!(output, [pl])
            if display_plot
                display(pl)
            end
        end
    end

    #Plot all trajectories in one
    Population_size = length(data)
    for s in names.s
        pl = plot(xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s)")
        if !isnothing(sols)
            plot!(sols[1], idxs = s, label="Concentration $(s)", linecolor = :blue)
            for i in 2:Population_size
                plot!(sols[i], idxs = s, label = "", linecolor = :blue)
            end
        end
        for i in 1:Population_size
            x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
            y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
            plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "")
        end
        #add estimate plots
        if !isnothing(sols2)
            plot!(sols2[1], idxs = s, label="Estimated concentration $(s)", linecolor = :red)
            for i in 2:Population_size
                plot!(sols2[i], idxs = s, label = "", linecolor = :red)
            end
        end
        plot!(legend=:outerbottom, legendcolumns=2)
        #add to output
        append!(output, [pl])
        if display_plot
            display(pl)
        end
    end

    return output
end

#function above for solutions not given
function plot_fit_param(mod::PKModel, data::Tuple; true_param::Union{ComponentArray, Nothing} = mod.θ, estimate_param::Union{ComponentArray, Nothing} = nothing, individuals::AbstractVector = [1], 
    endpoint::AbstractFloat = data[1].measurements[end].timepoint, display_plot::Bool = true, options = (AutoTsit5(Rosenbrock23()),))
    
    if !isnothing(true_param)
        sols = [solve_PK(mod, true_param, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
    else
        sols = nothing
    end
    if !isnothing(estimate_param)
        sols2 = [solve_PK(mod, estimate_param, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
    else 
        sols2 = nothing
    end
    output = plot_fit(mod, data, sols_true = sols, sols_estimated = sols2, individuals = individuals, display_plot = display_plot)
    return output
end