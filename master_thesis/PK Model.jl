using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations
using Plots
using Parameters

#Overtype of PK Models that will go into full model
#Potentially make step in between, PK_model_component
#then PK_model becomes list of drugs and corresponding model component, can switch out
abstract type PK_Model end

#later for checking if random effects need to be handled in inference
abstract type PK_Model_nonrandom <: PK_Model end

abstract type PK_Model_random <: PK_Model end
#For this need some sort of getter for which are random effects?

#Every model specification should have: Named tuple of parameters, list of keys of required covariates
#Every model needs create problem function

#Can that be handled globally?: Solve ODE system function given params, required covariates and doses

#Given that can be handled once for all models: Creation of dosing callbacks, 
#within group random effects Y/N also: creation of noisy measurements and returning likelihood 

#1)Specific model instances with their create problems

#A specific model instance, here very basic
@with_kw struct BasicModel{T<:NamedTuple, T2<:Tuple} <: PK_model_nonrandom
    θ::T=(k_el = 0.0, k_abs = 0.0)
    cov::T2 = () #no covariates required
end

function create_ode_system(mod::BasicModel; covariates=nothing) #does not actually need covariates, just for later
        @mtkmodel Internal begin
        @parameters begin
            k_el
            k_abs
        end
        @variables begin
            d(t) = 0.0  # depot compartment - no drug at beginning
            s(t) = 0.0  # internal/central compartment
            S(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
        end
        @equations begin
            D(d) ~ -k_abs * d
            D(s) ~ k_abs * d - k_el * s
            D(S) ~ s
        end
    end
    
    # Create the model with parameters
    θ = mod.θ
    @mtkcompile internal_model = Internal(; θ...)

    return internal_model
end

#2)Dosing
function dose_affect!(integrator; idx_d, dose_amount)
        integrator.u[idx_d] += dose_amount  # Add dose to depot (d)
end

function create_dosing_callbacks(dosing::Tuple, ode_system)
    callbacks = [
        PresetTimeCallback(
            dosing[i].t,
            integrator -> dose_affect!(
                integrator,
                idx_d = ModelingToolkit.variable_index(ode_system, dosing[i].state),
                dose_amount = dosing[i].dose
            ),
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

#3)Global functions for multiple models

#Problem creation and solution for nonrandom models, for random effects might have to do differently?
function create_problem(mod::PK_Model_nonrandom; dosing::Tuple, covariates<:NamedTuple=NamedTuple(), endpoint::AbstractFloat = 10.0)
    
    ode_system = create_ode_system(mod, covariates=covariates)

    # Create individual PresetTimeCallback for each dose
    # initialize is important if you have a dose at t=0
    callback_set = create_dosing_callbacks(dosing, ode_system)
    
    # Create ODE problem with callbacks
    problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, [], (0.0, endpoint), callback = callback_set)
    
    return problem
end

function solve_ODE(mod::PK_Model_nonrandom; dosing::Tuple, covariates<:NamedTuple=NamedTuple, endpoint::AbstractFloat=10.0)
    prob = create_problem(mod, dosing=dosing, covariates=covariates, endpoint=endpoint)
    sol = solve(prob,Tsit5())
    return sol
end
