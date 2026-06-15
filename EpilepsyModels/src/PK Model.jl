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
using SymbolicUtils

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
    θ::T=ComponentArray((k_abs = 72.0, c1 = 72.0, c2 = 1.0, c3 = 1.0, v1 = 40.0, v2 = 1.0, σ=0.5)) 
    cov::T2 = (:weight, :height, :kidney_disease, :CLCr) 
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
    @parameters (CLCr::type_use)(..) [tunable=false]
    @variables d_LEV(t) = 0.0  # depot compartment - no drug at beginning
    @variables s_LEV(t) = 0.0  # internal/central compartment
    @variables S_LEV(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_LEV(t)
    #d_LEV is not concentration but dose, so rate there not normalised by volume
    eqs = [D(d_LEV) ~ -k_abs * d_LEV,
            D(s_LEV) ~ (k_abs/(v1*BSA_normalised(weight(t), height(t))^v2)) * d_LEV - (c1*(weight(t)/70)^c2*((CLCr(t)+50*kidney_disease(t)+110*(1-kidney_disease(t)))/110)^c3/(v1*BSA_normalised(weight(t), height(t))^v2)) * s_LEV,
            D(S_LEV) ~ s_LEV, 
            obs_LEV ~ Normal(s_LEV, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for the PK behavior of Levetiracetam, when absorption is not modelled
@with_kw struct PKLEVNoAbsorption{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((c1 = 72.0, c2 = 1.0, c3 = 1.0, v1 = 40.0, v2 = 1.0, σ=0.5)) 
    cov::T2 = (:weight, :height, :kidney_disease, :CLCr) 
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
    @parameters (CLCr::type_use)(..) [tunable=false]
    @variables s_LEV_unnormalised(t) = 0.0 # depot compartment, here unnormalised
    @variables s_LEV(t)  # internal/central compartment
    @variables S_LEV(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_LEV(t)
    
    eqs = [s_LEV ~ s_LEV_unnormalised/(v1*BSA_normalised(weight(t), height(t))^v2), 
            D(s_LEV_unnormalised) ~ - (c1*(weight(t)/70)^c2*((CLCr(t)+50*kidney_disease(t)+110*(1-kidney_disease(t)))/110)^c3/(v1*BSA_normalised(weight(t), height(t))^v2)) * s_LEV_unnormalised,
            D(S_LEV) ~ s_LEV, 
            obs_LEV ~ Normal(s_LEV, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for the PK behavior of Carbamazepine
@with_kw struct PKCBZ{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs = 36.0, c1 = 72.0, c2 = 1.0, c3 = 0.0, v1 = 1.0, σ = 0.1)) 
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
    θ::T=ComponentArray((k_abs = 72.0, c1 = 7.5, c2 = 0.1, c3 = 1.0, v1 = 0.5, σ = 0.1)) 
    cov::T2 = (:gender, :weight) 
    set_daily_doses::T3 = ((drug_param = :d_VPA_daily, drug_var = :d_VPA, autoinduction = false, ind_param = :none),) 
    #parameter to update and corresponding state name for updates, bool if autoinduction, name of autoinduction parameter (not present here, just for sake of completeness)
    keys::T4 = (d = SA[:d_VPA], s = SA[:s_VPA], S = SA[:S_VPA], obs = SA[(:obs_VPA, :s_VPA)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKVPA) 
    #k_abs constant, V=v1*weight
    #-CL = c1*(dose/1000)^c2*c3^gender(1 for female, 0 for male)
    #take make with 100 around dose to avoid zero clearance when stop taking VPA
    #for multidrugmodel CBZ (and PB,PHT,CLB) dependence in clearance
    #Absorption rate k_abs/V, elimination CL/V
    θ = mod.θ
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters k_abs=θ.k_abs c1 = θ.c1 c2 = θ.c2 c3 = θ.c3 v1 = θ.v1 σ=θ.σ #normal system parameters
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
            D(s_VPA) ~ k_abs/(v1*weight(t)) * d_VPA - (c1*(max(100,d_VPA_daily)/1000)^c2*c3^gender(t))/(v1*weight(t)) * s_VPA,
            D(S_VPA) ~ s_VPA,
            obs_VPA ~ Normal(s_VPA, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for the PK behavior of Lamotrigine
@with_kw struct PKLTG{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs = 72.0, c1 = 72.0, c2 = 0.0, c3 = 0.0, c4 = 0.0, v1 = 1.0, σ=0.5)) 
    cov::T2 = (:weight, :kidney_disease, :CLCr, :smoking) 
    set_daily_doses::T3 = ()
    keys::T4 = (d = SA[:d_LTG], s = SA[:s_LTG], S = SA[:S_LTG], obs = SA[(:obs_LTG, :s_LTG)]) #for observations also records corresponding internal state
end

function create_ode_system(mod::PKLTG) 
    #V = v1*TBW
    #CL = c1*(Weight normalised)^c2*(1-c3*(CLCr -110))*(1+c4*smoking)
    #if only kindey_disease yes/no known sets CLCr to 50, if both known only CLCr is used (ensured in create_problem)
    #Absorption rate k_abs/V, elimination CL/V
    θ = mod.θ
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    #Define model, @mtkmodel doesnt agree with callable parameters
    @parameters k_abs=θ.k_abs c1=θ.c1 c2=θ.c2 c3=θ.c3 c4=θ.c4 v1=θ.v1 σ=θ.σ #normal system parameters
    #callable parameters for covariates
    @parameters (weight::type_use)(..) [tunable=false] 
    @parameters (smoking::type_use)(..) [tunable=false] 
    @parameters (kidney_disease::type_use)(..) [tunable=false]
    @parameters (CLCr::type_use)(..) [tunable=false]
    @variables d_LTG(t) = 0.0  # depot compartment - no drug at beginning
    @variables s_LTG(t) = 0.0  # internal/central compartment
    @variables S_LTG(t) = 0.0  #Integral over dose, always compute since don't know what seizure model requires
    @variables obs_LTG(t)
    #d_LTG is not concentration but dose, so rate there not normalised by volume
    eqs = [D(d_LTG) ~ -k_abs * d_LTG,
            D(s_LTG) ~ (k_abs/(v1*(weight(t)))) * d_LTG - (c1*(weight(t)/70)^c2*(1+c3*((CLCr(t)+50*kidney_disease(t)+110*(1-kidney_disease(t)))/110-1))*(1+c4*smoking(t))/(v1*(weight(t)))) * s_LTG,
            D(S_LTG) ~ s_LTG, 
            obs_LTG ~ Normal(s_LTG, σ)]
    
    @mtkcompile internal_model = System(eqs, t)

    return internal_model
end

#A model for the PK behavior of all 4 above drugs at once
@with_kw struct PKBigFour{T<:ComponentArray, T2<:Tuple, T3<:Tuple, T4<:NamedTuple} <: PKModelNonrandom
    θ::T=ComponentArray((k_abs_LTG = 72.0, c1_LTG = 72.0, c2_LTG = 0.0, c3_LTG = 0.0, c4_LTG = 0.0, c_Inh_LTG = 1.0, c_Ind_LTG = 1.0, v1_LTG = 1.0, σ_LTG=0.5,
                        k_abs_VPA = 72.0, c1_VPA = 7.5, c2_VPA = 0.1, c3_VPA = 1.0, c_Ind_VPA = 1.0, v1_VPA = 0.5, σ_VPA = 0.1,
                        k_abs_CBZ = 36.0, c1_CBZ = 72.0, c2_CBZ = 1.0, c3_CBZ = 0.0, v1_CBZ = 1.0, σ_CBZ = 0.1,
                        k_abs_LEV = 72.0, c1_LEV = 72.0, c2_LEV = 1.0, c3_LEV = 1.0, c_Inh_LEV = 1.0, c_Ind_LEV = 1.0, v1_LEV = 40.0, v2_LEV = 1.0, σ_LEV=0.5)) 
    cov::T2 = (:weight, :height, :kidney_disease, :CLCr, :smoking, :prev_CBZ, :gender) 
    set_daily_doses::T3 = ((drug_param = :d_VPA_daily, drug_var = :d_VPA, autoinduction = false, ind_param = :none), (drug_param = :d_CBZ_daily, drug_var = :d_CBZ, autoinduction = true, ind_param = :ind_CBZ))
    keys::T4 = (d = SA[:d_LTG, :d_LEV, :d_CBZ, :d_VPA], s = SA[:s_LTG, :s_LEV, :s_CBZ, :s_VPA], S = SA[:S_LTG, :S_LEV, :S_CBZ, :S_VPA], obs = SA[(:obs_LTG, :s_LTG), (:obs_LEV, :s_LEV), (:obs_CBZ, :s_CBZ), (:obs_VPA, :s_VPA)]) #for observations also records corresponding internal state
end

#CL LEV: *1.22 if coadministered with CBZ (/PHT/PB/PD) Toublanc et al. (2008), *0.812 for VPA Pigeolet et al. (2007)
#CL CBZ: only includes coadministration with PHT, not considered here
#CL VPA: *1.22 if coadministered with CBZ, multipliers for PB, PHT, CLB not of interest here
#CL LTG: *(1-0.579) for VPA or sertraline, *(1+0.546) for CBZ, PHT, PB
#-> need set doses for VPA, CBZ to check if >0, already needed those for dose dependence anyway
function create_ode_system(mod::PKBigFour) 
    θ = mod.θ
    BSA_normalised(weight, height) = sqrt(weight*height/3600)/1.68
    interpolator = ConstantInterpolation([0.0, 10.0], [1.1, 5.5])
    type_use = typeof(interpolator).name.wrapper
    @parameters k_abs_LTG=θ.k_abs_LTG c1_LTG=θ.c1_LTG c2_LTG=θ.c2_LTG c3_LTG=θ.c3_LTG c4_LTG=θ.c4_LTG c_Inh_LTG = θ.c_Inh_LTG c_Ind_LTG = θ.c_Ind_LTG v1_LTG=θ.v1_LTG σ_LTG=θ.σ_LTG 
    @parameters k_abs_VPA=θ.k_abs_VPA c1_VPA = θ.c1_VPA c2_VPA = θ.c2_VPA c3_VPA = θ.c3_VPA c_Ind_VPA = θ.c_Ind_VPA v1_VPA = θ.v1_VPA σ_VPA=θ.σ_VPA
    @parameters d_VPA_daily = 0.0 [tunable=false]
    @parameters k_abs_CBZ=θ.k_abs_CBZ c1_CBZ = θ.c1_CBZ c2_CBZ = θ.c2_CBZ c3_CBZ = θ.c3_CBZ v1_CBZ = θ.v1_CBZ σ_CBZ=θ.σ_CBZ 
    @parameters d_CBZ_daily = 0.0 [tunable=false]
    @parameters k_abs_LEV=θ.k_abs_LEV c1_LEV=θ.c1_LEV c2_LEV=θ.c2_LEV c3_LEV=θ.c3_LEV c_Inh_LEV = θ.c_Inh_LEV c_Ind_LEV = θ.c_Ind_LEV v1_LEV=θ.v1_LEV v2_LEV=θ.v2_LEV σ_LEV=θ.σ_LEV
    #callable parameters for covariates
    @parameters (weight::type_use)(..) [tunable=false] 
    @parameters (height::type_use)(..) [tunable=false] 
    @parameters (smoking::type_use)(..) [tunable=false] 
    @parameters (kidney_disease::type_use)(..) [tunable=false]
    @parameters (CLCr::type_use)(..) [tunable=false]
    @parameters (gender::type_use)(..) [tunable=false]
    @parameters (prev_CBZ::type_use)(..) [tunable=false] 
    @parameters ind_CBZ = 14*prev_CBZ(0.0) [tunable = false]
    @variables d_LTG(t) = 0.0  
    @variables d_VPA(t) = 0.0
    @variables d_CBZ(t) = 0.0
    @variables d_LEV(t) = 0.0
    @variables s_LTG(t) = 0.0  
    @variables s_VPA(t) = 0.0
    @variables s_CBZ(t) = 0.0
    @variables s_LEV(t) = 0.0
    @variables S_LTG(t) = 0.0  
    @variables S_VPA(t) = 0.0
    @variables S_CBZ(t) = 0.0
    @variables S_LEV(t) = 0.0
    @variables obs_LTG(t)
    @variables obs_VPA(t) 
    @variables obs_CBZ(t) 
    @variables obs_LEV(t) 
    
    eqs = [D(d_LTG) ~ -k_abs_LTG * d_LTG,
            D(d_VPA) ~ -k_abs_VPA * d_VPA,
            D(d_CBZ) ~ -k_abs_CBZ * d_CBZ,
            D(d_LEV) ~ -k_abs_LEV * d_LEV,
            D(s_LTG) ~ (k_abs_LTG/(v1_LTG*(weight(t)))) * d_LTG - (c1_LTG*(weight(t)/70)^c2_LTG*(1+c3_LTG*((CLCr(t)+50*kidney_disease(t)+110*(1-kidney_disease(t)))/110-1))*(1+c4_LTG*smoking(t))*c_Ind_LTG^(d_CBZ_daily>0)*c_Inh_LTG^(d_VPA_daily>0)/(v1_LTG*(weight(t)))) * s_LTG,
            D(s_VPA) ~ k_abs_VPA/(v1_VPA*weight(t)) * d_VPA - (c1_VPA*(max(100,d_VPA_daily)/1000)^c2_VPA*c3_VPA^gender(t)*c_Ind_VPA^(d_CBZ_daily>0))/(v1_VPA*weight(t)) * s_VPA,
            D(s_CBZ) ~ k_abs_CBZ/(v1_CBZ*weight(t)) * d_CBZ - (c1_CBZ*(c2_CBZ^(ind_CBZ>=14))+c3_CBZ*log(max(d_CBZ_daily/400, 1/4)))/(v1_CBZ*weight(t)) * s_CBZ,
            D(s_LEV) ~ (k_abs_LEV/(v1_LEV*BSA_normalised(weight(t), height(t))^v2_LEV)) * d_LEV - (c1_LEV*(weight(t)/70)^c2_LEV*((CLCr(t)+50*kidney_disease(t)+110*(1-kidney_disease(t)))/110)^c3_LEV*c_Ind_LEV^(d_CBZ_daily>0)*c_Inh_LEV^(d_VPA_daily>0)/(v1_LEV*BSA_normalised(weight(t), height(t))^v2_LEV)) * s_LEV,
            D(S_LTG) ~ s_LTG, 
            D(S_VPA) ~ s_VPA,
            D(S_CBZ) ~ s_CBZ,
            D(S_LEV) ~ s_LEV,
            obs_LTG ~ Normal(s_LTG, σ_LTG),
            obs_VPA ~ Normal(s_VPA, σ_VPA),
            obs_CBZ ~ Normal(s_CBZ, σ_CBZ),
            obs_LEV ~ Normal(s_LEV, σ_LEV)]

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

function get_covariate_type(typ)
    if typ <: SymbolicUtils.FnType
        return get_fntype(typ)
    else
        return typ
    end
end

function get_fntype(t::Type{SymbolicUtils.FnType{X, Y, Z}}) where {X,Y,Z} 
    return Z
end

function make_type(x, t::Type)
    if x isa t
        return x
    else 
        return t(x)
    end
end

#Problem creation with covariates and callbacks
function create_problem(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat = 10.0)
    
    ode_system = create_ode_system(mod)
    #check if just one of kidney_disease and creatinine clearance passed, if then set other to 0
    if issubset(mod.cov, keys(covariates))
        covariates = NamedTuple{mod.cov}(covariates)
        #if both CLCr and kidney_disease given set kidney_disease=0 for ODE system
        if issubset(([:kidney_disease], [:CLCr]), mod.cov)
            merge(NamedTuple{Tuple(setdiff(mod.cov,(:kidney_disease,)))}(covariates), (kidney_disease=0,))
        end
        covariates = NamedTuple{mod.cov}(covariates)
    elseif setdiff(mod.cov, keys(covariates)) in ([:kidney_disease], [:CLCr])
        covariates = merge(NamedTuple{Tuple(intersect(mod.cov,keys(covariates)))}(covariates), NamedTuple{Tuple(setdiff(mod.cov, keys(covariates)))}([0]))
    else
        error("Covariates of data does not match covariates required by PK model")
    end

    #Create Callbacks for doses, autoinduction and other potential dose related behavior
    names = get_keys_PK(mod)
    callback_set = create_dosing_callbacks(dosing, ode_system, names = names, set_daily_doses = mod.set_daily_doses)

    #get type info for covariates
    param_info = [(name = info.name, type = get_covariate_type(info.type)) for info in ModelingToolkit.dump_parameters(ode_system) if info.name in mod.cov]
    #interpolate covariates from given data or try typecasting, if unsuccessful throw error
    try
        covariate_interpolation = Dict((isa(covariates[info.name],Number) && info.type<:DataInterpolations.AbstractInterpolation) ? (info.name => info.type([covariates[info.name], covariates[info.name]], [0.0, endpoint])) : (info.name => make_type(covariates[info.name], info.type)) for info in param_info)
        #Create ODE problem with callbacks
        problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, covariate_interpolation, (0.0, endpoint), callback = callback_set)
    
        return problem
    catch e
        println(e)
        error("Passed type of covariate is not supported")
    end
end

#same function for ode_system already given, given a person instead of dosing and covariates
function create_problem(mod::PKModelNonrandom, ode_system::ODESystem; person::Person, endpoint::AbstractFloat = 10.0)
    dosing = person.dosing
    #check if just one of kidney_disease and creatinine clearance passed, if then set other to 0
    if issubset(mod.cov, keys(person.covariates))
        covariates = NamedTuple{mod.cov}(person.covariates)
        #if both CLCr and kidney_disease given set kidney_disease=0 for ODE system
        if issubset(([:kidney_disease], [:CLCr]), mod.cov)
            merge(NamedTuple{Tuple(setdiff(mod.cov,(:kidney_disease,)))}(covariates), (kidney_disease=0,))
        end
    elseif setdiff(mod.cov, keys(person.covariates)) in ([:kidney_disease], [:CLCr])
        covariates = merge(NamedTuple{Tuple(intersect(mod.cov,keys(person.covariates)))}(person.covariates), NamedTuple{Tuple(setdiff(mod.cov, keys(person.covariates)))}([0]))
    else
        error("Covariates of data does not match covariates required by PK model")
    end

    #Create Callbacks for doses, autoinduction and other potential dose related behavior
    names = get_keys_PK(mod)
    callback_set = create_dosing_callbacks(dosing, ode_system, names = names, set_daily_doses = mod.set_daily_doses)

    #get type info for covariates
    param_info = [(name = info.name, type = get_covariate_type(info.type)) for info in ModelingToolkit.dump_parameters(ode_system) if info.name in mod.cov]
    #interpolate covariates from given data or try typecasting, if unsuccessful throw error
    try
        covariate_interpolation = Dict((isa(covariates[info.name],Number) && info.type<:DataInterpolations.AbstractInterpolation) ? (info.name => info.type([covariates[info.name], covariates[info.name]], [0.0, endpoint])) : (info.name => make_type(covariates[info.name], info.type)) for info in param_info)
        #Create ODE problem with callbacks
        problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, covariate_interpolation, (0.0, endpoint), callback = callback_set)
    
        return problem
    catch e
        println(e)
        error("Passed type of covariate is not supported")
    end
end

#solve ODE without system given
function solve_ODE(mod::PKModelNonrandom; dosing::AbstractVector, covariates::NamedTuple=NamedTuple(), endpoint::AbstractFloat=10.0, start::Union{Tuple, Nothing} = nothing, options = (AutoTsit5(Rosenbrock23()),))
    prob = create_problem(mod, dosing=dosing, covariates=covariates, endpoint=endpoint)
    if !isnothing(start)
        new_tspan = (start[1], endpoint)
        if endpoint < start[1]
            error("Given endpoint for ODE is smaller than specified start")
        end
        new_u0 = start[2]
        prob = remake(prob, tspan=new_tspan, u0=new_u0)
    end
    if !(endpoint > 0)
        @warn "Endpoint for ODE solve is 0.0"
    end
    sol = solve(prob,options...; callback = PositiveDomain())
    return sol
end

#solve ODE given system
function solve_ODE(mod::PKModelNonrandom, sys::ODESystem; person::Person, endpoint::AbstractFloat=10.0, start::Union{Tuple, Nothing} = nothing, options = (AutoTsit5(Rosenbrock23()),))
    prob = create_problem(mod, sys, person=person, endpoint=endpoint)
    if !isnothing(start)
        new_tspan = (start[1], endpoint)
        if endpoint < start[1]
            error("Given endpoint for ODE is smaller than specified start")
        end
        new_u0 = start[2]
        prob = remake(prob, tspan=new_tspan, u0=new_u0)
    end
    if !(endpoint > 0)
        @warn "Endpoint for ODE solve is 0.0"
    end
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
function generate_measurements!(mod::PKModel, person::Person; timepoints::AbstractVector, endpoint::AbstractFloat = timepoints[end], start::Union{Tuple, Nothing} = nothing, options = (AutoTsit5(Rosenbrock23()),))
    cov = NamedTuple{mod.cov}(person.covariates)
    sol = solve_ODE(mod, dosing = person.dosing, covariates = cov, endpoint = endpoint, start = start, options = options)
    if !(SciMLBase.successful_retcode(sol))
        @warn "Unsuccessful ODE solve in data generation, you might want to adjust model parameters"
    end
    names = get_keys_PK(mod)
    measurements = [(timepoint = timepoint, measurement = rand(sol(timepoint, idxs = obs[1])), state = obs) for timepoint in timepoints for obs in names.obs]
    append!(person.measurements, measurements)
    return sol
end

#same function but given ODE system
function generate_measurements!(mod::PKModel, sys::ODESystem, person::Person; timepoints::AbstractVector, endpoint::AbstractFloat = timepoints[end], start::Union{Tuple, Nothing} = nothing, options = (AutoTsit5(Rosenbrock23()),))
    sol = solve_ODE(mod, sys, person=person, endpoint = endpoint, start = start, options = options)
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
function plot_fit(mod::PKModel, data::Tuple; sols_true::Union{AbstractVector, Nothing} = nothing, sols_estimated::Union{AbstractVector, Nothing} = nothing, sols_modified::Union{AbstractVector, Nothing} = nothing,
    individuals::AbstractVector = [1],  time::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int, Nothing} = nothing, display_plot::Bool = true)
    if !isnothing(sols_true)
        endpoint = sols_true[1].t[end]
    elseif !isnothing(sols_estimated)
        endpoint = sols_estimated[1].t[end]
    else
        endpoint = max(data[1].measurements[end].timepoint, data[1].seizure_counts[end].time[2])
    end
    if time isa Number
        time = (0.0,Float64(time))
    elseif isnothing(time)
        time = (0,endpoint)
    end
    if time[1]<0 || time[2] > endpoint || time[1] > time[2]
        error("Incorrectly defined time window for PK plotting")
    end
    output = Plots.Plot[]
    if !isnothing(sols_true) && any(.!(SciMLBase.successful_retcode.(sols_true)))
        @warn "Unsuccessful ODE solve in true parameters, true parameters will be ignored for plotting"
        sols_true = nothing
    end
    if !isnothing(sols_estimated) && any(.!(SciMLBase.successful_retcode.(sols_estimated)))
        @warn "Unsuccessful ODE solve in estimated parameters, true parameters will be ignored for plotting"
        sols_estimated = nothing
    end
    if !isnothing(sols_modified) && any(.!(SciMLBase.successful_retcode.(sols_modified)))
        @warn "Unsuccessful ODE solve in true parameters modified with random effects, modified parameters will be ignored for plotting"
        sols_modified = nothing
    end
    sols = sols_true
    sols2 = sols_estimated
    #Plot PK behavior (for each drug)
    names = get_keys_PK(mod)
    #Iterate over indices for which to plot
    for i in individuals
        #Iterate over drugs 
        for s in names.s
            pl = plot(xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s) for person $(i)", tspan = time)
            #true plot if param specified
            if !isnothing(sols)
                plot!(sols[i], idxs = s, label="Concentration $(s)", tspan = time)
            end
            if !isnothing(sols_modified)
                plot!(sols_modified[i], idxs = s, label="Concentration $(s) with random effects", linecolor = :green, tspan = time)
            end
            #add scattered measurements
            x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
            y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
            plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "", tspan = time)
            #add estimate plot if specified
            if !isnothing(sols2)
                plot!(sols2[i], idxs = s, label="Estimated concentration $(s)", linecolor = :red, tspan = time)
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
        pl = plot(xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s)", tspan = time)
        if !isnothing(sols)
            plot!(sols[1], idxs = s, label="Concentration $(s)", linecolor = :blue, tspan = time)
            for i in 2:Population_size
                plot!(sols[i], idxs = s, label = "", linecolor = :blue, tspan = time)
            end
        end
        if !isnothing(sols_modified)
            plot!(sols_modified[1], idxs = s, label="Concentration $(s) with random effects", linecolor = :green, tspan = time)
            for i in 2:Population_size
                plot!(sols_modified[i], idxs = s, label = "", linecolor = :green, tspan = time)
            end
        end
        for i in 1:Population_size
            x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
            y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
            plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "", tspan = time)
        end
        #add estimate plots
        if !isnothing(sols2)
            plot!(sols2[1], idxs = s, label="Estimated concentration $(s)", linecolor = :red, tspan = time)
            for i in 2:Population_size
                plot!(sols2[i], idxs = s, label = "", linecolor = :red, tspan = time)
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
    endpoint::Union{Nothing, AbstractFloat} = nothing, time::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int, Nothing} = nothing, display_plot::Bool = true, options = (AutoTsit5(Rosenbrock23()),))
    
    if isnothing(endpoint)
        measurements_ends = Tuple(person.measurements[end].timepoint for person in data)
        seizure_ends = Tuple(person.seizure_counts[end].time[2] for person in data)
        endpoint = max(measurements_ends...,seizure_ends...)
    end
    if isnothing(time)
        time = (0.0, endpoint)
    end
    if !isnothing(true_param)
        sols = [solve_PK(mod, true_param, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
        if length(data)>0 && !isempty(data[1].random_effects)
            person_param = [deepcopy(true_param) for person in data]
            for i in eachindex(data)
                for mod in data[i].random_effects
                    if mod[1] <= length(true_param)
                        person_param[i][mod[1]] += mod[2]
                    end
                end
            end
            sols_mod = [solve_PK(mod, person_param[i], data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
        else
            sols_mod = nothing
        end
    else
        sols = nothing
        sols_mod = nothing
    end
    if !isnothing(estimate_param)
        sols2 = [solve_PK(mod, estimate_param, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
    else 
        sols2 = nothing
    end
    output = plot_fit(mod, data, sols_true = sols, sols_estimated = sols2, sols_modified = sols_mod, individuals = individuals, time = time, display_plot = display_plot)
    return output
end