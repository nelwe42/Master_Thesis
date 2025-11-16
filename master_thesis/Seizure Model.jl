using ModelingToolkit
using Distributions
using Random
using Parameters
using ComponentArrays

#set seed
Random.seed!(42)

#Overtype of Seizure Models that will go into full model
abstract type SeizureModel end

#To distinguish if possibly decide to make time continuous models later
abstract type SeizureModelDiscrete <: SeizureModel end

#later for checking if random effects need to be handled in inference
abstract type SeizureModelNonrandom <: SeizureModelDiscrete end
#For this need some getter for which are random effects?

#Every model specification should have: ComponentArray of parameters, list of keys of required covariates
#Every model should have function returning intensity

#Within discrete/continuous and (non)random returning seizure probability, likelihoods and 
#generating data can be handled once

#1)Specific model instances with their intensities

@with_kw struct SeizureBasic{T<:ComponentArray, T2<:Tuple} <: SeizureModelNonrandom
    θ::T=ComponentArray((a = 0.0, b = 0.0)) #a base rate, b coefficient of drug (how to handle more later?)
    cov::T2 = () #no covariates required
end

#function intensity(mod::Seizure_Basic, n::Int; here take PK system/solution object)
function intensity(m::SeizureBasic, sol, n::AbstractFloat; covariates = nothing, θ::ComponentArray = m.θ)
    intensity = θ.a
    intensity = intensity + θ.b*(sol(n+1, idxs = :S)-sol(n,idxs = :S))
    #on day n natural number beginning with 0 are exposed to drug from time n to n+1
    #day 0 ist interval (0,1], day named after first number
    return intensity
end

#2) Implement Seizure Probabilities, Likelihoods and Data Generators for discrete, nonrandom

#k_n number of seizures on day n
function Seizure_prob_day(m::SeizureModelNonrandom, sol, n::AbstractFloat, k_n::Int64; covariates = nothing, θ::ComponentArray = m.θ)
    lambda = intensity(m,sol,n, covariates = covariates, θ = θ)
    return ((lambda^k_n)/factorial(k_n))*exp(-lambda)
end

function Seizure_prob(m::SeizureModelNonrandom, sol, person::Person; θ::ComponentArray = m.θ)
    prob = 1
    for i in eachindex(person.seizure_counts) 
        time = person.seizure_counts[i].time #get timepoint out of named tuple
        count = person.seizure_counts[i].count #get count out of tuple
        cov = NamedTuple{m.cov}(person.covariates) #create cov via person covariates and keys
        prob = prob * Seizure_prob_day(m, sol, time, count, covariates = cov, θ=θ)
    end
    return prob
end

function get_seizure_loglikelihood(θ::ComponentArray, m::SeizureModel, sol, person::Person)
    return log(Seizure_prob(m, sol, person, θ=θ))
end

#3) Implement generation of seizures for discrete, nonrandom models

#generates and appends seizures to person for given number of days start
function generate_seizures!(m::SeizureModelNonrandom, sol, person::Person; start::AbstractFloat = 0, day_number::AbstractFloat = 10.0)
    if day_number >=1
    cov = NamedTuple{m.cov}(person.covariates)
    new_seizures = [(time = n, count = rand(Poisson(intensity(m, sol, n, covariates=cov)))) for n in start:(start+day_number-1)]
    append!(person.seizure_counts, new_seizures)
    end
end