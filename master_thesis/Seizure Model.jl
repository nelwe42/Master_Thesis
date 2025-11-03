using ModelingToolkit
using Distributions
using Random
using Parameters

#set seed
Random.seed!(42)

#Overtype of Seizure Models that will go into full model
abstract type Seizure_Model end

#To distinguish if possibly decide to make time continuous models later
abstract type Seizure_Model_discrete <: Seizure_Model end

#later for checking if random effects need to be handled in inference
abstract type Seizure_Model_nonrandom <: Seizure_Model_discrete end
#For this need some getter for which are random effects?

#Every model specification should have: Named tuple of parameters, list of keys of required covariates
#Every model should have function returning intensity

#Within discrete/continuous returning seizure probability, likelihoods and generating data can be handled once

#1)Specific model instances with their intensities

@with_kw struct Seizure_Basic{T<:NamedTuple, T2<:Tuple} <: Seizure_Model_nonrandom
    θ::T=(a = 0.0, b = 0.0) #a base rate, b coefficient of drug (how to handle more later?)
    cov::T2 = () #no covariates required
end

#function intensity(mod::Seizure_Basic, n::Int; here take PK system/solution object)


#Old stuff
#Theta_Seizure = [a_0, list of b_d], i(t) = vector of i_d(t) solutions of ODE problem internal, n = day
function intensity(Theta_Seizure, i, n)
    intense = Theta_Seizure[1]
    for d in 1:length(i)
        integral = 0 #solve integral over [n,n+1) of i_d(t) here
        intense = intense + Theta_Seizure[d+1] * integral
    end
    return intense
end

#k_n number of seizures on day n
function Seizure_prob_day(Theta_Seizure, i, n, k_n)
    lambda = intensity(Theta_Seizure, i, n)
    return (lambda^k_n/factorial(k_n))*exp(-lambda)
end

#N vector of days, k vector of number of seizures on days
function Seizure_prob(Theta_Seizure, i, N, k)
    prob = 1
    for n in N #is this allowed syntax in Julia?
        k_n = k[n]
        prob = prob * Seizure_prob_day(Theta_Seizure, i, n, k_n)
    end
    return prob
end

print("Done")