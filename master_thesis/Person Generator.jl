using Random
using Distributions

@with_kw struct Person{T<:NamedTuple}
    covariates::T = NamedTuple() #named tuple of covariates
    dosing<:AbstractVector = [] #vector of NamedTuples of the form t, dose, state
    seizure_counts<:AbstractVector = [] #vector of NamedTuples of the form time, count
    measurements<:AbstractVector = [] #vector of NamedTuples of the form timepoint, measurement
    #later make attribute with individual values of random effects
end

abstract type PersonGenerator end

struct BasicPersonGenerator <: PersonGenerator end

function generate_population(m::BasicPersonGenerator, n::Int = 10)
    population = [Person() for i in 1:n]
    return population
end

