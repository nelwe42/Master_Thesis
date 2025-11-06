using Random
using Distributions

@with_kw struct Person{T<:NamedTuple, T2<:NamedTuple, T3<:NamedTuple}
    covariates::T = NamedTuple() #named tuple of covariates
    dosing<:AbstractVector{T2} = [] #vector of NamedTuples of the form t, dose, state
    seizure_counts<:AbstractVector{T3} = [] #vector of NamedTuples of the form time, count
    #later make attribute with individual values of random effects
end

abstract type Person_Generator end

struct Basic_Person_Generator <: Person_Generator end

function generate_population(m::Basic_Person_Generator, n = 10)
    population = [Person() for i in 1:n]
    return population
end

