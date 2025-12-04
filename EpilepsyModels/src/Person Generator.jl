using Random
using Distributions
using Parameters

#set seed
Random.seed!(42)

@with_kw struct Person{T<:NamedTuple}
    covariates::T = NamedTuple() #named tuple of covariates
    dosing::AbstractVector = Vector{@NamedTuple{t::AbstractFloat, dose::AbstractFloat, state::Symbol}}() #vector of NamedTuples of the form t, dose, state
    seizure_counts::AbstractVector = Vector{@NamedTuple{time::AbstractFloat, count::Int64}}() #vector of NamedTuples of the form time, count
    measurements::AbstractVector = Vector{@NamedTuple{timepoint::AbstractFloat, measurement::AbstractFloat, state::Tuple{Symbol, Symbol}}}() #vector of NamedTuples of the form timepoint, measurement
    #later make attribute with individual values of random effects
end

abstract type PersonGenerator end

struct BasicPersonGenerator <: PersonGenerator end

function generate_population(m::BasicPersonGenerator, n::Int = 10)
    population = [Person() for i in 1:n]
    return population
end

struct PersonGeneratorLEV <: PersonGenerator end

function generate_population(m::PersonGeneratorLEV, n::Int = 10)
    population = Vector{Person}()
    for i in 1:n
        height = rand(Normal(170,7)) #draw height from normal distribution
        weight = rand(Normal(26.0, 3.5))*(height/100)^2 #draw first BMI, weight = BMI*height(in m)^2
        #average BMI Germany was 26 in 2021, 45,3% in 18.5 to 25, 35.9 in 25 to 30
        kidney_disease = Float64(rand(Bernoulli(0.1))) #draw kindey_disease as binomial, potentially later dependent on weight
        person = Person(covariates = (height = height, weight = weight, kidney_disease = kidney_disease))
        push!(population, person)
    end
    return population
end

