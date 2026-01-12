using Random
using Distributions
using Parameters

#set seed
Random.seed!(42)

abstract type DoseGenerator end

@with_kw struct BasicDoses{T<:AbstractFloat, T2<:Int} <: DoseGenerator 
    default_dose::T = 5.0
    times_per_day::T2 = 1
end

#assign same dose to all for next timeframe days, wo_treatment gives first dosetime, afterwards dose every time unit (day)
function assign_dose!(m::BasicDoses, person::Person; names::NamedTuple, timeframe::AbstractFloat = 10.0, wo_treatment::AbstractFloat = 0.0)
    dose = m.default_dose
    times = m.times_per_day
    if isempty(person.dosing)
        last_dosetime = -1
    else
        last_dosetime = person.dosing[end].t
    end
    #for first wo_treatment time no treatment, don't need to add zero callbacks to dosing
    no_dose = min(wo_treatment,timeframe)
    last_dosetime += no_dose
    #append for each dose time and drug taken
    next_doses = [(t = i+1 + j/times, dose = dose, state = d) for i in last_dosetime:(last_dosetime+timeframe-(no_dose+1)) for j in 0:(times-1) for d in names.d]
    append!(person.dosing,next_doses)
end

struct DoseLeitlinie <: DoseGenerator end
#note: epileptic syndromes not considered


#functions: assign_dose!(person/people), change_dose(person) to decide if change, pick_drug, 
#pick_secondary_drug, pick_dose, potentially sixth function for probability of changing due to side effects

function pick_drug(person::Person)
    failed_drugs = [] #drugs in previous doses

    if person.covariates.sex == "f" && something #do some conditions for pregnancy possible/wanted here
        if !("Lamotrigin" in failed_drugs)
            if "Levetiracetam" in failed_drugs
                primary_drug = "Lamotrigin"
            else 
                #pick between Lamotrigin and Levetiracetam with certain probability
            end
        elseif !("Levetiracetam" in failed_drugs)
            primary_drug = "Levetiracetam"
        elseif person.covariates.type == "focal"
            if !("Oxcarbazepin" in failed_drugs)
                primary_drug = "Oxcarbazepin"
            elseif !("Eslicarbazepinacetat" in failed_drugs)
                if "Lacosamid" in failed_drugs
                    primary_drug = "Eslicarbazepinacetat"
                else 
                    #pick between Lacosamid and Eslicarbazepinacetat with certain probability
                end
            elseif !("Lacosamid" in failed_drugs)
                primary_drug = "Lacosamid"
            elseif !("Zonisamid" in failed_drugs)
                primary_drug = "Zonisamid"
            else
                primary_drug = "drug_resistant"
            end
        elseif !("Valproate" in failed_drugs) #Valproate as last resort for generalised/unclassified
            primary_drug = "Valproate"
        else
            primary_drug = "drug_resistant"
        end
    elseif person.covariates.age == "65+" && person.covariates.type == "focal"
        if !("Lamotrigin" in failed_drugs)
            primary_drug = "Lamotrigin"
        elseif !("Levetiracetam" in failed_drugs)
            if !("Gabapentin" in failed_drugs)
                if !("Lacosamid" in failed_drugs)
                    #pick between Gabapentin, Lacosamid, Levetiracetam with certain probability
                else
                    #pick between Gabapentin and Levetiracetam with certain probability
                end
            else
                if !("Lacosamid" in failed_drugs)
                    #pick between Lacosamid and Levetiracetam with certain probability
                else
                    primary_drug = "Levetiracetam"
                end
            end
        elseif !("Gabapentin" in failed_drugs)
            if !("Lacosamid" in failed_drugs)
                #pick between Lacosamid and Gabapentin with certain probability
            else
                primary_drug = "Gabapentin"
            end
        elseif !("Lacosamid" in failed_drugs)
            primary_drug = "Lacosamid"
        elseif !("Eslicarbazepinacetat" in failed_drugs)
            primary_drug = "Eslicarbazepinacetat"
        elseif !("Zonisamid" in failed_drugs)
            primary_drug = "Zonisamid"
        else
            primary_drug = "drug_resistant"
        #7 more drugs not recommended to start in old age, potentially add later
        end
    elseif person.covariates.type == "focal"
        #Mono: Carbamazepine, Eslicarbazepinacetat, Gabapentin, Lacosamid, Lamotrigin, Levetiracetam, 
        #Oxcarbazepin, Phenobarbital, Phenytoin, Primidon, Topiramat, Valproate (and Zonisamid as last resort)
        if !("Lamotrigin" in failed_drugs)
            primary_drug = "Lamotrigin"
        elseif !("Lacosamid" in failed_drugs)
            if "Levetiracetam" in failed_drugs
                primary_drug = "Lacosamid"
            else 
                #pick between Lacosamid and Levetiracetam with certain probability
            end
        elseif !("Levetiracetam" in failed_drugs)
            primary_drug = "Levetiracetam"
        elseif !("Eslicarbazepinacetat" in failed_drugs)
            if "Oxcarbazepin" in failed_drugs
                primary_drug = "Eslicarbazepinacetat"
            else 
                #pick between Eslicarbazepinacetat and Oxcarbazepin with certain probability
            end
        elseif !("Oxcarbazepin" in failed_drugs)
            primary_drug = "Oxcarbazepin"
        elseif !("Zonisamid" in failed_drugs) 
            primary_drug = "Zonisamid"
        else
            primary_drug = "drug_resistant"
        #8 other drugs not recommended for initial monotherapies, potentially add later
        end
    elseif person.covariates.type == "generalised"
        #Mono: Ethosuximid (if only absences), Lamotrigin, Mesuximid (if only absences as last resort),
        #Phenobarbital, Primidon, Topiramat, Valproate
        if person.covariates.seizure_type == "absence" && !("Ethosuximid" in failed_drugs)
            primary_drug = "Ethosuximid"
        elseif person.covariates.seizure_type == "tonic-clonic" && !("Valproate" in failed_drugs)
            primary_drug = "Valproate"
        elseif !("Lamotrigin" in failed_drugs)
            if "Levetiracetam" in failed_drugs
                primary_drug = "Lamotrigin"
            else 
                #pick between Lamotrigin and Levetiracetam with certain probability
            end
        elseif !("Levetiracetam" in failed_drugs)
            primary_drug = "Levetiracetam"
        else
            primary_drug = "drug_resistant"
        #4 more drugs not recommended for initial monotherapies, potentially add later
        end
    else #unclassified epilepsy
        if !("Levetiracetam" in failed_drugs)
            if !("Valproate" in failed_drugs)
                if !("Lamotrigin" in failed_drugs)
                    #pick between Valproate, Lamotrigin, Levetiracetam with certain probability
                else
                    #pick between Valproate and Levetiracetam with certain probability
                end
            else
                if !("Lamotrigin" in failed_drugs)
                    #pick between Lamotrigin and Levetiracetam with certain probability
                else
                    primary_drug = "Levetiracetam"
                end
            end
        elseif !("Valproate" in failed_drugs)
            if !("Lamotrigin" in failed_drugs)
                #pick between Lamotrigin and Valproate with certain probability
            else
                primary_drug = "Valproate"
            end
        elseif !("Lamotrigin" in failed_drugs)
            primary_drug = "Lamotrigin"
        else
            primary_drug = "drug_resistant"
        #Bromid, Phenobarbital, Primidon not recommended for initial monotherapies, potentially add later
        end
    end

    person.dosing.add(pick_dose(person, primary_drug))
end

function pick_secondary_drug()
    if person.covariates.type == "focal"
        #As secondary: Acetazolamid, Bivaracetam, Carbamazepine, Cenobamat, Clobazam, Cloneazepam, 
        #Eslicarbazepinacetat, Gabapentin, Kaliumbromid, Lacosamid, Lamotrigin, Levetiracetam, Oxcarbazepin, 
        #Perampanel, Phenobarbital, Phenytoin, Pregabalin, Primidon, Topiramat, Valproate, Vigibatrin
        #(and Zonisamid as last resort)
    elseif person.covariates.type == "generalised"
        #As secondary: Acetazolamid, Clobazam, Cloneazepam, Kaliumbromid, Lacosamid, Lamotrigin, Levetiracetam,
        #Perampanel, Phenobarbital, Primidon, Topiramat, Valproate
    else 
    end
end

function change_dose() 
    #check if partial/full reduction
    #if full stay on dose
    #if partial flip between add drug or increase dose if possible
    #involve reduction percentage here, if pregnancy possible stay monotherapies
    #with certain probability change anyway
end