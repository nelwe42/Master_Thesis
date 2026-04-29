using Random
using Distributions
using Parameters

abstract type DoseGenerator end

@with_kw struct BasicDoses{T<:AbstractFloat, T2<:Int} <: DoseGenerator 
    default_dose::T = 5.0
    times_per_day::T2 = 1
end

#assign same dose to all for next timeframe days, wo_treatment gives first dosetime, afterwards dose times_per_day every time unit (day)
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

@with_kw struct PolyDoses{T<:NamedTuple, T1<:Categorical, T2<:Categorical, T3<:Tuple, T4<:Tuple, T5<:AbstractFloat, T6<:Int} <: DoseGenerator 
    default_doses::T 
    distr_first::T1
    distr_second::T2
    names_first::T3
    names_second::T4
    prob_second::T5 = 0.0
    times_per_day_first::T6 = 1
    times_per_day_second::T6 = 1 
    assign_not_supported::Bool = false #controls if assign_dose! assigns drugs not given in names
end

#from probabilities given in named tuples constructs categorical and map to symbol as tuple of drug names
function PolyDoses(default_doses::NamedTuple, distr_first::NamedTuple, distr_second::NamedTuple; prob_second::AbstractFloat = 0.0, times_per_day_first::Int = 1, times_per_day_second::Int = 1, assign_not_supported::Bool = false)
    names_first = keys(distr_first)
    names_second = keys(distr_second)
    #check keys in distr_first/second have a key in default doses
    dose_names = keys(default_doses)
    if !(isempty(setdiff(names_first, dose_names))) || !(isempty(setdiff(names_second, dose_names)))
        @warn "Not all possible drug choices have an assigned default dose"
    end
    distr_one = Categorical(collect(values(distr_first)))
    distr_two = Categorical(collect(values(distr_second)))
    return PolyDoses(default_doses, distr_one, distr_two, names_first, names_second, prob_second, times_per_day_first, times_per_day_second, assign_not_supported)
end

#from PK model constructs default doses based on names, distribution for both uniform over names
function PolyDoses(m::PKModel; default_dose::AbstractFloat = 5.0, prob_second::AbstractFloat = 0.0, times_per_day_first::Int = 1, times_per_day_second::Int = 1, assign_not_supported::Bool = false)
    names = Tuple(get_keys_PK(m).d)
    N = length(names)
    default_doses = NamedTuple{names}([default_dose for i in 1:N])
    distr_first = Categorical([1/N for i in 1:N])
    return PolyDoses(default_doses, distr_first, distr_first, names, names, prob_second, times_per_day_first, times_per_day_second, assign_not_supported)
end

#picks first drug from distribution, with specified probability also second drug from that distribution, wo_treatment gives first dosetime, times per day and doses specified in model
#if second picked drug is same as first, no second drug is assigned
#when called again on person with doses already assigned, next doses are picked independently from current ones
#when a drug not contained in names is picked, the attribute assign_not_supported decides if assigned anyway
function assign_dose!(m::PolyDoses, person::Person; names::NamedTuple = (d = keys(m.default_doses),), timeframe::AbstractFloat = 10.0, wo_treatment::AbstractFloat = 0.0)
    if isempty(person.dosing)
        last_dosetime = -1
    else
        last_dosetime = person.dosing[end].t
    end
    #for first wo_treatment time no treatment, don't need to add zero callbacks to dosing
    no_dose = min(wo_treatment,timeframe)
    last_dosetime += no_dose
    #pick first drug
    drug_one = m.names_first[rand(m.distr_first)]
    #with assigned probability pick second drug
    if rand(Bernoulli(m.prob_second))
        #check second drug not the same as first
        drug_two = m.names_second[rand(m.distr_second)]
        if !(drug_one == drug_two)
            drugs = (drug_one, drug_two)
        else
            drugs = (drug_one,)
        end
    else
        drugs = (drug_one,)
    end
    doses = values(NamedTuple{drugs}(m.default_doses))
    times = (m.times_per_day_first, m.times_per_day_second)
    #check if picked drugs are supported or else is assign_not_supported is set to true
    if m.assign_not_supported
        info = Tuple((drug = drugs[i], dose = doses[i], times = times[i]) for i in eachindex(drugs))
    else
        info = Tuple((drug = drugs[i], dose = doses[i], times = times[i]) for i in eachindex(drugs) if (drugs[i] in names.d))
    end
    #append for each dose time and drug taken
    next_doses = [(t = i+1 + j/d.times, dose = d.dose, state = d.drug) for i in last_dosetime:(last_dosetime+timeframe-(no_dose+1)) for d in info for j in 0:(d.times-1)]
    append!(person.dosing,next_doses)
end

#draws dose for each person and drug from binomial distribution, 
#minimal dose (given by smallest pill) and average and maximal number number of those taken given in dose_distr
#allows to construct as binomial: minimal*(1+Binomial(maximal-1, (average-1)/(maximal-1))), then mean is average
#have to lower maximal and average by one to ensure doses are nonzero
@with_kw struct PolyDosesRandom{T<:NamedTuple, T1<:Categorical, T2<:Categorical, T3<:Tuple, T4<:Tuple, T5<:AbstractFloat, T6<:Int} <: DoseGenerator 
    dose_distr::T   #dose_distr = (drug_name = (min = minimal dose, avg_num = average number of minimal, max_num = max number of minimal))
    distr_first::T1
    distr_second::T2
    names_first::T3
    names_second::T4
    prob_second::T5 = 0.0
    times_per_day_first::T6 = 1
    times_per_day_second::T6 = 1 
    assign_not_supported::Bool = false #controls if assign_dose! assigns drugs not given in names
end

#from probabilities given in named tuples constructs categorical and map to symbol as tuple of drug names
function PolyDosesRandom(dose_distr::NamedTuple, distr_first::NamedTuple, distr_second::NamedTuple; prob_second::AbstractFloat = 0.0, times_per_day_first::Int = 1, times_per_day_second::Int = 1, assign_not_supported::Bool = false)
    names_first = keys(distr_first)
    names_second = keys(distr_second)
    #check keys in distr_first/second have a key in default doses
    dose_names = keys(default_doses)
    if !(isempty(setdiff(names_first, dose_names))) || !(isempty(setdiff(names_second, dose_names)))
        @warn "Not all possible drug choices have an assigned default dose"
    end
    distr_one = Categorical(collect(values(distr_first)))
    distr_two = Categorical(collect(values(distr_second)))
    return PolyDosesRandom(dose_distr, distr_one, distr_two, names_first, names_second, prob_second, times_per_day_first, times_per_day_second, assign_not_supported)
end

#from PK model constructs default doses based on names, distribution for both uniform over names
function PolyDosesRandom(m::PKModel; default_min_dose::AbstractFloat = 1.0, default_avg_multiple_dose::AbstractFloat = 5.0, default_max_multiple_dose::Int = 10, prob_second::AbstractFloat = 0.0, times_per_day_first::Int = 1, times_per_day_second::Int = 1, assign_not_supported::Bool = false)
    names = Tuple(get_keys_PK(m).d)
    N = length(names)
    dose_distr = NamedTuple{names}([(min = default_min_dose, avg_num = default_avg_multiple_dose, max_num = default_max_multiple_dose) for i in 1:N])
    distr_first = Categorical([1/N for i in 1:N])
    return PolyDosesRandom(dose_distr, distr_first, distr_first, names, names, prob_second, times_per_day_first, times_per_day_second, assign_not_supported)
end

#for 4 preimplemented mono drug PK models assign default min/avg etc from literature
function PolyDosesRandom(m::PKModel, appropriate_dosing::Bool)
    dose_distr = (d_VPA = (min = 150.0, avg_num = 5.0, max_num = 14), d_LEV = (min = 100.0, avg_num = 10.0, max_num = 30), s_LEV_unnormalised = (min = 100.0, avg_num = 10.0, max_num = 30),
                d_LTG = (min = 25.0, avg_num = 4.0, max_num = 24), d_CBZ = (min = 200.0, avg_num = 3.0, max_num = 8))
    if (typeof(m).name.wrapper in [PKLEV, PKLEVNoAbsorption, PKCBZ, PKVPA, PKLTG]) && appropriate_dosing 
        info = dose_distr[m.keys.d][1]
        dose_gen = PolyDosesRandom(m, default_min_dose = info.min, default_avg_multiple_dose = info.avg_num, default_max_multiple_dose = info.max_num, times_per_day_first = 2)
    else
        dose_gen = PolyDosesRandom(m, default_min_dose = 100.0)
        if appropriate_dosing
            @warn "No appropriate dosing preset for this drug"
        end
    end
    return dose_gen
end

#picks first drug from distribution, with specified probability also second drug from that distribution, wo_treatment gives first dosetime, times per day and doses specified in model
#if second picked drug is same as first, no second drug is assigned
#when called again on person with doses already assigned, next doses are picked independently from current ones
#when a drug not contained in names is picked, the attribute assign_not_supported decides if assigned anyway
#doses drawn according to specified distribution, multiple of min_dose via binomial with given values
function assign_dose!(m::PolyDosesRandom, person::Person; names::NamedTuple = (d = keys(m.default_doses),), timeframe::AbstractFloat = 10.0, wo_treatment::AbstractFloat = 0.0)
    if isempty(person.dosing)
        last_dosetime = -1
    else
        last_dosetime = person.dosing[end].t
    end
    #for first wo_treatment time no treatment, don't need to add zero callbacks to dosing
    no_dose = min(wo_treatment,timeframe)
    last_dosetime += no_dose
    #pick first drug, check if supported
    drug_one = m.names_first[rand(m.distr_first)]
    #with assigned probability pick second drug
    if rand(Bernoulli(m.prob_second))
        #check second drug not the same as first
        drug_two = m.names_second[rand(m.distr_second)]
        if !(drug_one == drug_two)
            drugs = (drug_one, drug_two)
        else
            drugs = (drug_one,)
        end
    else
        drugs = (drug_one,)
    end
    #randomly assign doses
    info_doses = values(NamedTuple{drugs}(m.dose_distr))
    doses = Tuple(d.min*(1+rand(Binomial(d.max_num-1, (d.avg_num-1)/(d.max_num-1)))) for d in info_doses)
    times = (m.times_per_day_first, m.times_per_day_second)
    #check if picked drugs are supported or else is assign_not_supported is set to true
    if m.assign_not_supported
        info = Tuple((drug = drugs[i], dose = doses[i], times = times[i]) for i in eachindex(drugs))
    else
        info = Tuple((drug = drugs[i], dose = doses[i], times = times[i]) for i in eachindex(drugs) if (drugs[i] in names.d))
    end
    #append for each dose time and drug taken
    next_doses = [(t = i+1 + j/d.times, dose = d.dose, state = d.drug) for i in last_dosetime:(last_dosetime+timeframe-(no_dose+1)) for d in info for j in 0:(d.times-1)]
    append!(person.dosing,next_doses)
end

#Outline for later assign drug not randomly, dependence on covariates

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