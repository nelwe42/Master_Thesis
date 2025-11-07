using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

struct FullModel
    pk_model::PK_Model
    seizure_model::Seizure_Model
    population_gen::Person_Generator
    dose_gen::Dose_Generator
end


#data should be [person structs], save seizure, measurement and dosing data in persons
function get_loglikelihood(m::FullModel, data::Tuple) 
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    #    return loglikeli
    function loglikelihood(θ)
        loglikeli = 0
        for i in eachindex(data)
            person = data[i]
            sol = sol = solve_PK(m.pk_model, θ.PK, person, endpoint = person.measurements[end].timepoint)
            loglikeli = loglikeli + get_PK_loglikelihood(θ.PK, person; sol=sol)
                + get_seizure_loglikelihood(θ.Seizure, m.seizure_model, sol, person)
        end
        return loglikeli
    end
end

#m determines model parts, n determines number of people, timepoints for measurements
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints::AbstractVector = 0:14:time)
    population = generate_population(m.population_gen, n)
    for i in eachindex(population)
        person = population[i]
        assign_dose!(m.dose_gen, person, time)
        sol = generate_measurements!(m.pk_model, person, timepoints)
        generate_seizures!(m.seizure_model, sol, person, start = 0, day_number = time)
        #note for time = 10 seizure counts end on day 9 (end on midnight between day 9 and 10)
    end
    return population
end


function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::AbstractFloat, timepoints::AbstractVector = 0:14:time)
    population = generate_population(m.population_gen, n)
    for i in eachindex(population)
        person = population[i]
        passed_time = 0
        while passed_time < time
            increment = max(time, passed_time + update_reg)
            passed_time = passed_time + increment
            current_timepoints = [t for t in timepoints if (passed_time-increment)<= t < passed_time] #filter timepoints in this interval
            assign_dose!(m.dose_gen, person, increment)
            sol = generate_measurements!(m.pk_model, person, current_timepoints)
            generate_seizures!(m.seizure_model, sol, person, start = passed_time+1, day_number = increment)
        end
    end
end