using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

struct FullModel
    #potentially split measurement and PK model?
    pk_model::PK_Model
    seizure_model::Seizure_Model
    population_gen::Person_Generator
    dose_gen::Dose_Generator
end


#data should be [person structs], save seizure and dosing data in persons
function get_likelihood(m::FullModel, data::Tuple) 
    #check if either model has random effects
    if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
        return 0
    else
        likeli = 1
        for i in eachindex(data)
            likeli = likeli*get_individual_likelihood(m, data[i])
            #this prob wont work for functions and i should move both into one
        end

        #do some optimisation on that
        return 0
    end
end



#no random effects here
function get_individual_likelihood(m::FullModel, person::Person)
    covariates = person.covariates
    #for each model write unpacker for covariates returning correct choice of covariates
    #in PK_model, Seizure_model save required covariates as list of keys
    covariates_PK = NamedTuple{m.pk_model.cov}(covariates)
    covariates_Seizure = NamedTuple{m.seizure_model.cov}(covariates)
    #Is this well-behaved if covariate list empty?
    function likelihood(Theta)
        #unpack Theta somehow into two named tuples
        Theta_PK = unpack_param(m.pk_model, Theta)
        Theta_Seizure = unpack_param(m.seizure_model, Theta)
        #Create new model instances with these thetas and covariates better?
        internal = 0 #type as m.pk_model with different 
        #solve PK ODE for this model instance
        solution = solve_ode(m.pk_model; param=Theta_PK, cov=covariates_PK) 
        likeli = measurement_likeli()
    end
end
