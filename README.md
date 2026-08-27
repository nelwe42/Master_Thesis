# Modelling Framework for Epileptic Seizures



Code written for my master thesis "A Simulation Study on Modelling Seizures in Epilepsy Patients" at the University of Bonn under the supervision of Prof. Dr. Jan Hasenauer. 



The folder EpilepsyModels contains all relevant code while the other folders contain outputs of the various evaluations and plots. In the folder EpilepsyModels, the Project.toml and Manifest.toml file specify the project environment used in julia 1.11.7. The src folder contains the following:



* Person Generator.jl defining the Person struct for conveying an individuals' data and methods for generating a simulated population
* Dose Generator.jl for assigning doses to a simulated person
* PK Model.jl defining various PK models for common medications in epilepsy and ways to handle simulated measurements and likelihood calculation
* Seizure Model.jl defining models for seizure behaviour based on the PK output
* EpilepsyModels.jl integrating the previous four files for simulating data and performing inference on a combined FullModel
* Testing.jl and MultiData.jl for running simulations
* PlottingScript.jl and mini\_plots2.jl for visualisation



The output files in the other folders are txt documents and can be parsed for easier handeling as in PlottingScript.jl.

