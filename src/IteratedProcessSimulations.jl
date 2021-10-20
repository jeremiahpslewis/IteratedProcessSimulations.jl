module IteratedProcessSimulations

import Base: @invokelatest

using DataFrames
using DataFrameMacros

using Soss
using Chain

using UUIDs

export IteratedProcessSimulation
export validate_inputs
export run_simulation
export generate_data

"""
	IteratedProcessSimulation(data_generating_process, simulation_description, fit_model, summarize_model, choose_observations)

A type which collects all elements necessary to run an iterated simulation of a machine learning process.

The following attributes must be supplied:
"""
struct IteratedProcessSimulation
	"`data_generating_process` is a structural model which can be sampled from to generate synthetic data; it may be a function of parameters specified in `simulation_description`"
	data_generating_process::Soss.Model
	"`simulation_description` is a dataframe which contains all parameters necessary to describe one epoch of the simulation"
	simulation_description::DataFrame
	"`fit_model` is a function which takes epoch_parameters from `simulation_description` and synthetic data generated by the `data_generating_process` and fits a predictive model"
	fit_model::Function
	"`summarize_model` is a function which extracts parameters post-model fit and appends them to a dataframe"
	summarize_model::Function
	"`choose_observations` is a function which modifies the newly generated data of a given epoch based on the predictive model, usually setting some observations to 'observed'"
	choose_observations::Function
end

# Write your package code here.
"""Check whether simulation_description dataframe is properly specified with n_datapoints column for i.i.d. sampling from data generating process"""
function validate_simulation_description(simulation_description::DataFrame)
    isa(simulation_description.n_datapoints, Vector{Int}) || throw(ArgumentError("simulation_description must contain a :n_datapoints column of type Int"))
	isa(simulation_description.epoch, Vector{Int}) || throw(ArgumentError("simulation_description must contain an :epoch column of type Int"))

    (sort(simulation_description.epoch) == simulation_description.epoch) || throw(ArgumentError("simulation_description must be sorted by :epoch"))
	
	length(simulation_description.epoch) == length(unique(simulation_description.epoch)) || throw(ArgumentError(":epoch may not have duplicate values in simulation_description"))
end

function validate_fit_model(fit_model::Function) end
function validate_summarize_model(summarize_model::Function) end
function validate_choose_observations(choose_observations::Function) end


"""
	validate_inputs(ips::IteratedProcessSimulation)

Validate IteratedProcessSimulation object.
"""
function validate_inputs(ips::IteratedProcessSimulation)
    validate_simulation_description(ips.simulation_description)
	validate_fit_model(ips.fit_model)
	validate_summarize_model(ips.summarize_model)
	validate_choose_observations(ips.choose_observations)
end

"""
	generate_data(data_generating_process::Soss.Model, epoch_parameters::DataFrameRow)

Generate data from a data_generating_process for a single epoch.

"""
function generate_data(data_generating_process::Soss.Model, epoch_parameters::DataFrameRow)
	n_datapoints = epoch_parameters.n_datapoints

	# Generate one cycle of data
	df = @chain epoch_parameters begin
		data_generating_process() # Apply data_generating_process function to epoch_parameters
		rand(n_datapoints) # Sample from data_generating_process
		DataFrame() # Return a DataFrame
		@transform(:id = string(UUIDs.uuid4()))
	end

	df = @chain df @transform(:epoch = epoch_parameters.epoch, :observed = false, :predicted_labels = nothing)
	
	return df
end

"""
	generate_data(data_generating_process::Soss.Model, epoch_parameters::DataFrame)

Generate data from a data_generating_process for a series of epochs.

"""
function generate_data(data_generating_process::Soss.Model, epoch_parameters::DataFrame)
	df = DataFrame()
	for epoch_row in eachrow(epoch_parameters)
		append!(df, generate_data(data_generating_process, epoch_row))
	end
	return df
end

"""
	run_simulation(ips::IteratedProcessSimulation)

Run a single iterated process simulation.
"""
function run_simulation(ips::IteratedProcessSimulation)
	
	simulation_data = DataFrame()
	model_summary = DataFrame()
	model_objects = []

	# Validate simulation inputs
	validate_inputs(ips)

	# Iterate over Epochs (starting with epoch = 1)
	for epoch_parameters in eachrow(ips.simulation_description)

		# Generate new data
		new_data = generate_data(ips.data_generating_process, epoch_parameters)

		# Skip model training and decision for 'historical' epochs prior to epoch = 1
		if epoch_parameters.epoch > 0

			# Train model on existing data
			m = ips.fit_model(epoch_parameters, simulation_data, new_data)

			# Save model summary for later analysis
			append!(model_summary, ips.summarize_model(epoch_parameters, m, simulation_data, new_data), promote=true)

			# Save model object for later analysis
			append!(model_objects, [m])
		else
			m = nothing
		end

		# Choose datapoint 'observations' based on model
		new_data = ips.choose_observations(epoch_parameters, m, new_data)

		# Add new data to dataset
		append!(simulation_data, new_data, promote=true)	
	end

	# Tag dataset with simulation id
	simulation_id = string(UUIDs.uuid4())
    simulation_data = @chain simulation_data @transform(:simulation_id = simulation_id)
    model_summary = @chain model_summary @transform(:simulation_id = simulation_id)

	return simulation_data, model_summary, model_objects
end

"""
	run_simulation(ips::IteratedProcessSimulation, n_simulations::Int)

Run an iterated process simulation `n_simulations` times.
"""
function run_simulation(ips::IteratedProcessSimulation, n_simulations::Int)
	simulation_data = DataFrame()
	model_summary = DataFrame()
	model_objects = []

	for i in 1:n_simulations
		d_1, ms_1, mo_1 = run_simulation(ips)
		append!(simulation_data, d_1)
		append!(model_summary, ms_1)
		append!(model_objects, [mo_1])
	end

	return simulation_data, model_summary, model_objects
end

end
