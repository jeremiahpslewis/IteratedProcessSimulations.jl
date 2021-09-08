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

struct IteratedProcessSimulation
	data_generating_process::Soss.Model
	simulation_description::DataFrame
	fit_model::Function
	summarize_model::Function
	choose_observations::Function
end

# Write your package code here.
"""Check whether simulation_description dataframe is properly specified with n_datapoints column for i.i.d. sampling from data generating process"""
function validate_simulation_description(simulation_description::DataFrame)
    isa(simulation_description.n_datapoints, Vector{Int})
    
    isa(simulation_description.epoch, Vector)

	# TODO: Check that dataframe is sorted / ordered by epoch and no duplicates!
end

# TODO: validate that model takes data frame and dataframe row as inputs and is type 'function'
function validate_train_model_function(train_model_function)

end

function validate_inputs(ips::IteratedProcessSimulation)
    validate_simulation_description(ips.simulation_description)
end

function generate_data(data_generating_process::Soss.Model, epoch_parameters::DataFrameRow)
	n_datapoints = epoch_parameters.n_datapoints

	# Generate one cycle of data
	# TODO: Using 'burn in here', look into and drop this if unnecessary...
	df = @chain epoch_parameters begin
		data_generating_process() # Apply epoch_parameters to the data_generating_process
		rand(n_datapoints * 5)
		DataFrame()
		last(n_datapoints)
	end

	# Add data generating process parameters for tracking
	df = crossjoin(df, DataFrame(epoch_parameters)[!, [:epoch]])
	df = @chain df @transform(:observed = false, :predicted_labels = nothing)
	
	return df
end

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
