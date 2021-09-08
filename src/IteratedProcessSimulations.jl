module IteratedProcessSimulations

using DataFrames
using DataFrameMacros

using Soss
using Chain

using UUIDs

export IteratedProcessSimulation
export validate_inputs
export run_simulation


struct IteratedProcessSimulation
	data_generating_process::Soss.Model
	simulation_description::DataFrame
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
    validate_train_model_function(train_model)
end

function run_simulation(ips::IteratedProcessSimulation)
	
	simulation_data = DataFrame()
	model_summary = DataFrame()
	model_objects = []

	# Validate simulation inputs
	validate_inputs(ips)

	# Iterate over Epochs (starting with epoch = 1)
	for epoch in ips.simulation_description[!, :epoch]

		# Generate new data
		new_data = generate_data(ips, epoch)

		# Skip model training and decision for 'historical' epochs prior to epoch = 1
		if epoch > 0

			# Train model on existing data
			m = fit_model(ips, simulation_df, new_data, epoch)

			# Save model summary for later analysis
			append!(model_summary, summarize_model(ips, m, simulation_data, new_data, epoch), promote=true)

			# Save model object for later analysis
			append!(model_objects, m)

			# Choose datapoint 'observations' based on model
			new_data = choose_observations(ips, m, new_data, epoch)
		end

		# Add new data to dataset
		append!(simulation_data, new_data, promote=true)	
	end

	# Tag dataset with simulation id
	simulation_id = string(UUIDs.uuid4())
    simulation_data = @chain simulation_data @transform(:simulation_id = simulation_id)
    model_summary = @chain model_summary @transform(:simulation_id = simulation_id)

	return simulation_data, model_summary, model_objects
end

function ml_biz_ops_sim(ips::IteratedProcessSimulation, n_simulations::Int)
	simulation_data = DataFrame()
	model_summary = DataFrame()

	for i in 1:n_simulations
		d_1, ms_1, mo_1 = run_simulation(ips)
		append!(simulation_data, d_1)
		append!(model_summary, m_1)
	end

	return simulation_data, model_summary
end

end
