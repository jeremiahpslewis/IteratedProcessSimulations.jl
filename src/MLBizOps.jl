module MLBizOps

# Write your package code here.
"""Check whether dgp_params dataframe is properly specified with n_datapoints column for i.i.d. sampling from data generating process"""
function validate_dgp_params(dgp_params)
    isa(dgp_params.n_datapoints, Vector{Int})
    
    isa(dgp_params.epoch, Vector)
end

# TODO: validate that model takes data frame and dataframe row as inputs and is type 'function'
function validate_train_model_function(train_model_function)

end

function validate_inputs(dgp_spec, dgp_params, train_model)
    validate_dgp_params(dgp_params)
    validate_train_model_function(train_model)
end

function generate_data(dgp_spec::Soss.Model, dgp_params)
	n_datapoints = dgp_params.n_datapoints
	# Generate one cycle of data
	# TODO: Using 'burn in here', look into and drop this if unnecessary...
	df = @chain dgp_params begin
		dgp_spec()
		rand(n_datapoints * 5)
		DataFrame()
		last(n_datapoints)
	end

	# Add data generating process parameters for tracking
	df = crossjoin(df, DataFrame(dgp_params)[!, [:epoch]])
	
	return df
end

function gather_historical_data(dgp_spec::Soss.Model, dgp_params::DataFrame)
	data = DataFrame()
	
	for x in eachrow(dgp_params)
		one_epoch = generate_data(dgp_spec, x)
		append!(data, one_epoch)
	end
	
	data = @chain data @transform(:observed = true, :predicted_labels = nothing)

	return data
end

function ml_biz_ops_sim(dgp_spec::Soss.Model, dgp_params::DataFrame, train_model::Function, summarize_model::Function, choose_and_label_data::Function)
	
	sim_data = DataFrame()
	model_summary = DataFrame()
	# Validate simulation inputs
	validate_inputs(dgp_spec, dgp_params, train_model)

	# Collect Historical Data
	@chain dgp_params begin
		@subset(:epoch <= 0)
		gather_historical_data(dgp_spec, _)
		append!(sim_data, _)
	end
	
	# Business Operations Begin Here

	dgp_sim_params = @chain dgp_params @subset(:epoch > 0)	

	# Iterate over Epochs (starting with epoch = 1)
	for epoch_params in eachrow(dgp_sim_params)

		# Train model on existing data
		m = @chain sim_data train_model(_, epoch_params)
		append!(model_summary, summarize_model(m, nrow(sim_data), epoch_params.epoch), promote=true)
		
		# Collect new data points (outcome unobserved)
		new_data = @chain epoch_params collect_unlabeled_data(dgp_spec, _)
		
		# Choose data points to be labeled and label them
		new_data = choose_and_label_data(new_data, m, epoch_params)
		append!(sim_data, new_data, promote=true)	
	end

	# Tag with simulation id
	simulation_id = string(UUIDs.uuid4())
    sim_data = @chain sim_data @transform(:simulation_id = simulation_id)
    model_summary = @chain model_summary @transform(:simulation_id = simulation_id)

	return sim_data, model_summary
end

function ml_biz_ops_sim(dgp_spec::Soss.Model, dgp_params::DataFrame, train_model::Function, summarize_model::Function, choose_and_label_data::Function, n_simulations::Int)
	df = DataFrame()
	model_summary = DataFrame()
	for i in 1:n_simulations
		d_1, m_1 = ml_biz_ops_sim(dgp_spec, dgp_params, train_model, summarize_model, choose_and_label_data)
		append!(df, d_1)
		append!(model_summary, m_1)
	end
	
	return df, model_summary
end

end
