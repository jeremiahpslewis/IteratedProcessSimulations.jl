using IteratedProcessSimulations
using Test
using DataFrames
using DataFrameMacros

using Soss
using MeasureTheory
using StatsModels
using Chain

import Distributions: Binomial
import GLM: glm, LogitLink, fit, coef, predict, @formula

@testset "IteratedProcessSimulations.jl" begin
	test_dgp = @model params begin
		income_eur ~ Normal(params[:mu_income], 0.1)
		individual_attributes ~ Normal(1, 0.1)
		default_prob = logistic(individual_attributes + income_eur)
		default ~ Bernoulli(default_prob)
	end
	
	n_epochs = 3

	test_simulation_description = DataFrame(
		"n_datapoints" => [200, 40, 30],
		"mu_income" => fill(1, n_epochs),
		"epoch" => 0:(n_epochs-1),
	)

	epoch_parameters = eachrow(test_simulation_description)[1]

	test_data = generate_data(test_dgp, epoch_parameters)
	@test nrow(test_data) == 200
	@test ncol(test_data) == 8

	epoch_parameters = eachrow(test_simulation_description)[2]
	test_data_1 = generate_data(test_dgp, epoch_parameters)
	@test nrow(test_data_1) == 40
	@test ncol(test_data_1) == 8

	test_data_2 = generate_data(test_dgp, test_simulation_description)
	@test nrow(test_data_2) == 270
	@test ncol(test_data_2) == 8

	function fit_model(epoch_parameters::DataFrameRow, training_data::DataFrame, new_data::DataFrame)
		# Drop unobserved outcomes
		training_data = @chain training_data @subset(:observed == true)
	
		# TODO: add example where model is dynamic, chosen via simulation parameters!
		glm(@formula(default ~ 1 + income_eur), training_data, Binomial(), LogitLink())	
	end

	test_data = @chain test_data @transform(:observed = true)
	m = fit_model(epoch_parameters, test_data, test_data_1)
	# @test coef(m) ≈ [0.18764107801174243, 0.06135920993917277]

	function transform_data(df::DataFrame)
		return df
	end

	function summarize_model(epoch_parameters::DataFrameRow, model::StatsModels.TableRegressionModel, simulation_data::DataFrame, new_data::DataFrame)
		DataFrame(:epoch => [epoch_parameters.epoch], :income_coef => coef(values(model))[1])
	end
	
	# @test summarize_model(epoch_parameters, m, test_data, test_data)[1, :income_coef] ≈ 0.019

	function choose_observations(epoch_parameters::DataFrameRow, model::Union{StatsModels.TableRegressionModel, Nothing}, new_data::DataFrame, simulation_data::DataFrame)
		# Select 'unobserved' datapoints

		# All 'historical' data points are considered observed
		if epoch_parameters.epoch < 1
			new_data = @chain new_data begin
				@transform(:observed = true)
			end			
		else
			new_data[!, :predicted_labels] = predict(model, new_data)
				
			new_data = @chain new_data begin
				@transform(:observed = true)
			end	
		end

		# Add new data to dataset
		append!(simulation_data, new_data, promote=true)	

		return simulation_data
	end		

	df = choose_observations(epoch_parameters, m, test_data_1, test_data_1)
	@test nrow(df) == 80
	@test (@chain df @combine(sum(:observed)) _[!, :observed_sum][1]) == 40

	ips = IteratedProcessSimulation(test_dgp, test_simulation_description, transform_data, fit_model, summarize_model, choose_observations)
	@test isa(ips, IteratedProcessSimulation)		

	sim_data, model_summary, model_objects = run_simulation(ips)

	@test ncol(sim_data) == 9
	@test nrow(sim_data) == 270

	sim_data, model_summary, model_objects = run_simulation(ips, 5)

	@test ncol(sim_data) == 9
	@test nrow(sim_data) == 1350
end
