using MLBizOps
using Test
using DataFrames
using DataFrameMacros

using Soss
using MeasureTheory
using Chain

import Distributions: Binomial
import GLM: glm, LogitLink, fit, coef, predict, @formula

@testset "MLBizOps.jl" begin
	@testset "ml_biz_ops_sim" begin


		function generate_data(ips::IteratedProcessSimulation, epoch::Int)
			epoch_params = ips.simulation_params[epoch, :]
			n_datapoints = epoch_params[!, :n_datapoints]
		
			# Generate one cycle of data
			# TODO: Using 'burn in here', look into and drop this if unnecessary...
			df = @chain epoch_params begin
				ips.data_generating_process(_) # Apply epoch_params to the data_generating_process
				rand(n_datapoints * 5)
				DataFrame()
				last(n_datapoints)
			end
		
			# Add data generating process parameters for tracking
			df = crossjoin(df, DataFrame(epoch_params)[!, [:epoch]])
			df = @chain df @transform(:observed = true, :predicted_labels = nothing)
			
			return df
		end
		
		function fit_model(ips::IteratedProcessSimulation, training_data::DataFrame, new_data::DataFrame, epoch::Int)
			# Drop unobserved outcomes
			training_data = @chain training_data @subset(:observed == true)
		
			# TODO: add example where model is dynamic, chosen via simulation parameters!
			glm(@formula(default ~ 1 + income_eur), training_data, Binomial(), LogitLink())	
		end
		
		function summarize_model(ips::IteratedProcessSimulation, model::StatsModels.TableRegressionModel, simulation_data::DataFrame, new_data::DataFrame, epoch::Int)
			DataFrame(:epoch => [epoch], :income_coef => coef(values(model))[1])
		end
		
		function choose_observations(ips::IteratedProcessSimulation, model::StatsModels.TableRegressionModel, new_data::DataFrame, epoch::Int)
			# Select 'unobserved' datapoints
			new_data[!, :predicted_labels] = predict(model, new_data)
					
			new_data = @chain new_data begin
				@transform(:observed = true)
			end
		
			return new_data
		end		

		test_dgp = @model params_1 begin
			income_eur ~ Poisson(params_1[:mu_income])
			individual_attributes ~ Normal(1, 10)
			default_prob = logistic(individual_attributes + income_eur)
			default ~ Bernoulli(default_prob)
		end
		
		n_epochs = 3
		n_data_points_per_epoch = 20

		test_simulation_description = DataFrame(
			"n_datapoints" => [20, 40, 30],
			"mu_income" => 1:n_epochs,
			"epoch" => 0:(n_epochs-1),
		)

		ips = IteratedProcessSimulation(test_dgp, test_simulation_description)
		test_data = generate_data(ips, 0)
		@test nrow(test_data) == 20
		@test ncol(test_data) == 5

		test_data = generate_data(ips, 1)
		@test nrow(test_data) == 20
		@test ncol(test_data) == 5

		hist_params = @chain test_dgp_params @subset(:epoch == 0)
		test_data = MLBizOps.gather_historical_data(test_dgp, hist_params)
		@test nrow(test_data) == 20
		@test ncol(test_data) == 7

	
		function train_model_demo_1(training_data::DataFrame, epoch_params)
			m = glm(epoch_params.model_formula, training_data, Binomial(), LogitLink())	
			return m
		end

		function choose_and_label_data(unlabeled_data::DataFrame, model, epoch_params)
			# feature_vars = epoch_params.feature_variables
		
			unlabeled_data[!, :predicted_labels] = predict(model, unlabeled_data)
			
			unlabeled_data = @chain unlabeled_data begin
				# @transform(:observed = (rand() < 0.1))
				@transform(:observed = true)
				# @transform(:observed = (:predicted_labels < 0.10) | (rand() < 0.1))		
			end
			
			
			
			return unlabeled_data
		end

		@test_throws MethodError ml_biz_ops_sim(DataFrame(), DataFrame(), train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
		@test_throws ArgumentError ml_biz_ops_sim(test_dgp, DataFrame(), train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
	
		test_data, model_objs = ml_biz_ops_sim(test_dgp, test_dgp_params, train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
		@test ncol(test_data) == 8
		@test nrow(test_data) == 60
	end

    @testset "Train model" begin
		# function train_model_demo_1(training_data::DataFrame, epoch_params)
		# 	m = glm(epoch_params.model_formula, training_data, Bernoulli(), LogitLink())	
		# 	return m
		# end		
	
        # d1 = DataFrame(Y=[1,1,0,0], X=[3,3,5,5])	
        # m = train_model_test_1(d1)
        # @test coef(m)[1] ≈ -0.12058218173904946
        
        # prediction_vect = predict(m, Matrix{Float64}(d1[!, [:X]]))
        # @test prediction_vect[1] ≈ 0.41053684244143024
    end

end
