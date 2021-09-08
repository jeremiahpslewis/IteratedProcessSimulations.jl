using IteratedProcessSimulations
using Test
using DataFrames
using DataFrameMacros

using Soss
using MeasureTheory
using StatsModels # TODO: remove from package dependencies
using Chain

import Distributions: Binomial
import GLM: glm, LogitLink, fit, coef, predict, @formula

@testset "IteratedProcessSimulations.jl" begin
	@testset "ml_biz_ops_sim" begin
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

		ips = IteratedProcessSimulation(test_dgp, test_simulation_description)#, generate_data, fit_model, summarize_model, choose_observations)
		@test isa(ips, IteratedProcessSimulation)

		function generate_data(ips::IteratedProcessSimulation, epoch::Int)
			epoch_params = @chain ips.simulation_description @subset(:epoch == epoch) eachrow() _[1]
			n_datapoints = epoch_params.n_datapoints
		
			# Generate one cycle of data
			# TODO: Using 'burn in here', look into and drop this if unnecessary...
			df = @chain epoch_params begin
				ips.data_generating_process # Apply epoch_params to the data_generating_process
				rand(n_datapoints * 5)
				DataFrame()
				last(n_datapoints)
			end
		
			# Add data generating process parameters for tracking
			df = crossjoin(df, DataFrame(epoch_params)[!, [:epoch]])
			df = @chain df @transform(:observed = false, :predicted_labels = nothing)
			
			return df
		end

		test_data = generate_data(ips, 0)
		@test nrow(test_data) == 200
		@test ncol(test_data) == 7

		test_data_1 = generate_data(ips, 1)
		@test nrow(test_data_1) == 40
		@test ncol(test_data_1) == 7

		function fit_model(ips::IteratedProcessSimulation, training_data::DataFrame, new_data::DataFrame, epoch::Int)
			# Drop unobserved outcomes
			training_data = @chain training_data @subset(:observed == true)
		
			# TODO: add example where model is dynamic, chosen via simulation parameters!
			glm(@formula(default ~ 1 + income_eur), training_data, Binomial(), LogitLink())	
		end

		test_data = @chain test_data @transform(:observed = true)
		m = fit_model(ips, test_data, test_data_1, 1)
		@test coef(m) ≈ [0.18764107801174243, 0.06135920993917277]
		
		function summarize_model(ips::IteratedProcessSimulation, model::StatsModels.TableRegressionModel, simulation_data::DataFrame, new_data::DataFrame, epoch::Int)
			DataFrame(:epoch => [epoch], :income_coef => coef(values(model))[1])
		end
		
		@test summarize_model(ips, m, test_data, test_data, 1)[1, :income_coef] ≈ 0.019

		function choose_observations(ips::IteratedProcessSimulation, model::StatsModels.TableRegressionModel, new_data::DataFrame, epoch::Int)
			# Select 'unobserved' datapoints
			new_data[!, :predicted_labels] = predict(model, new_data)
					
			new_data = @chain new_data begin
				@transform(:observed = true)
			end
		
			return new_data
		end		

		df = choose_observations(ips, m, test_data_1, 1)
		@test nrow(df) == 40
		@test (@chain df @combine(sum(:observed)) _[!, :observed_sum][1]) == 40
	

		# @test_throws 
		run_simulation(ips)
		# @test_throws ArgumentError run_(test_dgp, DataFrame(), train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
	
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
