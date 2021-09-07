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
		function summarize_logit_model_test(m, sim_data, epoch)
			DataFrame(:epoch => [epoch], :income_coef => coef(values(m))[1])
		end

		test_dgp = @model params_1 begin
			income_eur ~ Poisson(params_1[:mu_income])
			individual_attributes ~ Normal(1, 10)
			default_prob = logistic(individual_attributes + income_eur)
			default ~ Bernoulli(default_prob)
		end
		
		n_epochs = 3
		n_data_points_per_epoch = 20

		test_dgp_params = DataFrame(
			"n_datapoints" => fill(n_data_points_per_epoch, n_epochs),
			"mu_income" => 1:n_epochs,
			"epoch" => 0:(n_epochs-1),
			"target_variable" => fill(:default, n_epochs),
			"feature_variables" => fill([:income_eur], n_epochs),
			"model_formula" => fill(@formula(default ~ 1 + income_eur), n_epochs)
		)
		test_data = generate_data(test_dgp, test_dgp_params[1, :])
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
