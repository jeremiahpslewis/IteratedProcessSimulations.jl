using MLBizOps
using Test

@testset "MLBizOps.jl" begin
	@testset "ml_biz_ops_sim" begin
		function summarize_logit_model_test(m, epoch)
			DataFrame(:epoch => [epoch], :income_coef => coef(values(m)))
		end

		test_dgp = @model params_1 begin
			income_k_eur ~ Poisson(params_1[:mu_income])
			individual_attributes ~ Normal(1, 10)
			default_prob = logistic(individual_attributes + income_k_eur)
			default ~ Binomial(default_prob)
		end
	
		test_dgp_params = DataFrame(
				"n_datapoints" => [100, 200, 300],
				"mu_income" => [100, 200, 500],
				"epoch" => [0, 1, 2],
				"target_variable" => fill(:default, 3),
				"feature_variables" => fill([:income_k_eur], 3),
	
		)
		test_data = generate_data(test_dgp, test_dgp_params[1, :])
		@test nrow(test_data) == 100
		@test ncol(test_data) == 5

		hist_params = @chain test_dgp_params @subset(:epoch == 0)
		test_data = gather_historical_data(test_dgp, hist_params)
		@test nrow(test_data) == 100
		@test ncol(test_data) == 7

	
		function train_model_demo_1(training_data::DataFrame, epoch_params)
			m = fit(GeneralizedLinearModel,
					Matrix(training_data[!, epoch_params.feature_variables]),
					training_data[!, epoch_params.target_variable],
					Binomial(),
					LogitLink())
	
			return m
		end		
	
	
		@test_throws MethodError ml_biz_ops_sim(DataFrame(), DataFrame(), train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
		@test_throws ArgumentError ml_biz_ops_sim(test_dgp, DataFrame(), train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
	
		test_data, model_objs = ml_biz_ops_sim(test_dgp, test_dgp_params, train_model_demo_1, summarize_logit_model_test, choose_and_label_data)
		@test ncol(test_data) == 8
		@test nrow(test_data) == 600
	end

    @testset "Train model" begin
        function train_model_test_1(training_data::DataFrame)
            target_variable = :Y
            feature_variables = [:X]
            m = fit(GeneralizedLinearModel,
                    Matrix(training_data[!, feature_variables]),
                    training_data[!, target_variable],
                    Binomial(),
                    LogitLink())
    
            return m
        end
        d1 = DataFrame(Y=[1,1,0,0], X=[3,3,5,5])	
        m = train_model_test_1(d1)
        @test coef(m)[1] ≈ -0.12058218173904946
        
        prediction_vect = predict(m, Matrix{Float64}(d1[!, [:X]]))
        @test prediction_vect[1] ≈ 0.41053684244143024
    end

end
