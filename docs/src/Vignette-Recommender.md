# Recommender Vignette

```@example 1
using IteratedProcessSimulations
using Recommendation
using DataFrames
using Soss
using MeasureTheory
using Chain
using DataFrameMacros
using UUIDs
using VegaLite
import Distributions
```

## Simulation Premises

A bookstore has customer account data on previous purchases
as well as a monthly newsletter in which it can suggest three books to read,
personalized to each customer.

All books cost the same. Each book has latent attributes, quality and topicality,
which are fixed. Each customer has unique preferences weighting these two factors
and resulting in a utility score. The customer chooses the highest utility book each month,
as long as it has a utility greater than 0
(the utility of the most attractive non-book good available and average score across all books and attributes).

New books are released each month. The bookstore uses a collaborative filter to
identify optimal books to offer to each user. Once a user has chosen a book,
its user-specific utility becomes visible to the bookstore (i.e. a rating).

To gather feedback on newly released books, the bookstore distributes
copies to 10% of users in the release month, in exchange for their rating
which becomes available instantly. These books, for sake of simplicity
can be repurchased (consumed) by the user.

The simulation runs over a course of 36 months.

## Simulate User Preferences

```@example 1
n_users = 100
n_books_per_month = 15
n_months = 36
pct_pre_read = 0.1  # X% of new books are 'pre-read' by users

# Define data generating process for the users
user_dgp = @model params begin
        user_utility_weight_quality ~ Distributions.TruncatedNormal(0.5, 0.1, 0, 1)
        user_utility_weight_topicality = 1 - user_utility_weight_quality
end

user_sim_description = DataFrame(
    "n_datapoints" => [n_users],
    "epoch" => [0],
)
```

## Create user sample

```@example 1
user_attributes = @chain generate_data(user_dgp, user_sim_description) begin
    @transform(:user_id = @c 1:length(:id))
    @select(:user_id, :user_utility_weight_quality, :user_utility_weight_topicality)
end

first(user_attributes, 4)
```

## Define data generating process for the books

```@example 1
book_dgp = @model params begin
        quality ~ Distributions.TruncatedNormal(10, 3, 0, 100)
        topicality ~ Distributions.TruncatedNormal(10, 3, 0, 100)
end

book_sim_description = DataFrame(
    "n_datapoints" => fill(n_books_per_month, 36),
    "epoch" => 1:36,
)

# TODO: remove line, just for testing
book_attributes = generate_data(book_dgp, eachrow(book_sim_description)[2])

first(book_attributes, 4)
```

## Build user-book dataframe via `transform_data` function

```@example 1
function transform_data(book_df)

    book_df = @chain book_df @transform(:book_id = @c 1:nrow(book_df)) @transform(:book_id = :book_id + (:epoch - 1) * nrow(book_df))
    user_book_df = @chain book_df begin
        crossjoin(user_attributes)
    end

    user_book_df = @chain user_book_df begin
        @transform(
            :user_book_utility = :topicality * :user_utility_weight_topicality + :quality * :user_utility_weight_quality,
            # X% of new books are 'pre-read' by users
            :pre_read = rand(Bernoulli(pct_pre_read))
                   )
    end

    user_book_df[!, :predicted_utility] .= missing
    user_book_df[!, :predicted_utility] = convert(Vector{Union{Missing, Float64}}, user_book_df[!, :predicted_utility])

    return user_book_df
end

# TODO: remove line, just for testing
new_data = transform_data(book_attributes)

first(new_data, 4)
```

## Define Machine Learning Model

```@example 1
# TODO: remove line, just for testing
training_data = new_data

function convert_dataframe_to_recommender(df::DataFrame, n_users, n_books)
    event_list = []
    for row in eachrow(df)
        # Here we assume that a user knows and reports their utility after having read the book
        push!(event_list, Event(row[:user_id], row[:book_id], row[:user_book_utility]))
    end

    event_list = convert(Vector{Event}, event_list)
    
    data = DataAccessor(event_list, n_users, n_books)

    return data
end

function fit_model(epoch_parameters::DataFrameRow, training_data::DataFrame, new_data::DataFrame)
        # Note, the statement below permanently adds the new data to the training dataset
        append!(training_data, new_data, promote=true)

        n_users = maximum(training_data[!, :user_id])
        n_books = maximum(training_data[!, :book_id])
        # Drop unobserved outcomes
        training_data = @chain training_data @subset(:observed | :pre_read)
            

        data = convert_dataframe_to_recommender(training_data, n_users, n_books)
        recommender = SVD(data, 10)
        build!(recommender)

        return recommender
end

fit_model(eachrow(book_sim_description)[1], training_data, new_data)
```

```@example 1
# Skip using this to track parameter / model outcomes for now, but could be useful in a real study...
function summarize_model(epoch_parameters::DataFrameRow, model, simulation_data::DataFrame, new_data::DataFrame)
    DataFrame(:epoch => [epoch_parameters.epoch])
end
```


```@example 1
function choose_observations(epoch_parameters::DataFrameRow, recommender, new_data::DataFrame, simulation_data::DataFrame)
    # NOTE: as the new_data is already added to the simulation data during the model fit, no need to use `new_data` here

    # Each user gets to read an additional book!
    for user_id in unique((@chain simulation_data @subset(!:observed) _[!, :user_id]))
        user_prediction = recommend(recommender, user_id, 1, (@chain simulation_data @subset(!:observed & (:user_id == user_id)) @select(:book_id) unique _[!, :book_id]))
        best_book = user_prediction[1][1]
        best_book_score = user_prediction[1][2]
        simulation_data[((simulation_data[!, :user_id] .== user_id) .& (simulation_data[!, :book_id] .== best_book)), :observed] .= true
        simulation_data[((simulation_data[!, :user_id] .== user_id) .& (simulation_data[!, :book_id] .== best_book)), :predicted_utility] .= best_book_score
    end

    return simulation_data
end	
```


## Put it all together and run the simulation


```@example 1
ips = IteratedProcessSimulation(book_dgp, book_sim_description, transform_data, fit_model, summarize_model, choose_observations)

simulation_data, model_summary, model_objects = run_simulation(ips)

# TODO: for debugging, remove
user_id = 1
recommender = model_objects[36]
```

## Assess outcome quality
 
  
```@example 1
utility_rollup = @chain simulation_data begin
    @groupby(:user_id, :user_utility_weight_quality, :user_utility_weight_topicality)
    @combine(:user_utility_achieved = sum(:user_book_utility[:observed]),
             :user_utility_predicted = sum(:predicted_utility[:observed]), # this should be strictly positive
             :n_books_purchased = length(:predicted_utility[:observed])
             :user_utility_possible = @c sum(sort(:user_book_utility, rev=true)[1:n_months]) # user has the possibility of choosing X books = n_months
             )
    @transform(:pct_utility_achieved = :user_utility_achieved / :user_utility_possible)
end

first(utility_rollup, 6)
```


## Plot Utility Distribution across Users

```@example 1
utility_rollup |> @vlplot(:bar, width=500, height=300, x={:user_utility_achieved, bin={step=0.5}, title="Total Utility Achieved"}, y={"count()", title="User Count"}, title="Utility Achieved per User")
```

## Plot Predicted Utility vs Actual Utility

```@example 1
utility_rollup |> @vlplot(:point, width=500, height=500, x={:user_utility_achieved, title="Total Utility Achieved"}, y={:user_utility_possible, title="Total Utility Possible"}, title="Model Relatively Ineffective")
```

## Plot Percent Utility Achieved across Users

```@example 1
utility_rollup |> @vlplot(width=500, height=300, :bar, x={:pct_utility_achieved, bin={step=0.005}, title="Percent Utility Achieved", axis={format="%"}}, y={"count()", title="User Count"}, title="Percentage of Possible Utility Achieved per User")
```


## Plot User Preferences

```@example 1
utility_rollup |> @vlplot(width=500, height=300, :bar, x={:user_utility_weight_quality, bin={step=0.05}, title="User Preference for Quality (over Topicality)", axis={format="%"}}, y={"count()", title="User Count"}, title="User Preferences, Percentage Weight Quality (vs Topicality)")
```


## Plot Individual Preferences Against Total Utility Possible

```@example 1
utility_rollup |> @vlplot(width=500, height=300, :bar, x={:user_utility_weight_quality, bin={step=0.01}, title="User Preference for Quality (over Topicality)", axis={format="%"}}, y={"mean(user_utility_possible)", title="Possible Utility"}, title="Possible Utility by User Preferences")
```

## Plot Individual Preferences Against Utility

```@example 1
utility_rollup |> @vlplot(width=500, height=300, :bar, x={:user_utility_weight_quality, bin={step=0.01}, title="User Preference for Quality (over Topicality)", axis={format="%"}}, y={"mean(pct_utility_achieved)", title="Average Percent Utility Achieved", axis={format="%"}}, title="Percentage of Possible Utility Achieved by User Preferences")
```


```@example 1
utility_rollup |> @vlplot(width=500, height=300, :bar, x={:user_utility_weight_quality, bin={step=0.01}, title="User Preference for Quality (over Topicality)", axis={format="%"}}, y={"mean(user_utility_achieved)", title="Average Utility Achieved"}, title="Average Utility Achieved by User Preferences")
```
