# Recommender Vignette

```julia
using IteratedProcessSimulations
using Recommendation
using DataFrames
using Soss
using MeasureTheory
using Chain
using DataFrameMacros
using UUIDs
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
which becomes available instantly.

The simulation runs over a course of 36 months.

## Simulate User Preferences

```julia
n_users = 500
n_books_per_month = 10
n_months = 36
pct_pre_read = 0.1  # X% of new books are 'pre-read' by users

# Define data generating process for the users
user_dgp = @model params begin
        utility_weight_quality_raw ~ Normal(0, 0.3)
        user_utility_weight_quality = logistic(utility_weight_quality_raw)
        user_utility_weight_topicality = 1 - user_utility_weight_quality
end

user_sim_description = DataFrame(
    "n_datapoints" => [n_users],
    "epoch" => [0],
)
```

## Create user sample
```julia

user_attributes = @chain generate_data(user_dgp, user_sim_description) begin
    @transform(:user_id = @c 1:length(:id))
    @select(:user_id, :user_utility_weight_quality, :user_utility_weight_topicality)
end
```

## Define data generating process for the books

```julia
book_dgp = @model params begin
        quality ~ Normal(0, 5)
        topicality ~ Normal(0, 5)
end

book_sim_description = DataFrame(
    "n_datapoints" => fill(n_books_per_month, 36),
    "epoch" => 1:36,
)

# TODO: remove line, just for testing
book_attributes = generate_data(book_dgp, eachrow(book_sim_description)[2])
```

## Build user-book dataframe via `transform_data` function

```julia
function transform_data(book_df)

    book_df = @chain book_df @transform(:book_id = @c 1:nrow(book_df)) @transform(:book_id = :book_id + (:epoch - 1) * nrow(book_df))
    user_book_df = @chain book_df begin
        crossjoin(user_attributes)
    end

    user_book_df = @chain user_book_df begin
        @transform(
            :user_book_utility = :topicality * :user_utility_weight_quality + :quality * :user_utility_weight_topicality,
            # X% of new books are 'pre-read' by users
            :observed = rand(Bernoulli(pct_pre_read))
                   )
    end

    return user_book_df
end

# TODO: remove line, just for testing
new_data = transform_data(book_attributes)
```

## Define Machine Learning Model

```julia
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
        training_data = copy(training_data)
        append!(training_data, new_data, promote=true)

        n_users = maximum(training_data[!, :user_id])
        n_books =maximum(training_data[!, :book_id])
        # Drop unobserved outcomes
        training_data = @chain training_data @subset(:observed == true)
            

        data = convert_dataframe_to_recommender(training_data, n_users, n_books)
        recommender = SVD(data, 10)
        build!(recommender)

        return recommender
end

fit_model(eachrow(book_sim_description)[1], training_data, new_data)
```

```julia
# Skip using this for now, but could be useful in a real study...
function summarize_model(epoch_parameters::DataFrameRow, model, simulation_data::DataFrame, new_data::DataFrame)
    DataFrame(:epoch => [epoch_parameters.epoch])
end
```


```julia
function choose_observations(epoch_parameters::DataFrameRow, recommender, new_data::DataFrame, simulation_data::DataFrame)
    # Select 'unobserved' datapoints to observe
    append!(simulation_data, new_data, promote=true)

    # Each user gets to read an additional book!
    for user_id in unique((@chain simulation_data @subset(!:observed) _[!, :user_id]))
        user_prediction = recommend(recommender, user_id, 1, (@chain simulation_data @subset(!:observed & (:user_id == user_id)) @select(:book_id) unique _[!, :book_id]))
        best_book = user_prediction[1][1]
        best_book_score = user_prediction[1][2]
        # if best_book_score > 0 # If utility is positive, then flip to 'observed'
        #     simulation_data[(simulation_data[!, :user_id] .== user_id) .& (simulation_data[!, :book_id] .== best_book), :observed] = true
        # end
    end

    return simulation_data
end	
```


## Put it all together and run the simulation


```julia
ips = IteratedProcessSimulation(book_dgp, book_sim_description, transform_data, fit_model, summarize_model, choose_observations)

simulation_data, model_summary, model_objects = run_simulation(ips)

# TODO: for debugging, remove
user_id = 1
recommender = model_objects[36]
```