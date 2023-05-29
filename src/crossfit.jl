module Crossfit
using ..Estimators: DoublyRobust, dre_ate!, dre_first_stage!

"""
    traintest(X, Y, folds)

Create a train-test split.

If an iteration is specified, the train test split will be treated as time series/panel
data.

Examples
```julia-repl
julia> xtrain, ytrain, xtest, ytest = traintest(zeros(20, 2), zeros(20), 5)
```
"""
function generatefolds(X::Array{Float64}, Y::Array{Float64}, folds::Int64)
    msg = "the number of folds must be less than the number of observations"
    n = size(Y, 1)
    
    if folds >= n
        throw(ArgumentError(msg))
    end

    # Shuffled indices for all the data and the size of the training data
    idx, train_size = shuffle(1:n), @fastmath n*(folds-1)/folds

    # 1:train_size are the indices from randomly shuffled incies to use from training data
    # train_size + 1:n are the indices from randomly shuffled indices to use for testing
    train_idx, test_idx = view(idx, 1:floor(Int, train_size)), 
        view(idx, (floor(Int, train_size)+1):n)

    xtrain, ytrain, xtest, ytest = X[train_idx, :], Y[train_idx, :], 
        X[test_idx, :], Y[test_idx, :]
        
    return xtrain, ytrain, xtest, ytest
end

function crossfit(DRE::DoublyRobust, folds::Int64)
end

end
