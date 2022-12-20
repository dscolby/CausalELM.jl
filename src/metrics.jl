module Metrics

export mse, mae

function mse(y_actual, y_pred) 
    @assert length(y_actual) == length(y_pred) "y_actual and y_pred must be the same length"
    return @fastmath sum((y_actual - y_pred).^2) / length(y_actual)
end

function mae(y_actual, y_pred) 
    @assert length(y_actual) == length(y_pred) "y_actual and y_pred must be the same length"
    return @fastmath sum(abs.(y_actual .- y_pred)) / length(y_actual)
end

function confusion_matrix(y_actual, y_pred, threshold=0.5)
    
end
end