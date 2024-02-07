from sklearn.metrics import mean_squared_error

def metrics(y, ypred, y_preprocessor=None):
    """Evaluates metrics.
    
    Args:
        y:
        ypred:
        y_scaler:
        
    Returns:
        mse: mean squared error, float scaler.
        rmse: root mean squared error, float scaler.
    """
    if y_preprocessor is not None:
        y = y_preprocessor.inverse_transform(y)
        ypred = y_preprocessor.inverse_transform(ypred)
    
    mse = mean_squared_error(y, ypred)
    rmse = mean_squared_error(y, ypred, squared=False)
    return mse, rmse
