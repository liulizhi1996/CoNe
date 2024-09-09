from sklearn.metrics import root_mean_squared_error


def evaluate_rmse(test_pred, test_true):
    test_rmse = root_mean_squared_error(test_true, test_pred)
    return test_rmse
