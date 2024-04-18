from dependencies import os, np, SVR, MultiOutputRegressor, mean_absolute_error, MinMaxScaler, TimeSeriesSplit, ParameterGrid, Pool
from constants import SLIDING_WINDOW_SIZE, LAG_COUNT, FORECASTING_PERIOD, SEED, TEST_SIZE_SAMPLES, CROSS_VALIDATION_SPLITS
from utils import evaluate_performance, preprocess_data
import warnings


def find_best_SVR_parameters(parameter_combinations, X, Y, scaler):

    warnings.filterwarnings("ignore")
    best_mae = float("inf")
    best_params = None

    tscv = TimeSeriesSplit(n_splits=CROSS_VALIDATION_SPLITS, test_size=1)

    for params in parameter_combinations:
        svr = SVR(**params)
        svr = MultiOutputRegressor(svr, n_jobs=-1)

        fold_mae = []

        for train_index, test_index in tscv.split(X):
            train_X, test_X = X[train_index], X[test_index]
            train_Y, test_Y = Y[train_index], Y[test_index]

            svr.fit(train_X, train_Y)

            test_forecast = svr.predict(test_X)
            test_forecast = scaler.inverse_transform(
                test_forecast.reshape(-1, 1)).reshape(-1, 1)
            mae = mean_absolute_error(scaler.inverse_transform(
                test_Y.reshape(-1, 1)).reshape(-1, 1), test_forecast)
            fold_mae.append(mae)
        avg_mae = sum(fold_mae) / len(fold_mae)

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params
    return [best_params, best_mae]


def perform_SVR(data, arima_forecasts=None, best_params=None, exogenous_features=None):
    scaler = MinMaxScaler()
    exogenous_scaler = MinMaxScaler()

    if arima_forecasts is not None:
        scaler.fit(np.concatenate(
            (np.array(data).flatten(), arima_forecasts)).reshape(-1, 1))
        scaler.fit(arima_forecasts.reshape(-1, 1))
        data = scaler.transform(data.reshape(-1, 1))
        arima_forecasts = scaler.transform(
            arima_forecasts.reshape(-1, 1)).flatten()
        arima_forecasts_new = []
        for j in range(0, len(arima_forecasts)-LAG_COUNT, SLIDING_WINDOW_SIZE):
            arima_forecasts_new.append(arima_forecasts[j:j + LAG_COUNT])
        arima_forecasts_new = np.array(arima_forecasts_new)
        X, Y = preprocess_data(data)
        X = arima_forecasts_new
        if exogenous_features is not None:
            for i in range(len(exogenous_features)):
                scaled_feature = exogenous_scaler.fit_transform(
                    exogenous_features[i].reshape(-1, 1))
                feature, _ = preprocess_data(scaled_feature)
                X = np.hstack((X, feature))

        train_X, train_Y = X[:-TEST_SIZE_SAMPLES], Y[:-TEST_SIZE_SAMPLES]
        test_X, test_Y = X[-TEST_SIZE_SAMPLES:], Y[-TEST_SIZE_SAMPLES:]
    else:
        data = scaler.fit_transform(data.reshape(-1, 1))
        X, Y = preprocess_data(data)
        if exogenous_features is not None:
            for i in range(len(exogenous_features)):
                scaled_feature = exogenous_scaler.fit_transform(
                    exogenous_features[i].reshape(-1, 1))
                feature, _ = preprocess_data(scaled_feature)
                X = np.hstack((X, feature))
        train_X, train_Y = X[:-TEST_SIZE_SAMPLES], Y[:-TEST_SIZE_SAMPLES]
        test_X, test_Y = X[-TEST_SIZE_SAMPLES:], Y[-TEST_SIZE_SAMPLES:]

    svr = None
    if best_params is None:
        parameter_grid = {"C": np.arange(0.8, 1.2, 0.1),
                          "gamma": [*np.arange(0.05, 5, 0.5), 'auto', 'scale'],
                          "kernel": ['poly', 'rbf', 'linear', 'sigmoid'],
                          "epsilon": np.arange(0.05, 0.5, 0.05),
                          "degree": [2, 3, 4, 5, 6]}
        param_combinations = list(ParameterGrid(parameter_grid))
        batch_size = len(param_combinations)//os.cpu_count()  # 8 cores
        with Pool() as pool:
            batches = [param_combinations[i:i+batch_size]
                       for i in range(0, len(param_combinations), batch_size)]
            results = pool.starmap(find_best_SVR_parameters, [(
                batch, train_X, train_Y, scaler) for batch in batches])
        best_params = min(results, key=lambda x: x[1])[0]
        svr = SVR(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'],
                  epsilon=best_params['epsilon'], degree=best_params['degree'])
        svr = MultiOutputRegressor(svr, n_jobs=-1)
        svr.fit(train_X, train_Y)
        with open('bestparams.txt', 'a') as file:
            file.write(f"best params svr: {best_params}")
    else:
        svr = SVR(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'],
                  epsilon=best_params['epsilon'], degree=best_params['degree'])
        svr = MultiOutputRegressor(svr, n_jobs=-1)
        svr.fit(train_X, train_Y)

    train_forecast = svr.predict(train_X)
    train_forecast = scaler.inverse_transform(
        train_forecast.reshape(-1, 1)).reshape(-1, 1)

    test_forecast = svr.predict(test_X)
    test_forecast = scaler.inverse_transform(
        test_forecast.reshape(-1, 1)).reshape(-1, 1)

    error_values = {}

    print("Training data evaluation:")
    error_values["MAE_train"], error_values["RMSE_train"], error_values["MAPE_train"], _ = evaluate_performance(
        train_forecast, scaler.inverse_transform(train_Y.reshape(-1, 1)).reshape(-1, 1))
    print("Testing data evaluation:")
    error_values["MAE_test"], error_values["RMSE_test"], error_values["MAPE_test"], _ = evaluate_performance(
        test_forecast, scaler.inverse_transform(test_Y.reshape(-1, 1)).reshape(-1, 1))

    return train_forecast.flatten(), test_forecast.flatten(), error_values
