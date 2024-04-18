from dependencies import np, product, Pool, os, TimeSeriesSplit, SARIMAX
from utils import evaluate_performance, mean_absolute_error
from constants import TEST_SIZE_SAMPLES_ARIMA, FORECASTING_PERIOD, CROSS_VALIDATION_SPLITS
import warnings


def find_best_ARIMA_parameters(data, params):
    best_mae = float("inf")
    best_params = None
    tscv = TimeSeriesSplit(n_splits=CROSS_VALIDATION_SPLITS,
                           test_size=FORECASTING_PERIOD)
    warnings.filterwarnings("ignore")

    for order, seasonal_order in params:
        fold_mae = []
        for train_index, test_index in tscv.split(data):
            train, test = data[train_index], data[test_index]
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            model.initialize_approximate_diffuse()

            model = model.fit(full_output=0, disp=0)

            test_forecast = model.forecast(len(test))
            test_forecast[np.isnan(test_forecast)] = np.finfo(np.float64).max
            mae = mean_absolute_error(test, test_forecast)
            fold_mae.append(mae)
        avg_mae = sum(fold_mae) / len(fold_mae)

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = [order, seasonal_order]
    return [best_params, best_mae]


def perform_ARIMA(data, params=None, exogenous_features=None):
    order, seasonal_order = None, None

    train, test = data[:-
                       TEST_SIZE_SAMPLES_ARIMA], data[-TEST_SIZE_SAMPLES_ARIMA:]

    if params == None:
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        P_values = range(0, 3)
        D_values = range(0, 2)
        Q_values = range(0, 3)
        param_combinations = list(product(
            product(p_values, d_values, q_values),
            product(P_values, D_values, Q_values, [12])
        ))
        batch_size = len(param_combinations)//os.cpu_count()  # 8 cores
        with Pool() as pool:
            batches = [param_combinations[i:i+batch_size]
                       for i in range(0, len(param_combinations), batch_size)]
            results = pool.starmap(find_best_ARIMA_parameters, [(
                train, batch) for batch in batches])

        order, seasonal_order = min(results, key=lambda x: x[1])[0]
        print(f"Best parameters for SARIMA: {order}, {seasonal_order}")
    else:
        order, seasonal_order = params
    model = None
    exog_train, exog_test = None, None
    if exogenous_features is not None:
        exog = np.vstack(exogenous_features).T
        exog_train, exog_test = exog[:-
                                     TEST_SIZE_SAMPLES_ARIMA], exog[-TEST_SIZE_SAMPLES_ARIMA:]
        model = SARIMAX(endog=train, exog=exog_train,
                        order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        predictions = model_fit.forecast(
            steps=TEST_SIZE_SAMPLES_ARIMA, exog=exog_test)
        train_index = np.arange(len(train))
        train_forecast = model_fit.predict(start=train_index[0],
                                           end=train_index[-1],
                                           typ='levels', exog=exog_train)
    else:
        model = SARIMAX(endog=train, order=order,
                        seasonal_order=seasonal_order)

        model_fit = model.fit()
        predictions = model_fit.forecast(steps=TEST_SIZE_SAMPLES_ARIMA)
        train_index = np.arange(len(train))
        train_forecast = model_fit.predict(start=train_index[0],
                                           end=train_index[-1],
                                           typ='levels')

    error_values = {}
    print("Training data evaluation:")
    error_values["MAE_train"], error_values["RMSE_train"], error_values["MAPE_train"], _ = evaluate_performance(
        train_forecast, train)
    print("Testing data evaluation:")
    error_values["MAE_test"], error_values["RMSE_test"], error_values["MAPE_test"], _ = evaluate_performance(
        predictions, test)

    return np.array(train_forecast), np.array(predictions), error_values
