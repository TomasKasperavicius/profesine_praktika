from dependencies import np, pd, plt
from utils import read_data, plot_results, interpolate_data

from svr import perform_SVR
from arima import perform_ARIMA
from lstm import perform_LSTM
from constants import *

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    label = 'Kiekis, vnt'

    df_snow = read_data('sniego_valytuvai.csv')
    df_cheese = read_data('suriai.csv')
    df_notebooks = read_data('sasiuviniai.csv')

    datasets = [[df_snow, 'Kiekis, vnt', 'Sniego valytuvų eksportas iš Lietuvos į Latviją',

                 'snow_cleaners'],
                [df_notebooks, 'Kiekis, kg', 'Sąsiuvinių eksportas iš Lietuvos į Latviją',

                 'notebooks'],
                [df_cheese, 'Kiekis, kg', 'Sūrių eksportas iš Lietuvos į Latviją',

                 'cheese']]
    for i in datasets:
        df = i[0]
        df = np.array(df).flatten()

        test_forecasts = []
        train_forecasts = []

        print("---ARIMA---")
        arima_forecast_train, arima_forecast_test, error_values_sarima = perform_ARIMA(
            df, i[1], [(35, 0, 1), (0, 0, 0, 12)])

        test_forecasts.append([
            arima_forecast_test])
        train_forecasts.append([arima_forecast_train])

        print("---SVR-----")
        train_forecast, test_forecast, error_values_svr = perform_SVR(
            df, best_params={'C': 0.9, 'degree': 5, 'epsilon': 0.01, 'gamma': 0.05, 'kernel': 'linear'})

        test_forecasts.append([test_forecast])
        train_forecasts.append([train_forecast])

        print("---LSTM----")
        train_forecast, test_forecast, error_values_lstm = perform_LSTM(
            df, i[3],
            use_model=False, parameters={'activation_function': 'tanh', 'batch_size': 1, 'dropout_rate': 0.1, 'epochs': 100, 'learning_rate': 0.05, 'num_layers': 0, 'units': 12})

        test_forecasts.append([test_forecast])
        train_forecasts.append([train_forecast])

        print("---ARIMA+SVR-----")
        arima_forecasts = np.concatenate(
            (arima_forecast_train, arima_forecast_test))
        train_forecast, test_forecast, error_values_sarima_svr = perform_SVR(
            df, arima_forecasts, arima_forecasts=arima_forecasts, best_params={'C': 1.1, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear'})

        test_forecasts.append([test_forecast])
        train_forecasts.append([train_forecast])

        test = df[-TEST_SIZE_SAMPLES_ARIMA:]
        train = df[:-TEST_SIZE_SAMPLES_ARIMA]

        date_labels = pd.date_range(
            start='2022-01-01', end='2022-12-01', freq='MS')
        plot_results(test, test_forecasts, ['SARIMA', 'SVR', 'LSTM', 'SARIMA+SVR'],
                     'Sąsiuvinių eksportas iš Lietuvos į Latviją', 'Kiekis, kg', date_labels)
