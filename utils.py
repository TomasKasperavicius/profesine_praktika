from dependencies import np, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, pd, plt, os, set_random_seed, MinMaxScaler, TimeseriesGenerator, r2_score
from constants import SLIDING_WINDOW_SIZE, LAG_COUNT, FORECASTING_PERIOD, SEED, TEST_SIZE_SAMPLES_ARIMA, INTERPOLATION_SIZE


def interpolate_data(df, num_interpolations, endpoint=False):
    interpolated_values = []

    for i in range(len(df) - 1):
        interpolated = np.linspace(
            df[i], df[i + 1], num=num_interpolations, endpoint=endpoint)
        interpolated_values.extend(interpolated)

    return np.array(interpolated_values)


def preprocess_data(data):
    X = []
    Y = []
    for i in range(0, len(data)-LAG_COUNT, SLIDING_WINDOW_SIZE):
        X.append(data[i:i + LAG_COUNT])
        Y.append(data[i + LAG_COUNT:i + LAG_COUNT+FORECASTING_PERIOD])

    X = np.array(X).reshape(-1, LAG_COUNT)
    Y = np.array(Y).reshape(-1, FORECASTING_PERIOD)
    return X, Y


def evaluate_performance(forecast, real):
    mae = mean_absolute_error(real, forecast)
    rmse = np.sqrt(mean_squared_error(real, forecast))
    mape = mean_absolute_percentage_error(real, forecast)
    r2 = r2_score(real, forecast)
    print(
        f"Mean absolute error: {mae}, Root mean squared error: {rmse}, Mean absolute percentage error: {mape}, R squared: {r2}")
    return mae, rmse, mape, r2


def read_data(file_name):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep=',')
        df['Laikotarpis'] = pd.to_datetime(df['Laikotarpis'])
        df.set_index('Laikotarpis', inplace=True)
        df.index.freq = 'MS'
        return df
    else:
        print(f"The file '{file_name}' does not exist.")


def plot_error_values(error_values_sarima, error_values_svr, error_values_lstm, error_values_sarima_svr):
    error_values = [error_values_svr, error_values_sarima,
                    error_values_lstm, error_values_sarima_svr]
    model_names = ['SVR', 'ARIMA', 'LSTM', 'ARIMA+SVR']
    error_names = ['RMSE', 'MAE', 'MAPE']
    titles = ['Vidutinė kvadratinė paklaida', 'Vidutinė absoliuti paklaida',
              'Vidutinė absoliuti procentinė paklaida']
    colors = ['green', 'orange', 'blue', 'red']
    _, axs = plt.subplots(3, 1, figsize=(14, 7))
    for k, error_name in enumerate(error_names):
        all_errors = []
        for i, error_value in enumerate(error_values):
            all_errors.append(error_value[f'{error_name}_train'])
            axs[k].barh(
                i, [error_value[f'{error_name}_train']], color=colors[i])
            axs[k].set_yticks(range(len(model_names)))
            axs[k].set_yticklabels(model_names, fontsize=14)
            axs[k].set_title(f'{titles[k]}', fontsize=14)
            axs[k].tick_params(axis='x', which='both', labelsize=12)

        for j, value in enumerate(all_errors):
            axs[k].text(value, j, f'{value:.2f}', ha='left',
                        va='center', color='black', fontsize=14)
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.suptitle('Mokymo duomenų aibė', fontsize=14)
    plt.tight_layout()
    plt.show()
    _, axs = plt.subplots(3, 1, figsize=(14, 7))
    for k, error_name in enumerate(error_names):
        all_errors = []
        for i, error_value in enumerate(error_values):
            all_errors.append(error_value[f'{error_name}_test'])
            axs[k].barh(
                i, [error_value[f'{error_name}_test']], color=colors[i])
            axs[k].set_yticks(range(len(model_names)))
            axs[k].set_yticklabels(model_names, fontsize=14)
            axs[k].set_title(f'{titles[k]}', fontsize=14)
            axs[k].tick_params(axis='x', which='both', labelsize=12)

        for j, value in enumerate(all_errors):
            axs[k].text(value, j, f'{value:.2f}', ha='left',
                        va='center', color='black', fontsize=14)
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.suptitle('Testavimo duomenų aibė', fontsize=14)
    plt.tight_layout()
    plt.show()


def undo_interpolation(df, INTERPOLATION_SIZE):
    array = []
    for i in range(0, len(df), INTERPOLATION_SIZE):
        array.append(df[i])
    return array


def plot_results(original, forecasts, labels, title, ylabel, dates):
    plt.plot(dates, original,
             label='Originalios reikšmės', color="#800080", marker='o')
    plt.xticks(dates, [date.strftime('%Y-%m-%d')
                       for date in dates])
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel('Laikotarpis', fontsize=16)
    for i in range(len(forecasts)):
        plt.plot(dates, np.array(forecasts[i]).flatten(), label=labels[i],
                 linestyle='--', color=plt.cm.tab10(i), marker='o')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.grid()
    plt.show()
