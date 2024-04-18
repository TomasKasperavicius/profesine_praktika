from dependencies import set_random_seed, EarlyStopping, np, tf, MinMaxScaler,  Dense,  Sequential, ModelCheckpoint, os, Adam, LSTM, TimeSeriesSplit, ParameterGrid, Pool
from utils import evaluate_performance, mean_absolute_error, preprocess_data
from constants import FORECASTING_PERIOD, LAG_COUNT, SEED, TEST_SIZE_SAMPLES, CROSS_VALIDATION_SPLITS
import warnings


def create_lstm_model(X, units=24, activation_function='relu', num_layers=1, dropout_rate=0, learning_rate=0.001, compile=True):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(units, activation=activation_function, dropout=dropout_rate,
                      return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            continue
        model.add(LSTM(units, activation=activation_function,
                  dropout=dropout_rate, return_sequences=True))
    model.add(LSTM(units, activation=activation_function,
              dropout=dropout_rate, input_shape=(X.shape[1], X.shape[2])))

    model.add(Dense(units=FORECASTING_PERIOD, activation='tanh'))
    if compile:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


def find_best_LSTM_architecture(parameter_combinations, X, Y, scaler):
    tf.get_logger().setLevel('ERROR')
    warnings.filterwarnings("ignore")
    set_random_seed(SEED)

    best_mae = float("inf")
    best_params = None
    tscv = TimeSeriesSplit(n_splits=CROSS_VALIDATION_SPLITS, test_size=1)

    for params in parameter_combinations:

        fold_mae = []

        lstm = create_lstm_model(X, units=params['units'], activation_function=params['activation_function'],
                                 num_layers=params['num_layers'], dropout_rate=params['dropout_rate'], learning_rate=params['learning_rate'])
        for train_index, test_index in tscv.split(X):
            train_X, test_X = X[train_index], X[test_index]
            train_Y, test_Y = Y[train_index], Y[test_index]
            train_X = train_X.reshape(
                train_X.shape[0], train_X.shape[1], 1)
            test_X = test_X.reshape(
                test_X.shape[0], test_X.shape[1], 1)
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True)
            lstm.fit(train_X, train_Y,
                     epochs=params['epochs'],
                     batch_size=params['batch_size'],
                     workers=os.cpu_count(), use_multiprocessing=True, verbose=0, callbacks=[early_stopping])
            test_forecast = lstm.predict(test_X, verbose=0)
            test_forecast = scaler.inverse_transform(
                test_forecast.reshape(-1, 1))
            mae = mean_absolute_error(scaler.inverse_transform(
                test_Y.reshape(-1, 1)).reshape(-1, 1), test_forecast.reshape(-1, 1))
            fold_mae.append(mae)
        avg_mae = sum(fold_mae) / len(fold_mae)

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params
    return [best_params, best_mae]


def perform_LSTM(data, original_data, model_name, parameters=None, use_model=False, exogenous_features=None):
    set_random_seed(SEED)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    exogenous_scaler = MinMaxScaler(feature_range=(-1, 1))

    data = scaler.fit_transform(data.reshape(-1, 1))
    X, Y = preprocess_data(data)

    if exogenous_features is not None:
        X = X.reshape(-1, 1)
        for i in range(len(exogenous_features)):
            scaled_feature = exogenous_scaler.fit_transform(
                exogenous_features[i].reshape(-1, 1))
            feature, _ = preprocess_data(scaled_feature)
            X = np.hstack((X, feature.reshape(-1, 1)))
        X = X.reshape(-1, LAG_COUNT, len(exogenous_features)+1)
        train_X, train_Y = X[:-TEST_SIZE_SAMPLES], Y[:-TEST_SIZE_SAMPLES]
        test_X, test_Y = X[-TEST_SIZE_SAMPLES:], Y[-TEST_SIZE_SAMPLES:]
    else:
        train_X, train_Y = X[:-TEST_SIZE_SAMPLES], Y[:-TEST_SIZE_SAMPLES]
        test_X, test_Y = X[-TEST_SIZE_SAMPLES:], Y[-TEST_SIZE_SAMPLES:]
        train_X = train_X.reshape(
            train_X.shape[0], train_X.shape[1], 1)
        test_X = test_X.reshape(
            test_X.shape[0], test_X.shape[1], 1)

    if parameters == None:
        param_grid = {
            'num_layers': [0, 1, 2],
            'units': [6, 12, 24, 36],
            'dropout_rate': [0.1],
            'activation_function': ['relu', 'tanh'],
            'learning_rate': [0.1, 0.01, 0.01, 0.001],
            'batch_size': [1, 2],
            'epochs': [10, 20, 30, 40],
        }

        param_combinations = list(ParameterGrid(param_grid))

        batch_size = len(param_combinations)//os.cpu_count()  # 8 cores
        with Pool() as pool:
            batches = [param_combinations[i:i+batch_size]
                       for i in range(0, len(param_combinations), batch_size)]
            results = pool.starmap(find_best_LSTM_architecture, [(
                batch, train_X, train_Y, scaler) for batch in batches])

        parameters = min(results, key=lambda x: x[1])[0]
        print(f'Best parameters for LSTM: {parameters}')

    model = create_lstm_model(train_X, units=parameters['units'], activation_function=parameters['activation_function'],
                              num_layers=parameters['num_layers'], dropout_rate=parameters['dropout_rate'], learning_rate=parameters['learning_rate'])
    checkpoint = ModelCheckpoint(
        f'lstm_model_{model_name}.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=0)
    model.compile(optimizer=Adam(
        learning_rate=parameters['learning_rate']), loss='mean_squared_error')

    if use_model and os.path.exists(f'lstm_model_{model_name}.h5'):
        model.load_weights(f'lstm_model_{model_name}.h5')
    else:
        model.fit(train_X[:-TEST_SIZE_SAMPLES],
                  train_Y[:-TEST_SIZE_SAMPLES],
                  epochs=parameters['epochs'],
                  batch_size=parameters['batch_size'],
                  validation_data=(
                      train_X[-TEST_SIZE_SAMPLES:], train_Y[-TEST_SIZE_SAMPLES:]),
                  callbacks=[checkpoint],
                  use_multiprocessing=True
                  )
        model.load_weights(f'lstm_model_{model_name}.h5')

    train_forecast = model.predict(train_X, verbose=0)
    train_forecast = scaler.inverse_transform(train_forecast)

    test_forecast = model.predict(test_X, verbose=0)
    test_forecast = scaler.inverse_transform(test_forecast)

    error_values_lstm = {}
    print("Training data evaluation:")
    error_values_lstm["MAE_train"], error_values_lstm["RMSE_train"], error_values_lstm["MAPE_train"], _ = evaluate_performance(
        train_forecast.reshape(-1, 1), scaler.inverse_transform(train_Y.reshape(-1, 1)))
    print("Testing data evaluation:")
    error_values_lstm["MAE_test"], error_values_lstm["RMSE_test"], error_values_lstm["MAPE_test"], _ = evaluate_performance(
        test_forecast.reshape(-1, 1), scaler.inverse_transform(test_Y.reshape(-1, 1)))

    return train_forecast.flatten(), test_forecast.flatten(), error_values_lstm
