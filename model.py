from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def data_preparation(company_input_data):
    """
    funkcja wykonująca przygotowanie danych przez normalizacje.
    :param company_input_data: dane giełdowe firmy.
    :return: zwraca tablice x , y i informacje normalizujące
    """
    company_val_data = company_input_data.iloc[:, 1:2].values
    scalar = MinMaxScaler(feature_range=(0, 1))
    apple_training_scaled = scalar.fit_transform(company_val_data)
    features= []
    labels = []
    for i in range(60, company_val_data.size):
        features.append(apple_training_scaled[i - 60:i, 0])
        labels.append(apple_training_scaled[i, 0])
    features, labels = np.array(features), np.array(labels)
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))

    return features, labels, scalar


def model_training(features_set, labels):
    """
    Funkcja wykonujące uczenie modelu. Zbiera tez parametry uczenia w tym walidacyjne.
    :param features_set: tablica wartosci x.
    :param labels: tablica wartosci y.
    :return: zwraca nauczony model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(features_set, labels, epochs=100, batch_size=32, validation_split=0.1,
              callbacks=[tensorboard_callback])

    return model


def prediction(model_obj, scalar, company_testing_complete_data, company_data):
    """
    Funkcja przewidująca notowanie giełdowe.

    :param model_obj: nauczony model.
    :param scalar: informacje o
    :param company_testing_complete_data: dane testowe.
    :param company_data: dane notowań.
    :return: przewidywania.
    """
    test_features = get_test_feature(company_data, company_testing_complete_data, scalar)
    predictions = model_obj.predict(test_features)
    predictions = scalar.inverse_transform(predictions)

    return predictions


def get_test_feature(company_data, company_testing_complete_data, scaler):
    """
    Funkcja przygtoująca dane testowe.
    :param company_data: notowania giedowe.
    :param company_testing_complete_data: dane testowe.
    :param scaler: informacje normalizujące.
    :return: przetworzone dane testowe.
    """
    apple_total = pd.concat((company_data['Open'], company_testing_complete_data['Open']), axis=0)
    test_inputs = apple_total[len(apple_total) - len(company_testing_complete_data) - 60:].values
    test_inputs = test_inputs.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)
    test_features = []
    for i in range(60, 80):
        test_features.append(test_inputs[i - 60:i, 0])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    return test_features


def trend_modelling(train_data, test_data, label):
    """
    Funkcja prezntująca wyniki przewidywań.
    :param train_data: dane wynikajace z przewidywań.
    :param test_data: dane testowe.
    :param label: nazwa firmy.
    """
    company_input_data = pd.read_csv(train_data)
    data, labels_set, scaler_obj = data_preparation(company_input_data)
    apple_testing_complete = pd.read_csv(test_data)
    apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values
    model = model_training(data, labels_set)
    predictions_data = prediction(model, scaler_obj, apple_testing_complete, company_input_data)
    plt.figure(figsize=(10, 6))
    plt.plot(apple_testing_processed, color='blue', label='Rzeczywista cena giełdowa')
    plt.plot(predictions_data, color='red', label='Przewidywana cena giełdowa')
    plt.title(label + ' Przewidywanie trendów giełdowych.')
    plt.xlabel('Data w dniach (Relatywna)')
    plt.ylabel('Cena giełdowa')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """
    Prosty model giełdy papierów wartościowych oparty na LSTM. 
    Praktycznym zastosowaniem dla takiego modelu jest potrzeba przewidywania trendów zmian notowań giełdowych.
    """

    se_data = {'Apple': (r'data/apple/input/AAPL.csv', r'data/apple/test/AAPL_test.csv'),
               'Google': (r'data/google/input/GOOG.csv', r'data/google/test/GOOG_test.csv')}

    for label, cmp_data in se_data.items():
        trend_modelling(cmp_data[0], cmp_data[1], label)

