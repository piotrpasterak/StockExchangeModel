from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def datapreparation(company_input_data):
    company_val_data = company_input_data.iloc[:, 1:2].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    apple_training_scaled = scaler.fit_transform(company_val_data)
    features= []
    labels = []
    for i in range(60, company_val_data.size):
        features.append(apple_training_scaled[i - 60:i, 0])
        labels.append(apple_training_scaled[i, 0])
    features, labels = np.array(features), np.array(labels)
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))

    return features, labels, scaler;


def modeltraining(features_set, labels):

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
    model.fit(features_set, labels, epochs=100, batch_size=32)

    return model


def prediction(model_obj, scaler, apple_testing_complete_data, company_data):
    apple_total = pd.concat((company_data['Open'], apple_testing_complete_data['Open']), axis=0)
    test_inputs = apple_total[len(apple_total) - len(apple_testing_complete_data) - 60:].values
    test_inputs = test_inputs.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)
    test_features = []
    for i in range(60, 80):
        test_features.append(test_inputs[i - 60:i, 0])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    predictions = model_obj.predict(test_features)
    predictions = scaler.inverse_transform(predictions)

    return predictions


def trendmodelling(train_data, test_data, label):
    company_input_data = pd.read_csv(train_data)
    data, labels_set, scaler_obj = datapreparation(company_input_data)
    model = modeltraining(data, labels_set)
    apple_testing_complete = pd.read_csv(test_data)
    apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values
    predictions_data = prediction(model, scaler_obj, apple_testing_complete, company_input_data)
    plt.figure(figsize=(10, 6))
    plt.plot(apple_testing_processed, color='blue', label='Actual Stock Price')
    plt.plot(predictions_data, color='red', label='Predicted Stock Price')
    plt.title(label + ' Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    se_data = {'Apple': (r'data/apple/input/AAPL.csv', r'data/apple/test/AAPL_test.csv'),
               'Google': (r'data/google/input/GOOG.csv', r'data/google/test/GOOG_test.csv')}

    for label, cmp_data in se_data.items():
        trendmodelling(cmp_data[0], cmp_data[1], label)

