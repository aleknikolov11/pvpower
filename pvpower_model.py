from __future__ import absolute_import, division, print_function, unicode_literals

import random
import os, sys
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def args_parser():
    parser = argparse.ArgumentParser(description='Train a model to predict pv power generation')
    parser.add_argument('-forecast', dest='forecast_csv', default='', required=False, help='Provide a forecast dataset for model predictions')

    return parser.parse_args()

# Normalization function
def norm(x, train_stats):
    return(x - train_stats['mean']) / train_stats['std']

if __name__ == "__main__":

    # Get script arguments
    args = args_parser()

    
    # Load dataset
    dataset = pd.read_csv('PV_training_dataset.csv')
    dataset = dataset.dropna()

    # Clear dataset
    dataset.pop('period_end')
    dataset.pop('period')
    dataset.pop('capacity')

    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
             
    # Separate dataset into training and test datasets        
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Get statistical information of training dataset
    train_stats = train_dataset.describe()
    train_stats.pop('pv_estimate')
    train_stats = train_stats.transpose()

    # Get training and test labels
    train_labels = train_dataset.pop('pv_estimate')
    test_labels = test_dataset.pop('pv_estimate')

    # Normalize the datasets
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    #Check if there is a saved model
    if(not os.path.exists('pvpower_model.h5')):

        # Define the model
        model = keras.Sequential([
            layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=[len(train_dataset.keys())]),
            layers.Dropout(0.5),
            layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=[64]),
            layers.Dense(1)
        ])

        # Define a decaying learning rate
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=len(normed_train_data),
            decay_rate=1,
            staircase=False)
            
        # Define the optimizer
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        # Compile the model
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        example_batch = normed_train_data[:10]
        example_result = model.predict(example_batch)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # Train the model
        EPOCHS = 200
        history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=2, callbacks=[early_stop])

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        # Evaluate the model and get predictions
        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
        test_predictions = model.predict(normed_test_data).flatten()

        # Scatter plot of test labels vs test predictions
        a = plt.axes(aspect='equal')
        plt.figure(1)
        plt.subplot(311)
        plt.scatter(test_labels, test_predictions)
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        lims = [0, 10]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)

        # Plot of Validation Loss
        plt.subplot(312)
        plt.plot(history.history['val_loss'])

        # Histogram of error of predictions
        error = test_predictions - test_labels
        plt.subplot(313)
        plt.xlabel("Prediction Error [MPG]")
        _ = plt.ylabel("Count")
        plt.hist(error, bins = 25)
        plt.plot()
        plt.show()

        #Save the model
        model.save('pvpower_model.h5')
    
    # Load model
    model = tf.keras.models.load_model('pvpower_model.h5')
    
    forecast_dataset = args.forecast_csv if(args.forecast_csv) else 'PV_forecast_dataset.csv'

    try:
        # Load forecast dataset
        raw_dataset = pd.read_csv('PV_forecast_dataset.csv')
        raw_dataset = raw_dataset.dropna()
        dataset = raw_dataset.copy()
        
        # Remove pv estimates from dataset
        dataset.pop('pv_estimate')
        
        dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
        
        normed_dataset = norm(dataset, train_stats)
        
        # Get predictions
        predictions = model.predict(normed_dataset).flatten()
        predictions = predictions.tolist()
        cleaned_predictions = list()
        
        # Remove very small or negative values
        for prediction in predictions:
            if(prediction < 0.005):
                prediction = 0.0;
            cleaned_predictions.append(prediction)
                
        #Write predictions in a csv file
        predictions_row = pd.DataFrame({'pv_predictions': cleaned_predictions})
        predictions_row.to_csv('model_predictions.csv')

    except OSError as err:
        print('Error: {0}'.format(err))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
