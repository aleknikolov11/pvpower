# PV Power Forecast #

A script that trains model to forecast pv power production, based on meteorological data. The script uses the following libraries:

  * Tensorflow
  * Numpy
  * MatPlotLib
  * Pandas

The model is trained on data from each world capital, where the following parameters are provided:

  * capacity = 10MW
  * tilt = 23 degrees
  * azimuth = 0 degrees
  * loss factor = 0.9

## Run the model ##

To run the model, run the pvpower_model script. It will load a model, if one exists, or otherwise train and save a new model.

**Example**

	python pvpower_model.py -forecast "sample_forecast_dataset.csv"

The '-forecast' argument is optional, and if not provided, the script will load the sample PV_forecast_dataset.csv, provided in the project. (NOTE: The provided dataset should contain the same columns as PV_forecast_dataset)

The results are generated in a file named model_predictions.csv



