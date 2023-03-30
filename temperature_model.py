import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

dataset = pd.read_csv("data.csv")

# create column for humidity shifted by 1
dataset['recnt_Temperature'] = dataset['Temperature'].shift(1)

# remove first row of dataset
dataset = dataset.drop(dataset.index[0])

# rename columns
dataset = dataset.rename(columns={'Humidity': 'recnt_Humidity', 'Temperature': 'Target_Temperature'})

# switch columns
dataset = dataset[['UTC', 'recnt_Temperature', 'recnt_Humidity', 'Target_Temperature']]

# convert UTC to datetime
dataset['UTC'] = pd.to_datetime(dataset['UTC'], unit='ms')

# Split the date into year, month, day, hour, minute, second columns
dataset['year'] = dataset['UTC'].dt.year
dataset['month'] = dataset['UTC'].dt.month
dataset['day'] = dataset['UTC'].dt.day
dataset['hour'] = dataset['UTC'].dt.hour
dataset['minute'] = dataset['UTC'].dt.minute
dataset['second'] = dataset['UTC'].dt.second

# drop UTC column
dataset = dataset.drop(['UTC'], axis=1)

dataset = dataset[['year', 'month', 'day', 'hour', 'minute', 'second', 'recnt_Temperature', 'recnt_Humidity', 'Target_Temperature']]

print(dataset.head(5))

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1:].values

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
model.fit(x, y, epochs=5000, batch_size=16)
model.save('Temperature_predictor_model')

load_model = tf.keras.models.load_model('Temperature_predictor_model')
converter = tf.lite.TFLiteConverter.from_keras_model(load_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("Temperature_predictor.tflite", "wb").write(tflite_model)

