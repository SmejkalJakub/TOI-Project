import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

dataset = pd.read_csv("data.csv")

# create column for humidity shifted by 1
dataset['recnt_Humidity'] = dataset['Humidity'].shift(1)

# remove first row of dataset
dataset = dataset.drop(dataset.index[0])

# rename columns
dataset = dataset.rename(columns={'Humidity': 'Target_Humidity', 'Temperature': 'recnt_Temperature'})

# switch columns
dataset = dataset[['UTC', 'recnt_Humidity', 'recnt_Temperature', 'Target_Humidity']]

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

dataset = dataset[['year', 'month', 'day', 'hour', 'minute', 'second', 'recnt_Humidity', 'recnt_Temperature', 'Target_Humidity']]


print(dataset.head(5))

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1:].values

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
model.fit(x, y, epochs=5000, batch_size=16)
model.save('Humidity_predictor_model')

load_model = tf.keras.models.load_model('Humidity_predictor_model')
converter = tf.lite.TFLiteConverter.from_keras_model(load_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("humidity_predictor.tflite", "wb").write(tflite_model)