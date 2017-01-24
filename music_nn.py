from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Poo

# Model parameters for easier access
IN_DIMENSION = 512
N_FEATURES_1 = 64
N_FEATURES_2 = 32
N_FEATURES_3 = 16
OUT_DIMENSION = 128
AFUNC = 'relu'


# Setup data for training
# TODO: ACTUALLY IMPORT DATA FROM MUSIC_DATA SCRIPT!

# Define model
model = Sequential()

model.add(Dense(output_dim=N_FEATURES_1, input_dim=IN_DIMENSION))
model.add(Activation(AFUNC))
model.add(Dense(output_dim=N_FEATURES_2))
model.add(Activation(AFUNC))
model.add(Dense(output_dim=N_FEATURES_3))
model.add(Activation(AFUNC))
model.add(Dense(output_dim=OUT_DIMENSION))
model.add(Activation(AFUNC))


# Specify model training options and fit the model
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=5, batch_size=32)







