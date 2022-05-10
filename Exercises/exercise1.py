from tensorflow.keras import layers, models, metrics, datasets

######### Construct a Neural Network #########
# Keras Sequential Class (for 'simple' networks) rather than Keras Functional (used for more complex architectures)
# "Creating"/setting up the base for a neural network
myFirstModel = models.Sequential()

# To add a layer to the network, you simply call the "add" method of the .Sequential class, and insert the layer.TYPE of choice.
# E.g. adding a fully connected layer (called Dense in Keras/Tensorflow) with two neurons and ReLU activation would look like:
# (NOTE: The first layer requires an input "shape", in order to prepare for the data we'll be giving it)
myFirstModel.add(layers.Dense(2, activation='relu', input_shape=(1,)))

# You can visualize how your network looks like by calling the .summary function:
myFirstModel.summary()

# We can add things the loss functions, optimizer to use and metrics fairly easy:
myFirstModel.compile(loss="mse", optimizer="RMSprop", metrics=[metrics.RootMeanSquaredError()])

# To train the model, we would need to only call the .fit method:
x_train = 1
y_train = 1
myFirstModel.fit(x_train, y_train)

# and to save the model (architecture + weights), we simply do (.h5 is the filetype Tensorflow/Keras) use:
myFirstModel.save('myfirstmodel.h5')


###############################################################################
# Example of a Keras "native" dataset:
(x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=1337)

# input shape should match how x_train looks and output neurons like y_train
input_shape = x_train[0].shape

# Updating our model accordingly:
mySecondModel = models.Sequential()
mySecondModel.add(layers.Dense(2, activation='relu', input_shape=input_shape))
mySecondModel.add(layers.Dense(1, activation='relu'))

# You can visualize how your network looks like by calling the .summary function:
mySecondModel.summary()

# We can add things the loss functions, optimizer to use and metrics fairly easy:
mySecondModel.compile(loss="mse", optimizer="RMSprop", metrics=[metrics.RootMeanSquaredError()])

# To train the model, we would need to only call the .fit method:
mySecondModel.fit(x_train, y_train, batch_size=4)

# and to save the model (architecture + weights), we simply do (.h5 is the filetype Tensorflow/Keras) use:
mySecondModel.save('mySecondModel.h5')
