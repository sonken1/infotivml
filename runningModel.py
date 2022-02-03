import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import multilabel_confusion_matrix

model = keras.models.load_model('mnist_cnn.h5')

