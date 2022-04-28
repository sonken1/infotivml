import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
model = load_model("C:/Users/elias/Documents/infotivml/mnist_cnn.h5")
default_img_size = (28, 28)
default_path = r"C:/Users/elias/Documents/infotivml/tempImg.jpg"
threshold = 0.1

while True:
    ret, img = cam.read()
    cv2.imshow("", img)
    waitkey = cv2.waitKey(1)
    if (waitkey & 0xFF == ord("q")) or (waitkey == 27):
        break
    if (waitkey & 0xFF == ord("p")) or (waitkey == 32):
        img = cv2.resize(img, default_img_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(default_path, img)
        time.sleep(1)
        test_image = cv2.imread(default_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.bitwise_not(img_resized)
        # pred_img = cv2.blur(img_resized, (2, 2))  # Går att lägga till
        pred_img = np.expand_dims(img_resized, 2)   # arg 1 ska vara img_resized
        pred_img = np.expand_dims(pred_img, 0)
        pred_array = model.predict(pred_img)
        predicted_value = pred_array.argmax()
        if pred_array[0][predicted_value] > threshold:
            print("Value: ", predicted_value)
            loaded_img = cv2.imread(default_path)
            cv2.imshow("Predicted Value: {}, Probability: {}".format(predicted_value, pred_array[0][predicted_value].round(3)), loaded_img)
            #plt.imshow(pred_img_init)
            #plt.title(["Predicted Value: {}, Probability: {}".format(predicted_value, pred_array[0][predicted_value].round(3))])
            #plt.imshow(pred_img_init/255)
        else:
            print("Array of probs: ", pred_array)


cam.release()
cv2.destroyAllWindows()