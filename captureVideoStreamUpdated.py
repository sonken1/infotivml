import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import time

cam = cv2.VideoCapture(0)
model = load_model("./mnist_cnn.h5")
default_img_size = (28, 28)
default_path = "./tempImg.jpg"
default_path_orgSize = "./tempImgOrgSz.jpg"
threshold = 0.1

while True:
    ret, img = cam.read()
    cv2.imshow("", img)
    waitkey = cv2.waitKey(1)
    if (waitkey & 0xFF == ord("q")) or (waitkey == 27):
        break
    if (waitkey & 0xFF == ord("p")) or (waitkey == 32):
        cv2.imwrite(default_path_orgSize, img)
        img = cv2.resize(img, default_img_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(default_path, img)
        time.sleep(1)
        test_image = cv2.imread(default_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.bitwise_not(img_resized)
        pred_img = cv2.blur(img_resized, (2, 3))  # Går att lägga till
        pred_img = np.expand_dims(pred_img, 2)   # arg 1 ska vara img_resized
        pred_img = np.expand_dims(pred_img, 0)
        pred_array = model.predict(pred_img)
        predicted_value = pred_array.argmax()
        if pred_array[0][predicted_value] > threshold:
            print("Value: ", predicted_value)
            loaded_img = cv2.imread(default_path_orgSize)
            cv2.imshow("Predicted Value: {}, Probability: {}".format(predicted_value, pred_array[0][predicted_value]), loaded_img)
            for idx, val in enumerate(pred_array[0]):
                print("Probability of {} = {}".format(idx, val))
        else:
            print("Too low prediction certainty! Array of probs: ", pred_array)


cam.release()
cv2.destroyAllWindows()