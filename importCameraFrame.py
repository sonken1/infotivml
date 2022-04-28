import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)
#
# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False
#
# while rval:
#     cv2.imshow("preview", frame)
#     rval, frame = vc.read()
#     key = cv2.waitKey(20)
#     if key == 27: # exit on ESC
#         break
#
# vc.release()
# cv2.destroyWindow("preview")

cam = cv2.VideoCapture(0) # 0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
model = keras.models.load_model('mnist_cnn.h5')

img_size = (28, 28)
path = r"C:\Users\elias\Documents\codealong\presentationmaterial\handwritten-2.jpg"
imgRef = load_img(path, target_size=img_size, color_mode="grayscale")
imgRef = np.expand_dims(imgRef, 2)
imgRef = np.expand_dims(imgRef, 0)
pred = model.predict(imgRef).argmax()
print("", pred)

while True:
    ## read frames
    ret, img = cam.read()
    ## predict yolo
    # img, preds = yolo.detectCustomObjectsFromImage(input_image=img,
    #                   custom_objects=None, input_type="array",
    #                   output_type="array",
    #                   minimum_percentage_probability=70,
    #                   display_percentage_probability=False,
    #                   display_object_name=True)
    ## display predictions
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    grayImg = np.expand_dims(gray, -1)
    grayNew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("", gray)
    #cv2.imshow("", img)
    ## press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
## close camera
cam.release()
cv2.destroyAllWindows()