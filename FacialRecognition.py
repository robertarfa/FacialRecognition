
# import standard dependecies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import tarfile

# import tensorflow dependecies https://www.tensorflow.org/guide/keras/functional_api
# two images and figure out if the two images are the same

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# import uuid library to generate unique image names
import uuid

# avoid out of memory erros by setting GPU Memory Consuption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setup paths
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negative")
ANC_PATH = os.path.join("data", "anchor")

# Make the directories
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# http://vis-www.cs.umass.edu/lfw/

# Uncompress Tar GZ Labelled Faces in the Wild Dataset - Has to be in the same folder of the rest of the code

#!tar -xf lfw.tgz

# tf = tarfile.open("lfw.tgz")

# tf.extractall()

# Move LFW Images to the following repository data/negative

# for directory in os.listdir("lfw"):
#     for file in os.listdir(os.path.join("lfw", directory)):
#         EX_PATH = os.path.join("lfw", directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)

# Establish a connection to the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Cut down frame to 250x250px
    frame = frame[120 : 120 + 250, 200 : 200 + 250, :]
    
    # collect anchors
    if cv2.waitKey(1) & 0xFF == ord("a"):
        pass
    
    # collect positives
    if cv2.waitKey(1) & 0xFF == ord("p"):
        pass
    
    # Show image back to screen
    cv2.imshow("Image Collection", frame)

    # breaking gracefully
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()

plt.imshow(frame)
plt.show()

# print(frame.shape)
