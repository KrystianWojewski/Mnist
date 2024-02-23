import os

import matplotlib.pyplot as plt

# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2     # python -m pip install opencv-python
from keras.datasets import mnist

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model filename
MODEL = 'model.h5'

if __name__ == "__main__":
    
    # Load trained model
    model = tf.keras.models.load_model(MODEL)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for i in range (0, len(x_test)):
        
        # Load the image
        # image = cv2.imread(TEST_DATASET_DIR + os.path.sep + image_name, cv2.IMREAD_GRAYSCALE)
        image = x_test[i]

        # Pre-process the image for classification
        image_data = image.astype('float32') / 255
        image_data = tf.keras.preprocessing.image.img_to_array(image_data)
        # Expand dimension (28,28,1) -> (1,28,28,1)
        image_data = np.expand_dims(image_data, axis=0)
        
        # Classify the input image
        prediction = model.predict(image_data)
        
        # Find the winner class and the probability
        winner_class = np.argmax(prediction)
        winner_probability = np.max(prediction)*100

        second_class = np.argsort(np.max(prediction, axis=0))[-2]
        second_probability = np.unique(prediction)[-2]*100

        third_class = np.argsort(np.max(prediction, axis=0))[-3]
        third_probability = np.unique(prediction)[-3] * 100

        # Build the text label
        label = f"prediction = {winner_class} ({winner_probability:.2f}%)"
        label2 = f"prediction 2 = {second_class} ({second_probability:.2f}%)"
        label3 = f"prediction 3 = {third_class} ({third_probability:.2f}%)"

        # Draw the label on the image
        output_image = cv2.resize(image, (500,500))
        output_image = cv2.putText(output_image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        if (second_probability > 1) :
            output_image = cv2.putText(output_image, label2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
            if (third_probability > 1):
                output_image = cv2.putText(output_image, label3, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)

        # Show the output image        
        cv2.imshow("Output", output_image)
        
        # Break on 'q' pressed, continue on the other key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
