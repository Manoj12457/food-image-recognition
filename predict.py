import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('food_classifier_model.h5')  # Replace with the path to your trained model

# Load an image for prediction
img_path = 'path/to/your/test/image.jpg'  # Replace with the path to your test image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Make predictions
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)
confidence = predictions[0][predicted_class_index] * 100

# Get class label based on index
class_names = ['apple', 'banana', 'watermelon', 'orange', '...', '...', '...', '...']  # Replace with your class names
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
