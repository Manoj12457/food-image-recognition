from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Upload the image file
uploaded = files.upload()

# Read the uploaded image
img_path = list(uploaded.keys())[0]
image = cv2.imread(img_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask image
mask = np.zeros_like(image)

# Draw contours on the mask
cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)

# Apply the mask to the original image
segmented_image = cv2.bitwise_and(image, mask)

# Display the original and segmented images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
axs[1].set_title('Segmented Image')
axs[1].axis('off')
plt.show()