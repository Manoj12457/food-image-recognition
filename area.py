from google.colab import files
import cv2

# Upload the image file
uploaded = files.upload()

# Read the uploaded image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour (assuming it represents the object)
largest_contour = max(contours, key=cv2.contourArea)

# Calculate the area of the largest contour
area = cv2.contourArea(largest_contour)

# Display the area
print("Area of the object:", area)