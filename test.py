from skimage.feature import hog
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cupy as cp
print(cp.__version__)
# Load and process the image
image = io.imread('data/train_images/000001.png')  # Replace with your image path
gray_image = color.rgb2gray(image)  # Convert to grayscale

# Extract HOG features
features, hog_image = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

# Plot the HOG image (optional)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Features Visualization')
plt.show()

print("HOG features shape:", features.shape)  # This will show the number of extracted features