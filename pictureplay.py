import cv2
import numpy as np
from picturework import adjust_brightness_to_mean,mask_pixels_with_neighbors_gpu, apply_blue_tone_and_extract_feature, hog_area, detect_floor_region, gpu_sobel_edge_detection, compute_rgb_gradient,doandmask
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specify the folder containing images
folder_path = 'data/test_images'  # Replace with your folder path

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Loop through each image file and display it
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image_bgr = cv2.imread(str(image_path))
    lol = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #lol = adjust_brightness_to_mean(lol)
    #lol = apply_blue_tone_and_extract_feature(lol)
    lol = doandmask(lol)
    #lol = detect_floor_region(lol)
    
    #lol = np.where(lol > 120, 255, 0).astype(np.uint8)
    lol = cv2.resize(lol, (30, 30), interpolation=cv2.INTER_AREA)
    
    
    
    plt.imshow(lol
    #, cmap='gray'
    )
    plt.title(image_file)
    plt.axis('off')  # Hide axis
    plt.show()