import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cupy as cp
from collections import Counter
from scipy import ndimage  # Use scipy.ndimage for convolution
import matplotlib.pyplot as plt
import time
from skimage.feature import hog
from cupyx.scipy import ndimage as cpx_ndimage
from PIL import Image

def get_color_difference_gpu(pixel1, pixel2):
    """
    Returns the Euclidean distance between two RGB colors (GPU accelerated).
    """
    return cp.linalg.norm(pixel1 - pixel2)

def mask_pixels_with_neighbors_gpu(image_rgb, color_threshold=10):
    """
    Masks every pixel in the lower 40% of the image that is no more than 'color_threshold' different from its neighbors
    and the given most_common_color. The masked pixels will be colored in a gradient from light blue (bottom) to dark blue (top).

    takes in rgb
    """

    # Convert image to RGB and upload it to the GPU
    #image_rgb = cp.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_rgb = cp.asarray(image_rgb)
    # Get the height and width of the image
    height, width, _ = image_rgb.shape

    # Focus on the lower 40% of the image
    lower_40_percent = image_rgb[int(height * 0.6):, :]

    # Reshape the image region to a 2D array of RGB values
    pixels = lower_40_percent.reshape(-1, 3)

    # Convert each pixel to a tuple (hashable)
    pixel_tuples = [tuple(pixel.get().astype(int)) for pixel in pixels]
    color_counts = Counter(pixel_tuples)

    # Count the occurrences of each color
    color_counts = Counter(pixel_tuples)

    # Get the most common color
    most_common_color = color_counts.most_common(1)[0][0]

    # Initialize the mask (True for pixels to be kept)
    mask = cp.zeros((height, width), dtype=cp.bool_)

    # Convert most_common_color to a GPU array
    most_common_color_gpu = cp.asarray(most_common_color)

    # Iterate over the lower 40% of the image
    for y in range(int(height * 0.6), height - 1):  # Only lower 40%
        for x in range(1, width - 1):  # Avoid borders to check neighbors
            pixel = image_rgb[y, x]
            
            # Check neighbors (top, bottom, left, right)
            neighbors = [
                image_rgb[y - 1, x],  # Top
                image_rgb[y + 1, x],  # Bottom
                image_rgb[y, x - 1],  # Left
                image_rgb[y, x + 1]   # Right
            ]
            
            # Check if the color difference with the most common color is within the threshold
            if get_color_difference_gpu(pixel, most_common_color_gpu) <= color_threshold:
                # Check if the pixel is similar to its neighbors
                # Convert the list of boolean results to a cupy.ndarray
                similar_to_neighbors = cp.any(cp.asarray([get_color_difference_gpu(pixel, neighbor) <= color_threshold for neighbor in neighbors]))

                
                # If it's similar to neighbors, mask the pixel
                if similar_to_neighbors:
                    mask[y, x] = True

    # Apply the blue gradient to the masked pixels
    for y in range(int(height * 0.6), height):  # Lower 40%
        for x in range(0, width):
            if mask[y, x]:
                # Calculate blue intensity based on vertical position
                blue_intensity = int(255 - (y / height) * 255)  # Darker blue at the top, lighter at the bottom
                image_rgb[y, x] = cp.array([23, 66, blue_intensity], dtype=cp.uint8)  # Set to a blue color gradient
  # Set to a blue color gradient

    # Convert the mask back to CPU and apply it to the original image
    return cp.asnumpy(image_rgb)  # Convert back to CPU for saving

    # Save the result
    #output_path = Path("try")
    #result_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

    #output_path.mkdir(parents=True, exist_ok=True)

    # Save the result image
    #output_file = output_path / f"{image_path.stem}_processed.png"
    #cv2.imwrite(str(output_file), result_bgr)


  # Return the processed image if needed

def compute_rgb_gradient(image):
    # Ensure the input is a CuPy array (GPU processing)
    if isinstance(image, np.ndarray):
        image = cp.asarray(image)
    
    # Separate the RGB channels
    r_channel = image[:,:,0]
    g_channel = image[:,:,1]
    b_channel = image[:,:,2]
    
    # Compute gradients (finite difference)
    grad_r = cp.abs(r_channel[1:, :] - r_channel[:-1, :])  # Gradient along the y-axis (vertical)
    grad_g = cp.abs(g_channel[1:, :] - g_channel[:-1, :])  # Same for G channel
    grad_b = cp.abs(b_channel[1:, :] - b_channel[:-1, :])  # Same for B channel
    
    # Padding the gradient results to match the original image size
    grad_r = cp.pad(grad_r, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    grad_g = cp.pad(grad_g, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    grad_b = cp.pad(grad_b, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    
    # Combine the gradients back into a single image
    gradient_image = cp.stack([grad_r, grad_g, grad_b], axis=-1)
    
    # Normalize the gradients to [0, 255] range for visualization
    gradient_image = (255 * (gradient_image / gradient_image.max())).astype(cp.uint8)

    return gradient_image.get()


def process_image_with_gpu(image_path, output_dir):
    # Load image to CPU
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Move image to GPU
    image_gpu = cp.asarray(image_rgb)
    height, width, _ = image_rgb.shape

    # Focus on the lower 10% of the image
    lower_10_percent = image_rgb[int(height * 0.9):, :]

    # Reshape the image region to a 2D array of RGB values
    pixels = lower_10_percent.reshape(-1, 3)

    # Count the occurrences of each color (using a tuple for RGB values)
    color_counts = Counter(tuple(pixel) for pixel in pixels)

    # Get the most common color
    most_common_color = color_counts.most_common(1)[0][0]
    
    # Perform GPU-based color difference calculation
    most_common_color_gpu = cp.array(most_common_color, dtype=cp.uint8)
    color_diff_gpu = cp.abs(image_gpu - most_common_color_gpu)

    # Create mask based on color difference threshold
    mask_gpu = cp.all(color_diff_gpu <= 20, axis=-1)  # Check all color channels (R, G, B)

    # Apply mask to the lower half of the image
    lower_half_mask = cp.zeros_like(mask_gpu)
    lower_half_mask[height // 2 :, :] = mask_gpu[height // 2 :, :]
    mask_gpu = lower_half_mask  # Final mask applied to the lower half

    # Apply blue-toning across the image using GPU
    blue_tone = cp.zeros_like(image_gpu, dtype=cp.uint8)

    # Ensure that the computation stays within CuPy for blue intensity
    blue_intensity = 255 * cp.exp(-cp.arange(height, dtype=cp.float32) / height)
    blue_intensity = cp.asarray(blue_intensity, dtype=cp.uint8)

    # Create arrays for RGB channels
    zeros_array = cp.zeros_like(blue_intensity, dtype=cp.uint8)
    blue_255_array = cp.full_like(blue_intensity, 255, dtype=cp.uint8)

    # Expand dimensions to create 2D arrays for stacking
    zeros_array_expanded = cp.expand_dims(zeros_array, axis=-1)  # Shape becomes (height, 1)
    blue_255_array_expanded = cp.expand_dims(blue_255_array, axis=-1)  # Shape becomes (height, 1)
    blue_intensity_expanded = cp.expand_dims(blue_intensity, axis=-1)  # Shape becomes (height, 1)

    # Stack the arrays to form the final blue-tinted color
    combined_blue = cp.concatenate([blue_intensity_expanded, zeros_array_expanded, blue_255_array_expanded], axis=-1)

    # Broadcast the combined blue color to match the image shape (height, width, 3)
    combined_blue = cp.broadcast_to(combined_blue, (height, width, 3))

    # Expand the mask to 3 channels (height, width, 3)
    mask_expanded = cp.expand_dims(mask_gpu, axis=-1)
    mask_expanded_broadcast = cp.broadcast_to(mask_expanded, (height, width, 3))

    # Apply the mask to the blue-tinted areas
    blue_tone[mask_expanded_broadcast] = combined_blue[mask_expanded_broadcast]

    # Combine the original image with the blue-toned image
    result_gpu = cp.where(mask_expanded_broadcast, blue_tone, image_gpu)

    # Convert the result back to BGR format
    result_bgr = cp.asnumpy(result_gpu)
    result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2BGR)
    
    return result_bgr


def gpu_sobel_edge_detection(image):
    if isinstance(image, np.ndarray):
        image = cp.asarray(image)

    sobel_y = cp.array([[-1, -2, -1], 
                        [ 0,  0,  0], 
                        [ 1,  2,  1]], dtype=cp.float32)

    sobel_diag_45 = cp.array([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]], dtype=cp.float32)

    sobel_diag_135 = cp.array([[ 1, 0, -1], 
                               [ 2, 0, -2], 
                               [ 1, 0, -1]], dtype=cp.float32)

    r_channel = image[:,:,0]  # Red channel
    g_channel = image[:,:,1]  # Green channel
    b_channel = image[:,:,2]  # Blue channel

    grad_y_r = cpx_ndimage.convolve(r_channel, sobel_y, mode='constant', cval=0.0)
    grad_y_g = cpx_ndimage.convolve(g_channel, sobel_y, mode='constant', cval=0.0)
    grad_y_b = cpx_ndimage.convolve(b_channel, sobel_y, mode='constant', cval=0.0)

    grad_diag_45_r = cpx_ndimage.convolve(r_channel, sobel_diag_45, mode='constant', cval=0.0)
    grad_diag_45_g = cpx_ndimage.convolve(g_channel, sobel_diag_45, mode='constant', cval=0.0)
    grad_diag_45_b = cpx_ndimage.convolve(b_channel, sobel_diag_45, mode='constant', cval=0.0)

    grad_diag_135_r = cpx_ndimage.convolve(r_channel, sobel_diag_135, mode='constant', cval=0.0)
    grad_diag_135_g = cpx_ndimage.convolve(g_channel, sobel_diag_135, mode='constant', cval=0.0)
    grad_diag_135_b = cpx_ndimage.convolve(b_channel, sobel_diag_135, mode='constant', cval=0.0)

    # Combine the gradients from all three channels for each direction
    grad_y = cp.sqrt(grad_y_r**2 + grad_y_g**2 + grad_y_b**2)
    grad_diag_45 = cp.sqrt(grad_diag_45_r**2 + grad_diag_45_g**2 + grad_diag_45_b**2)
    grad_diag_135 = cp.sqrt(grad_diag_135_r**2 + grad_diag_135_g**2 + grad_diag_135_b**2)

    # Combine the results for horizontal and diagonal gradients
    grad_magnitude = cp.sqrt(grad_y**2 + grad_diag_45**2 + grad_diag_135**2)
    edges = grad_magnitude.get()
    edges_normalized = (255 * (edges / edges.max())).astype(np.uint8)
    edges_rgb = np.stack([edges_normalized] * 3, axis=-1)
    result = cv2.cvtColor(edges_rgb, cv2.COLOR_RGB2GRAY)
    return result

def detect_floor_with_gpu(image_path, output_folder, edge_threshold=0.5):
    color_range=((25, 25, 25), (120, 120, 120))
    max_color_diff=20
    """
    Detects the floor in an image by applying edge detection and masking.
    The floor area will be highlighted, and the rest will be black.
    """
    # Load the image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert image to RGB and upload it to the GPU
    image_rgb = cp.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Get the height and width of the image
    height, width, _ = image_rgb.shape

    # Focus on the lower 40% of the image
    lower_40_percent = image_rgb[int(height * 0.6):, :]

    # Perform Sobel edge detection
    min_color = cp.array(color_range[0], dtype=cp.float32)
    max_color = cp.array(color_range[1], dtype=cp.float32)

    # Compute the absolute difference between R, G, and B channels
    r_channel = lower_40_percent[:,:,0]
    g_channel = lower_40_percent[:,:,1]
    b_channel = lower_40_percent[:,:,2]

    # Calculate the absolute differences between R, G, and B channels
    diff_rg = cp.abs(r_channel - g_channel)
    diff_rb = cp.abs(r_channel - b_channel)
    diff_gb = cp.abs(g_channel - b_channel)

    # Create a mask for pixels where all channel differences are within the max_color_diff
    color_diff_mask = (diff_rg < max_color_diff) & (diff_rb < max_color_diff) & (diff_gb < max_color_diff)

    # Create a mask that selects pixels within the specified color range
    color_range_mask = cp.all((lower_40_percent >= min_color) & (lower_40_percent <= max_color), axis=-1)

    # Combine the color range mask and the color difference mask
    combined_mask = color_range_mask & color_diff_mask

    # Apply the combined mask to the lower 40% region of the image
    lower_40_percent_color_filtered = lower_40_percent[combined_mask]

    # Compute the magnitude of the gradients
    #grad_magnitude = gpu_sobel_edge_detection(lower_40_percent_color_filtered)

    # Threshold to identify edges
    #floor_edge_mask = grad_magnitude > edge_threshold

    # Focus on the region of the image where the floor is more likely to be
    # In the lower portion of the image (where the floor usually appears)
    floor_mask = cp.zeros((height, width), dtype=cp.bool_)
    lower_40_percent_grayscale = cp.mean(lower_40_percent_color_filtered, axis=2)

# Now apply the condition to the grayscale image
    lower_40_percent_filtered_condition = lower_40_percent_grayscale < 10

# Resize the filtered result to match the shape of floor_mask if necessary
    lower_40_percent_filtered_condition_resized = lower_40_percent_filtered_condition[:floor_mask.shape[0] - int(height * 0.6), :]

# Assign to the floor_mask
    floor_mask[int(height * 0.6):, :] = lower_40_percent_filtered_condition_resized

    # Create a result image initialized to the original image
    detected_floor_image = cp.copy(image_rgb)

    # Apply the mask: Set non-floor pixels to black
    detected_floor_image[~floor_mask] = [0, 0, 0]

    # Convert the result to CPU for saving
    detected_floor_image_cpu = cp.asnumpy(detected_floor_image)

    # Save the result image
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{image_path.stem}_processed.png"
    result_bgr = cv2.cvtColor(detected_floor_image_cpu, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_file), result_bgr)


def apply_blue_tone_to_filtered_image(image_rgb):
    """
    Applies blue tone effect to a filtered region of the image based on color similarity and intensity decay.

    Parameters:
        image_rgb (cp.ndarray): The input image on GPU.
        color_range (tuple): A tuple containing the min and max color ranges for filtering.
        max_color_diff (float): The maximum allowed color difference for filtering.
        most_common_color (tuple): The most common color in the lower 10% region for creating a mask.

    Returns:
        cp.ndarray: The image with the blue-toned effect applied.
    """
    image_rgb = cp.asarray(image_rgb)
    image_gpu = image_rgb

    color_range=((60, 60, 60), (110, 110, 100))
    max_color_diff=10

    height, width, _ = image_rgb.shape

    # Step 1: Focus on the lower 40% of the image
    lower_40_percent = image_rgb[int(height * 0.6):, :]
    loheight, lowidth, _ = lower_40_percent.shape

    # Step 2: Perform color range and difference filtering
    min_color = cp.array(color_range[0], dtype=cp.float32)
    max_color = cp.array(color_range[1], dtype=cp.float32)

    # Compute absolute differences between channels
    r_channel = lower_40_percent[:, :, 0]
    g_channel = lower_40_percent[:, :, 1]
    b_channel = lower_40_percent[:, :, 2]

    # Calculate differences between RGB channels
    diff_rg = cp.abs(r_channel - g_channel)
    diff_rb = cp.abs(r_channel - b_channel)
    diff_gb = cp.abs(g_channel - b_channel)

    # Mask based on channel differences
    color_diff_mask = (diff_rg < max_color_diff) & (diff_rb < max_color_diff) & (diff_gb < max_color_diff)

    # Mask based on color range
    color_range_mask = cp.all((lower_40_percent >= min_color) & (lower_40_percent <= max_color), axis=-1)

    # Combine both masks
    mask_gpu = color_range_mask 
    #& color_diff_mask

    # Step 6: Apply blue-toning across the image using GPU
    
    blue_height = image_rgb.shape[0] - loheight
    lower_half_mask = cp.zeros((300, 300), dtype=cp.bool_)
    lower_half_mask[blue_height:, :] = mask_gpu

    mask_gpu = lower_half_mask  # Final mask applied to the lower half

    # Apply blue-toning across the image using GPU
    blue_tone = cp.zeros_like(image_gpu, dtype=cp.uint8)

    # Ensure that the computation stays within CuPy for blue intensity
    blue_intensity = 255 * cp.exp(-cp.arange(height, dtype=cp.float32) / height)
    blue_intensity = cp.asarray(blue_intensity, dtype=cp.uint8)

    # Create arrays for RGB channels
    zeros_array = cp.zeros_like(blue_intensity, dtype=cp.uint8)
    blue_255_array = cp.full_like(blue_intensity, 255, dtype=cp.uint8)

    # Expand dimensions to create 2D arrays for stacking
    zeros_array_expanded = cp.expand_dims(zeros_array, axis=-1)  # Shape becomes (height, 1)
    blue_255_array_expanded = cp.expand_dims(blue_255_array, axis=-1)  # Shape becomes (height, 1)
    blue_intensity_expanded = cp.expand_dims(blue_intensity, axis=-1)  # Shape becomes (height, 1)

    # Stack the arrays to form the final blue-tinted color
    combined_blue = cp.concatenate([blue_intensity_expanded, zeros_array_expanded, blue_255_array_expanded], axis=-1)

    # Broadcast the combined blue color to match the image shape (height, width, 3)
    combined_blue = cp.broadcast_to(combined_blue, (height, width, 3))

    # Expand the mask to 3 channels (height, width, 3)
    mask_expanded = cp.expand_dims(mask_gpu, axis=-1)
    mask_expanded_broadcast = cp.broadcast_to(mask_expanded, (height, width, 3))

    # Apply the mask to the blue-tinted areas
    blue_tone[mask_expanded_broadcast] = combined_blue[mask_expanded_broadcast]

    # Combine the original image with the blue-toned image
    result_gpu = cp.where(mask_expanded_broadcast, blue_tone, image_gpu)

    # Convert the result back to BGR format
    result_bgr = cp.asnumpy(result_gpu)
    result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2BGR)

    """
    
    lower_half_mask_expanded = cp.pad(lower_half_mask, ((0, blue_height - lower_half_mask.shape[0]), (0, 0)), mode='constant')

# Step 2: Broadcast across color channels
    mask_gpu = cp.broadcast_to(lower_half_mask_expanded[:, :, cp.newaxis], (blue_height, 300, 3))

    blue_tone = cp.zeros_like(image_rgb, dtype=cp.uint8)

    # Ensure that the computation stays within CuPy for blue intensity
    # Calculate blue intensity curve for lower part

    blue_intensity = 255 * cp.exp(-cp.arange(blue_height, dtype=cp.float32) / blue_height)
    blue_intensity = cp.asarray(blue_intensity, dtype=cp.uint8)

    # Create RGB blue gradient: (blue_intensity, 0, 255)
    zeros_array = cp.zeros_like(blue_intensity, dtype=cp.uint8)
    blue_255_array = cp.full_like(blue_intensity, 255, dtype=cp.uint8)

    # Shape (blue_height, 1)
    blue_intensity_exp = cp.expand_dims(blue_intensity, axis=-1)
    zeros_exp = cp.expand_dims(zeros_array, axis=-1)
    blue_255_exp = cp.expand_dims(blue_255_array, axis=-1)

    # Stack to RGB, shape (blue_height, 3), then broadcast to full width
    combined_blue = cp.concatenate([blue_intensity_exp, zeros_exp, blue_255_exp], axis=-1)  # (180, 3)
    combined_blue = combined_blue[:, cp.newaxis, :]  # Now shape is (180, 1, 3)
    combined_blue = cp.broadcast_to(combined_blue, (blue_height, image_rgb.shape[1], 3))

    # Prepare the mask
    mask_broadcast = cp.broadcast_to(lower_half_mask_expanded[:, :, cp.newaxis], (blue_height, image_rgb.shape[1], 3))

    # Prepare blue_tone layer
    blue_tone = cp.zeros_like(image_rgb, dtype=cp.uint8)
    blue_tone[loheight:, :][mask_broadcast] = combined_blue[mask_broadcast]

    # Apply to image
    result_gpu = cp.where(mask_broadcast, blue_tone[loheight:, :], image_rgb[loheight:, :])

    # Stitch final result into full image
    final_result = image_rgb.copy()
    final_result[loheight:, :] = result_gpu


    # Convert the result back to BGR format
    result_bgr = cp.asnumpy(result_gpu)
    result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2BGR)



        output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{image_path.stem}_processed.png"
    cv2.imwrite(str(output_file), result_bgr)
    """
    
    return result_bgr


def process_and_color_floor_3(image_path, output_dir, save_output=1, floor_color=(255, 0, 255), min_neighbors=800):
    """
    Segments pixels near (30, 30, 30) to (60, 60, 60) RGB and assigns a color based on perspective.
    Ensures that each selected pixel has at least 100 neighboring pixels within the range.
    """
    # Load image (RGB)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    # Define RGB range for detection (30, 30, 30) to (60, 60, 60)
    lower_rgb = np.array([25, 25, 25])
    upper_rgb = np.array([120, 120, 120])

    # Create binary mask based on RGB range
    mask = np.all(np.logical_and(image_rgb >= lower_rgb, image_rgb <= upper_rgb), axis=-1)

    mask = mask & (np.abs(image_rgb[:, :, 0] - image_rgb[:, :, 2]) <= 20)

    lower_half_mask = np.zeros_like(mask)
    lower_half_mask[height // 2 :, :] = mask[height // 2 :, :]

    # Perform connected components analysis to find connected regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lower_half_mask.astype(np.uint8), connectivity=8)

    result = image_rgb.copy()
    
    for label in range(1, num_labels):  # label 0 is the background
        # Count number of pixels in the component
        component_size = stats[label, cv2.CC_STAT_AREA]

        # If the component has less than min_neighbors pixels, discard it
        if component_size < min_neighbors:
            lower_half_mask[labels == label] = 0
        else:
            # Color the selected pixels with blue tones based on the pixel's y-coordinate (height)
            for y in range(height):
                for x in range(width):
                    if labels[y, x] == label:
                        # Linearly interpolate blue intensity based on y (height)
                        blue_intensity = int(255 * np.exp(-y / height))  # Darker blue as height increases
                        result[y, x] = [blue_intensity, 0, 255]  # Set to blue tones (B, G, R)
    
    # Save result if enabled
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_colored.png"
        out_path = os.path.join(output_dir, filename)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, result_bgr)

    return result


def adjust_brightness_to_mean(image_np, target_mean=128.0):
  
    # Load image using OpenCV (BGR format, uint8)
    # Convert to float32 and move to GPU
    #image_cp = cp.asarray(image_np, dtype=cp.float32)

    # Split into B, G, R channels
    channels = np.split(image_np, indices_or_sections=3, axis=2)

    # Adjust each channel independently
    adjusted_channels = []
    for ch in channels:
        mean_val = np.mean(ch)
        shift = target_mean - mean_val
        ch_adjusted = np.clip(ch + shift, 0, 255)
        adjusted_channels.append(ch_adjusted)

    # Merge channels back together
    adjusted_cp = np.concatenate(adjusted_channels, axis=2).astype(np.uint8)
    #adjusted_np = cp.asnumpy(adjusted_cp)

    #result_bgr = cv2.cvtColor(adjusted_cp, cv2.COLOR_RGB2BGR)

    return adjusted_cp


def adjust_image_using_floor_reference(image_path, output_folder, target_floor_color=(80, 80, 80), floor_pct=0.2):
    """
    Adjusts the image color using the average color of the floor (bottom portion of the image).

    Parameters:
        image_path (str or Path): Path to the image.
        output_folder (str or Path): Folder to save the processed image.
        target_floor_color (tuple): Expected BGR color of the floor (default: (50, 50, 50)).
        floor_pct (float): Percentage of image height to consider as floor (default: 0.2).

    Returns:
        numpy.ndarray: Adjusted image as NumPy array (dtype=uint8).
    """
    image_np = cv2.imread(str(image_path))
    if image_np is None:
        raise ValueError(f"Could not load image from {image_path}")

    h, w, _ = image_np.shape
    floor_start = int(h * (1 - floor_pct))
    floor_region = image_np[floor_start:, :, :]

    image_cp = cp.asarray(image_np, dtype=cp.float32)
    floor_cp = cp.asarray(floor_region, dtype=cp.float32)

    floor_mean = cp.mean(floor_cp, axis=(0, 1))  # Shape: (3,)
    target_cp = cp.array(target_floor_color, dtype=cp.float32)

    shift = target_cp - floor_mean  # Shape: (3,)
    shifted_cp = image_cp + shift[cp.newaxis, cp.newaxis, :]

    adjusted_cp = cp.clip(shifted_cp, 0, 255).astype(cp.uint8)
    adjusted_np = cp.asnumpy(adjusted_cp)

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{image_path.stem}.png"
    cv2.imwrite(str(output_file), adjusted_np)

    return adjusted_np

def apply_blue_tone_and_extract_feature(image_rgb, save = False, output_folder = None, image_path = None):
    """
    Applies blue tone effect to an image and extracts the blue intensity map as a feature for regression.
    
    Parameters:
        image_path (Path): Path to the input image.
        output_folder (Path): Path to save the processed image.
    
    Returns:
        blue_intensity_map (cp.ndarray): The blue intensity map used as a feature.
        result_bgr (np.ndarray): The image with blue-toning effect applied.
    
    image = rgb
    """
    # Load the image in BGR format
    #image_rgb = cp.asarray(image_rgb)
    height, width, _ = image_rgb.shape
    lo_start = int(height * 0.6)
    floor_region = image_rgb[lo_start:, :, :]  # Bottom 40% of the image

    # Convert the floor region to LAB on CPU
    lab = cv2.cvtColor(floor_region, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    #print(A, B)
    #time.sleep(10)
    # Define grey-ish LAB mask (dark, low chroma)
    lab_mask = (
        (A > 100) & (A < 140) &  # a ≈ 128
        (B > 100) & (B < 140) &  # b ≈ 128
        (L > 25) & (L < 90)      # L in dark/mid range
    ).astype(np.uint8)

    # Convert mask to full image height
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[lo_start:, :] = lab_mask

    # Send full mask and image to GPU
    mask_gpu = cp.asarray(full_mask, dtype=cp.bool_)

    # Create blue intensity map (linearly fading from bottom to top)
    decay_rate = 5.0  # You can tweak this value to control the sharpness of the decay
    x = cp.linspace(1, 0, height)
    blue_intensity = (255 * cp.exp(-decay_rate * x)).astype(cp.uint8)

    # Create blue overlay: (R, G, B) = (intensity, 0, 255)
    blue_overlay = cp.stack([
        blue_intensity[:, None].repeat(width, axis=1),         # R channel
        cp.zeros((height, width), dtype=cp.uint8),             # G channel
        cp.full((height, width), 255, dtype=cp.uint8)          # B channel
    ], axis=2)

    # Expand mask to 3 channels
    mask_broadcast = cp.broadcast_to(mask_gpu[..., None], (height, width, 3))
    image_rgb = cp.asarray(image_rgb)
    # Apply blue overlay where mask is true
    result_gpu = cp.where(mask_broadcast, blue_overlay, image_rgb)

    # Convert back to BGR for the final image
    result_bgr = cp.asnumpy(result_gpu)
    #result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2BGR)

    # Save the result image with the blue-toning effect
    if save:
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{image_path.stem}_blue.png"
        cv2.imwrite(str(output_file), result_bgr)

    return result_bgr

def detect_floor_region(image_rgb):

    new_img = image_rgb
    kvar = 3

    # Calculate the gradient magnitude for each channel (R, G, B)
    grad_x_r = cv2.Sobel(image_rgb[:, :, 0], cv2.CV_64F, 1, 0, ksize=kvar)
    grad_y_r = cv2.Sobel(image_rgb[:, :, 0], cv2.CV_64F, 0, 1, ksize=kvar)
    grad_x_g = cv2.Sobel(image_rgb[:, :, 1], cv2.CV_64F, 1, 0, ksize=kvar)
    grad_y_g = cv2.Sobel(image_rgb[:, :, 1], cv2.CV_64F, 0, 1, ksize=kvar)
    grad_x_b = cv2.Sobel(image_rgb[:, :, 2], cv2.CV_64F, 1, 0, ksize=kvar)
    grad_y_b = cv2.Sobel(image_rgb[:, :, 2], cv2.CV_64F, 0, 1, ksize=kvar)

    wol = grad_x_r + grad_y_r + grad_x_g + grad_y_g + grad_x_b + grad_y_b


    # Compute the gradient magnitudes for each channel
    
    wol_normalized = cv2.normalize(wol, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 for displaying in plt.imshow
    wol_display = np.round(wol_normalized.astype(np.uint8), 4)
    wol_rgb = cv2.cvtColor(wol_display, cv2.COLOR_GRAY2RGB)

    return wol_rgb


def doandmask(image):
    result = image.copy()
    lol = detect_floor_region(image)
    lol = adjust_brightness_to_mean(lol)
    lol = np.where(lol > 120, 1, 0).astype(np.uint8)
    wol_gray = cv2.cvtColor(lol, cv2.COLOR_RGB2GRAY)
    #print(lol.shape)
    mask_indices = (wol_gray != 1)
    result[mask_indices, 0] = 0    # Red channel
    result[mask_indices, 1] = 0    # Green channel
    result[mask_indices, 2] = 255  # Blue channel
    return result

def hog_area_old(image, areainf = True, hogo = True):
    #image = adjust_brightness_to_mean(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    max_areas = 2
    if areainf:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas) if areas else 0
        mean_area = np.mean(areas) if areas else 0

        if len(areas) < max_areas:
            areas += [0] * (max_areas - len(areas))
        else:
            areas = areas[:max_areas]


    if hogo:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features, _ = hog(
    gray_image,
    orientations=9,  #9
    pixels_per_cell=(15,15),  #15,15
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
        )
    if areainf & hogo:
     return np.concatenate([[np.mean(gray),np.std(gray)],areas, hog_features])
    elif hogo:
        return hog_features
    else:
        return [max_area, mean_area]


def hog_area(image, areainf=True, hogo=True, max_areas=6):
    gray = image.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    features = [np.mean(gray), np.std(gray)]

    contour_features = []

    if areainf:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (descending), then keep top N
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_areas]

        for c in contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"] / w  # normalize x
                cy = M["m01"] / M["m00"] / h  # normalize y
            else:
                cx, cy = 0, 0

            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / bh if bh != 0 else 0

            contour_features.extend([area, cx, cy, aspect_ratio])

        # Pad to fixed length
        while len(contour_features) < 4 * max_areas:
            contour_features.extend([0, 0, 0, 0])
    else:
        contour_features = [0] * (4 * max_areas)

    hog_features = []
    if hogo:
        hog_features, _ = hog(
            gray,
            orientations=9,
            pixels_per_cell=(15, 15),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
    return np.concatenate([features, contour_features, hog_features])


def meta_finder(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    means = np.mean(image, axis=(0, 1))  # Mean across height and width
    variances = np.var(image, axis=(0, 1))  # Variance across height and width
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    mean_area = np.mean(areas) if areas else 0
    floor_region = doandmask(image)
    gray_floor = cv2.cvtColor(floor_region, cv2.COLOR_RGB2GRAY)
    gray_floor = np.array(gray_floor.resize((10,10))).reshape(-1)

    return np.concatenate([gray_floor, means, variances, [max_area, mean_area]])



def process_folder(input_dir, output_folder):
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() == '.png']
    progressbar = tqdm(total=len(image_paths), desc="Processing images")
    for image_path in image_paths:
        #process_image_with_gpu(image_path,Path("worked_gpu3"))
        #apply_blue_tone_to_filtered_image(image_path,output_folder)
        apply_blue_tone_and_extract_feature(image_path,output_folder)
        #adjust_brightness_to_mean(image_path,output_folder)
        #adjust_image_using_floor_reference(image_path,output_folder)
        #detect_floor_region(image_path,output_folder)
        """process_and_color_floor_3(
            image_path=image_path,
            output_dir=output_folder,
            save_output=1,
            floor_color=floor_color
        )"""
        progressbar.update(1)
    progressbar.close()

#input_dir = Path("data/train_images")

#output_dir = Path("worked_data_blue/train_images")
#process_folder(input_dir, output_dir)



"""
def process_and_color_floor(image_path, output_dir, save_output=1, floor_color=(60, 60, 60)):

    Segments floor and assigns it a distinct RGB color (default: magenta).
    If save_output == 1, saves the result to output_dir.

    # Load image (RGB)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define floor HSV range — tune these as needed
    lower_floor = np.array([0, 0, 40])     # Example: low saturation, medium brightness
    upper_floor = np.array([180, 60, 255]) # Example: light-colored floor

    # Create binary mask for floor
    mask = cv2.inRange(hsv, lower_floor, upper_floor)

    # Create colored overlay image
    colored_floor = np.full_like(image_rgb, floor_color, dtype=np.uint8)

    # Combine with original image using mask
    result = image_rgb.copy()
    result[mask > 0] = colored_floor[mask > 0]

    # Save result if enabled
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_colored.png"
        out_path = os.path.join(output_dir, filename)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, result_bgr)

    return result

def process_and_color_floor(image_path, output_dir, save_output=1, min_neighbors=100):

    Segments pixels near the mean color of the bottom half and assigns blue tones
    based on the height of the pixel, using a non-linear scaling for perspective.
    Ensures that color differentiation and spatial proximity are considered.
    Only processes the lower half of the image.

    # Load image (RGB)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    # Define the bottom half of the image
    bottom_half = image_rgb[height // 2 :, :]

    # Compute the mean color of the bottom half of the image
    mean_color = np.mean(bottom_half, axis=(0, 1))

    # Create a mask based on the difference between each pixel and the mean color
    color_diff_threshold = 10  # Allow for some color variation within 10 units

    # Create an empty mask
    mask = np.zeros_like(image_rgb[:, :, 0], dtype=bool)

    for y in range(height // 2, height):  # Only process the bottom half
        for x in range(width):
            pixel = image_rgb[y, x]
            # Calculate color differences between the pixel and the mean color
            color_diff = np.abs(pixel - mean_color)
            if np.all(color_diff <= color_diff_threshold):
                mask[y, x] = True

    # Now apply the differentiation rule based on neighbors
    # Calculate the difference between neighboring pixels and ensure it’s small
    spatial_diff_threshold = 10  # Max difference between neighbors' color
    refined_mask = np.zeros_like(mask)

    for y in range(1, height):  # Iterate over all pixels, skip first row (no top neighbor)
        for x in range(1, width):  # Skip first column (no left neighbor)
            if mask[y, x]:
                # Calculate difference with the top and left neighbors
                top_diff = np.abs(image_rgb[y, x] - image_rgb[y - 1, x])
                left_diff = np.abs(image_rgb[y, x] - image_rgb[y, x - 1])
                
                # Check if the color difference with both neighbors is within the spatial threshold
                if np.all(top_diff <= spatial_diff_threshold) and np.all(left_diff <= spatial_diff_threshold):
                    refined_mask[y, x] = True

    # Create the result image with blue tones based on height (y-coordinate)
    result = image_rgb.copy()

    for y in range(height):
        for x in range(width):
            if refined_mask[y, x]:
                # Apply exponential decay based on y (height) to mimic perspective
                blue_intensity = int(255 * np.exp(-y / height))  # Exponential decay function
                result[y, x] = [blue_intensity, 0, 255]  # Set to blue tones (B, G, R)

    # Save result if enabled
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_colored_blue.png"
        out_path = os.path.join(output_dir, filename)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, result_bgr)

    return result
"""
