import cv2
import numpy as np
from scipy.optimize import minimize

def transform_image(image, angle, scale, skew):
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Define the rotation matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += skew[0]
    M[1, 2] += skew[1]

    # Apply the transformation
    return cv2.warpAffine(image, M, (w, h))

def calculate_error(params, target_image, template):
    # Transform the image
    transformed_image = transform_image(target_image, params[0], params[1], params[2:])

    # Resize the images to match the smallest dimensions
    h = min(template.shape[0], transformed_image.shape[0])
    w = min(template.shape[1], transformed_image.shape[1])
    template_resized = cv2.resize(template, (w, h))
    transformed_image_resized = cv2.resize(transformed_image, (w, h))

    # Calculate the absolute difference between the template and the transformed image
    diff = cv2.absdiff(template_resized, transformed_image_resized)

    # Return the mean error
    return np.mean(diff)

def align_images(target_image, template):
    # Initialize the parameters
    params = [1.8, 1, -2.6, 2.6]

    # Minimize the error
    result = minimize(calculate_error, params, args=(target_image, template), method='BFGS')

    # Print the final parameters and error
    print('Final parameters:', result.x)
    print('Final error:', result.fun)

    # Return the best parameters
    return result.x

# Load the images
target_image = cv2.imread('data/Training_Cropped/7/7-22.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('archery_target_thicker_lines_2048x2048.png', cv2.IMREAD_GRAYSCALE)

target_image_blurred = cv2.GaussianBlur(target_image, (5, 5), 3)
# Apply Canny edge detection to the images
target_image_edges = 1-cv2.Canny(target_image_blurred, 150, 400)

# Align the images
params = align_images(target_image_edges, template)

# Transform the target image
aligned_image = transform_image(target_image, params[0], params[1], params[2:])

# Display the aligned image
cv2.imshow('Aligned Image', aligned_image)
cv2.waitKey(0)

# Save the aligned image
cv2.imwrite('aligned_image.png', aligned_image)