import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

class ImagePreprocessor:
    """
    This class is responsible for preprocessing the input image to prepare it for further analysis.
    This includes noise reduction, edge detection, and perspective correction.
    """
    
    def __init__(self, image_path=None, image=None):
        """
        Initialize the ImagePreprocessor with the path to the image.
        """
        if image_path:
            self.image_path = image_path
            self.original_image = self.load_image()
        else:
            self.image_path = ""
            self.original_image = image

    def show_image(self, image, title="Image", wait_key_time=0, scale=0.75):
        """
        Display the image in a window with optional scaling.
        :param image: The image to be displayed.
        :param title: The title of the window.
        :param wait_key_time: Time in milliseconds to wait for a key event. 0 means wait indefinitely.
        :param scale: Scaling factor for the image size.
        """
        if scale != 1.0:
            # Calculate the new dimensions based on the scaling factor
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            # Resize the image using the new dimensions
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image

        cv2.imshow(title, scaled_image)
        cv2.waitKey(wait_key_time)
        cv2.destroyAllWindows()
    
    def load_image(self):
        """
        Load the image from the given path, convert it to grayscale, and scale it down to fit within 1024x1024.
        """
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image at {self.image_path} cannot be loaded.")
        
        # Calculate the scaling factor to fit the image within 1024x1024
        max_dimension = max(image.shape[:2])
        scale_factor = 1024 / max_dimension if max_dimension > 1024 else 1
        
        # Resize the image if necessary
        if scale_factor != 1:
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        


        return grayscale_image

    
    def apply_gaussian_blur(self, kernel_size=(5, 5), sigma=3):
        """
        Apply Gaussian blur to the image to reduce noise.
        :param kernel_size: Size of the Gaussian kernel.
        :param sigma_x: Gaussian kernel standard deviation in X direction.
        """
        blurred_image = cv2.GaussianBlur(self.original_image, kernel_size, sigmaX=sigma, sigmaY=sigma)
        return blurred_image
    
    def detect_edges(self, image, low_threshold=100, high_threshold=300):
        """
        Use Canny edge detection to find edges in the image.
        :param low_threshold: Lower bound for the hysteresis thresholding.
        :param high_threshold: Upper bound for the hysteresis thresholding.
        """
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges
    

    def detect_ellipses(self, edges, min_contour_size=1200):
        """
        Detect ellipses in the image using contour detection and ellipse fitting.
        :param edges: Edge-detected image.
        :param min_contour_size: Minimum size of the contour to be considered for ellipse fitting.
        :return: A list of ellipses with parameters (center, axes, orientation).
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ellipses = []
        for contour in contours:
            if len(contour) >= min_contour_size:
                ellipse = cv2.fitEllipse(contour)
                ellipses.append(ellipse)
        return ellipses

    def estimate_ellipse_to_circle_transformation(self, ellipse):
        """
        Estimate the perspective transformation matrix to correct an ellipse to a circle.
        :param ellipse: The parameters of the ellipse (center, axes, orientation).
        :return: A 3x3 perspective transformation matrix.
        """
        (xc, yc), (d1, d2), angle = ellipse
        if d1 < d2:  # Ensure d1 is always the major axis
            d1, d2 = d2, d1
            angle += 90.0

        # The scaling factors for x and y axes
        scale_x = d2 / d1
        scale_y = 1.0  # No scaling on the minor axis

        # Calculate the rotation needed to align the major axis with the x-axis
        rotation_matrix = cv2.getRotationMatrix2D((xc, yc), angle, 1.0)
        # Convert to a full 3x3 matrix
        rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])

        # Calculate the inverse rotation needed to revert the alignment
        inv_rotation_matrix = cv2.getRotationMatrix2D((xc, yc), -angle, 1.0)
        inv_rotation_matrix = np.vstack([inv_rotation_matrix, [0, 0, 1]])

        # Calculate the scaling matrix to scale the major axis
        scale_matrix = np.array([[scale_x, 0, xc * (1 - scale_x)],
                                [0, scale_y, 0],
                                [0, 0, 1]], dtype=np.float32)

        # Combine the transformations: first rotate, then scale, then rotate back
        transformation = np.dot(inv_rotation_matrix, np.dot(scale_matrix, rotation_matrix))

        return transformation
    
    def ellipse_fits_in_image(self, ellipse, image_shape):
        """
        Check if the given ellipse fits entirely within the image boundaries.
        :param ellipse: The parameters of the ellipse (center, axes, orientation).
        :param image_shape: The shape of the image (height, width).
        :return: True if the ellipse fits inside the image, False otherwise.
        """
        (xc, yc), (d1, d2), angle = ellipse
        height, width = image_shape[:2]
        # Calculate the bounding box of the ellipse
        bbox_x0 = xc - d1/2
        bbox_x1 = xc + d1/2
        bbox_y0 = yc - d2/2
        bbox_y1 = yc + d2/2
        # Check if the bounding box fits within the image boundaries
        return (bbox_x0 >= 0 and bbox_x1 < width and bbox_y0 >= 0 and bbox_y1 < height)
    
    def detect_and_correct_ovals(self, canny_threshold1=50, canny_threshold2=200, max_ellipse_axis_ratio=1.01):
        """
        Detect ovals in the image and correct the perspective to make them circles.
        :param canny_threshold1: Lower threshold for the Canny edge detector.
        :param canny_threshold2: Upper threshold for the Canny edge detector.
        :param max_ellipse_axis_ratio: The maximum ratio between the major and minor axis to consider an ellipse as an oval needing correction.
        """

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(self.original_image)

        # Blur image 
        blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 2)
        
        # Detect edges
        edges = cv2.Canny(blurred_image, canny_threshold1, canny_threshold2)
        
        # Detect ellipses using Hough Ellipse Transform or other ellipse-fitting techniques
        ellipses = self.detect_ellipses(edges)
        
        # Filter out ellipses that don't fit entirely inside the image
        valid_ellipses = [e for e in ellipses if self.ellipse_fits_in_image(e, self.original_image.shape)]

        # Find the largest ellipse based on the area (product of the axes lengths)
        # that fits entirely inside the image
        if valid_ellipses:
            main_ellipse = max(valid_ellipses, key=lambda e: e[1][0] * e[1][1])
        else:
            raise ValueError("No valid ellipses found that fit within the image boundaries.")
        
        # Check if the detected ellipse is actually an oval (axis ratio exceeds threshold)
        axis_ratio = max(main_ellipse[1]) / min(main_ellipse[1])
        if axis_ratio <= max_ellipse_axis_ratio:
            # The shape is close enough to a circle, no correction needed
            return self.original_image
        
        # Estimate the transformation to correct the oval to a circle
        transformation = self.estimate_ellipse_to_circle_transformation(main_ellipse)
        
        # Apply the transformation to the image
        corrected_image = cv2.warpPerspective(self.original_image, transformation, (self.original_image.shape[1], self.original_image.shape[0]))
        
        return corrected_image
    